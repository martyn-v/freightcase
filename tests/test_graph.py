import json
from pathlib import Path
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from freightcase.execution import StubToolExecutor
from freightcase.graph import build_graph

FIXTURES = Path(__file__).parent / "fixtures" / "emails"
executor = StubToolExecutor()
# One checkpointed graph for the module; per-test thread_ids keep runs isolated.
graph = build_graph(executor=executor, checkpointer=InMemorySaver())


def run_to_pause(config: RunnableConfig) -> dict:
    """Invoke the graph on the road-freight fixture up to the HITL gate."""
    eml = FIXTURES / "quote_road_plain_es.eml"
    return graph.invoke({"eml_file_path": str(eml)}, config=config)


def test_graph_pauses_for_confirmation_then_confirms():
    """Pins the full loop: intake -> classify -> extract -> pause at confirm
    with the standardized payload -> resume approved -> status flips.
    Extraction quality is pinned in test_extraction.py; assertions here
    stick to wiring and unambiguous fields.
    """
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    # Phase 1: run until the HITL gate.
    paused = run_to_pause(config)

    (pause,) = paused["__interrupt__"]
    payload = pause.value
    assert payload["action"] == {"tool": "create_quote", "function": "quote_request"}
    assert "8400 kg" in payload["summary"]
    # The email states city names but no LOCODEs; road mode requires dims.
    assert payload["missing"] == [
        "origin.locode",
        "destination.locode",
        "cargo.0.dimensions",
    ]

    # Phase 2: human approves, filling the dimensions gap ('confirmed'
    # implies complete: approving with gaps would bounce back).
    done = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {
                    "origin.locode": "COBOG",
                    "destination.locode": "COMDE",
                    "cargo.0.dimensions": {
                        "length": 120,
                        "width": 100,
                        "height": 180,
                        "unit": "cm",
                    },
                },
            }
        ),
        config=config,
    )

    intake = done["intake"]
    assert intake is not None
    assert intake.body_text.startswith("Buenas tardes")

    current = done["results"][0]
    assert current.function == "quote_request"
    assert current.status == "executed"
    assert current.execution_ref == "Q-STUB-1"

    assert len(executor.calls) == 1
    assert executor.calls[0][0] == "create_quote"
    sent = executor.calls[0][1]
    assert sent["cargo"][0]["weight"]["kg"] == 8400  # canonicals reach the TMS
    assert sent["origin"]["locode"] == "COBOG"  # human edit reached the TMS

    assert current.missing == []

    extraction = current.output
    assert extraction is not None
    assert extraction.mode == "road"
    assert extraction.incoterm is not None
    assert extraction.incoterm.rule == "DAP"
    assert extraction.cargo[0].weight.kg == 8400

    assert done["current"] is None  # no current result after finalization


def test_graph_rejection_marks_result_rejected():
    """A human declining the confirmation must not look like an approval."""
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    paused = run_to_pause(config)
    assert "__interrupt__" in paused

    done = graph.invoke(Command(resume={"approved": False}), config=config)

    current = done["results"][0]
    assert current.status == "rejected"
    assert current.output is not None  # rejection preserves the extraction
    assert done["current"] is None  # no current result after rejection


def test_graph_handles_extraction_error():
    """If the extraction fails, the graph must not crash; it should produce a
    SpecialistResult with status=failed and an error message.
    """
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    eml = FIXTURES / "quote_road_plain_es.eml"

    # Two bad responses: the repair round must also fail for status=failed.
    fake = GenericFakeChatModel(messages=iter(["not json {{{", "still not json"]))
    graph = build_graph(
        executor=StubToolExecutor(), checkpointer=InMemorySaver(), model=fake
    )

    done = graph.invoke({"eml_file_path": str(eml)}, config=config)

    assert "__interrupt__" not in done

    result = done["results"][0]
    assert result.status == "failed"
    assert result.error is not None
    assert "Model did not return valid JSON" in result.error


def test_confirm_reprompts_on_bad_edit_then_accepts_fix():
    """Pins the edit loop end to end, deterministically (fake model, no LLM):
    pause -> approve with an invalid edit -> re-prompt carries `problems` ->
    approve with a valid edit -> executed, gap filled, missing recomputed.
    """
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    # Valid extraction with exactly one gap: weight unstated.
    extraction_json = json.dumps(
        {
            "mode": "road",
            "origin": {"name": "Rotterdam", "locode": "NLRTM"},
            "destination": {"name": "Houston", "locode": "USHOU"},
            "incoterm": {"rule": "EXW", "named_place": "Rotterdam"},
            "cargo": [
                {
                    "description": "Crated lathe",
                    "pieces": 1,
                    "dimensions": {
                        "length": 200,
                        "width": 120,
                        "height": 150,
                        "unit": "cm",
                    },
                }
            ],
        }
    )
    fake = GenericFakeChatModel(messages=iter([extraction_json]))
    graph = build_graph(
        executor=StubToolExecutor(), checkpointer=InMemorySaver(), model=fake
    )

    eml = FIXTURES / "quote_road_plain_es.eml"
    paused = graph.invoke({"eml_file_path": str(eml)}, config=config)

    (pause,) = paused["__interrupt__"]
    assert pause.value["missing"] == ["cargo.0.weight"]
    assert pause.value["problems"] == []

    # Human approves but supplies a malformed weight: must re-prompt, not crash.
    reprompted = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"cargo.0.weight": {"value": 10, "unit": "quintales"}},
            }
        ),
        config=config,
    )

    (pause,) = reprompted["__interrupt__"]
    assert pause.value["problems"] != []
    assert "quintales" in pause.value["problems"][0]

    # Valid fix: run completes, edit lands in the executed result.
    done = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"cargo.0.weight": {"value": 1.2, "unit": "toneladas"}},
            }
        ),
        config=config,
    )

    assert "__interrupt__" not in done
    result = done["results"][0]
    assert result.status == "executed"
    assert result.missing == []
    assert result.output is not None
    weight = result.output.cargo[0].weight
    assert weight is not None
    assert weight.kg == 1200


def test_malformed_resume_reprompts_instead_of_wedging_the_thread():
    """A resume that fails ConfirmationResume validation (e.g. Studio sends
    {'accepted': ...}) must re-prompt, never raise: LangGraph caches the bad
    resume in the checkpoint and replays it on every retry, so raising here
    permanently wedges the thread — no later correct resume can get through.
    """
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    extraction_json = json.dumps(
        {
            "mode": "road",
            "origin": {"name": "Rotterdam", "locode": "NLRTM"},
            "destination": {"name": "Houston", "locode": "USHOU"},
            "incoterm": {"rule": "EXW", "named_place": "Rotterdam"},
            "cargo": [
                {
                    "description": "Crated lathe",
                    "pieces": 1,
                    "weight": {"value": 950, "unit": "kg"},
                    "dimensions": {
                        "length": 200,
                        "width": 120,
                        "height": 150,
                        "unit": "cm",
                    },
                }
            ],
        }
    )
    fake = GenericFakeChatModel(messages=iter([extraction_json]))
    graph = build_graph(
        executor=StubToolExecutor(), checkpointer=InMemorySaver(), model=fake
    )

    eml = FIXTURES / "quote_road_plain_es.eml"
    paused = graph.invoke({"eml_file_path": str(eml)}, config=config)
    assert "__interrupt__" in paused

    # Malformed resume: wrong key. Must come back as a re-prompt, not an error.
    reprompted = graph.invoke(Command(resume={"accepted": True}), config=config)
    payload = reprompted["__interrupt__"][0].value
    assert payload["problems"] != []
    assert "approved" in payload["problems"][0]

    # The thread is still alive: a correct resume completes the run.
    done = graph.invoke(Command(resume={"approved": True, "edits": {}}), config=config)
    assert "__interrupt__" not in done
    assert done["results"][0].status == "executed"


def test_repair_warning_reaches_payload_and_final_result():
    """A repaired extraction must announce itself: the warning shows in the
    confirmation payload (human sees it before approving) and survives into
    the finalized result (ops/audit sees it after).
    """
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    complete_json = json.dumps(
        {
            "mode": "road",
            "origin": {"name": "Rotterdam", "locode": "NLRTM"},
            "destination": {"name": "Houston", "locode": "USHOU"},
            "incoterm": {"rule": "EXW", "named_place": "Rotterdam"},
            "cargo": [
                {
                    "description": "Crated lathe",
                    "pieces": 1,
                    "weight": {"value": 950, "unit": "kg"},
                    "dimensions": {
                        "length": 200,
                        "width": 120,
                        "height": 150,
                        "unit": "cm",
                    },
                }
            ],
        }
    )
    # First attempt garbage, repair succeeds.
    fake = GenericFakeChatModel(messages=iter(["not json {{{", complete_json]))
    graph = build_graph(
        executor=StubToolExecutor(), checkpointer=InMemorySaver(), model=fake
    )

    eml = FIXTURES / "quote_road_plain_es.eml"
    paused = graph.invoke({"eml_file_path": str(eml)}, config=config)

    payload = paused["__interrupt__"][0].value
    assert payload["warnings"] != []
    assert "attempts" in payload["warnings"][0]
    # Confidence is derived at extraction and reaches the human untouched.
    assert payload["confidence"]["mode"] == "stated"
    assert payload["confidence"]["cargo.0.weight.unit"] == "stated"

    done = graph.invoke(Command(resume={"approved": True, "edits": {}}), config=config)
    assert done["results"][0].warnings == payload["warnings"]


def test_confirm_remediates_gaps_across_rounds():
    """Approve-with-gaps re-prompts ('confirmed' implies complete), and valid
    edits stick between rounds: fixing one gap per resume must converge, not
    demand the human re-send everything each time.
    """
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    # Two gaps: weight and incoterm.
    extraction_json = json.dumps(
        {
            "mode": "road",
            "origin": {"name": "Rotterdam", "locode": "NLRTM"},
            "destination": {"name": "Houston", "locode": "USHOU"},
            "cargo": [
                {
                    "description": "Crated lathe",
                    "pieces": 1,
                    "dimensions": {
                        "length": 200,
                        "width": 120,
                        "height": 150,
                        "unit": "cm",
                    },
                }
            ],
        }
    )
    fake = GenericFakeChatModel(messages=iter([extraction_json]))
    graph = build_graph(
        executor=StubToolExecutor(), checkpointer=InMemorySaver(), model=fake
    )

    eml = FIXTURES / "quote_road_plain_es.eml"
    paused = graph.invoke({"eml_file_path": str(eml)}, config=config)
    assert paused["__interrupt__"][0].value["missing"] == [
        "incoterm",
        "cargo.0.weight",
    ]

    # Round 1: fix only the weight. Approval must bounce - incoterm still open.
    reprompted = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"cargo.0.weight": {"value": 950, "unit": "kg"}},
            }
        ),
        config=config,
    )

    payload = reprompted["__interrupt__"][0].value
    assert payload["problems"] == ["Still missing: incoterm"]
    assert payload["missing"] == ["incoterm"]
    # The weight fix is visible in the re-prompt: fields and summary updated,
    # and provenance already shows the human supplied it.
    assert payload["fields"]["cargo"][0]["weight"]["kg"] == 950
    assert "950 kg" in payload["summary"]
    assert payload["confidence"]["cargo.0.weight"] == "edited"
    assert "cargo.0.weight.unit" not in payload["confidence"]

    # Round 2: fix only the incoterm - the weight edit must have stuck.
    done = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"incoterm": {"rule": "EXW", "named_place": "Rotterdam"}},
            }
        ),
        config=config,
    )

    assert "__interrupt__" not in done
    result = done["results"][0]
    assert result.status == "executed"
    assert result.missing == []
    assert result.output is not None
    assert result.output.incoterm is not None
    assert result.output.incoterm.rule == "EXW"
    weight = result.output.cargo[0].weight
    assert weight is not None
    assert weight.kg == 950
    # Provenance accumulated across both rounds survives into the final result.
    assert result.confidence["cargo.0.weight"] == "edited"
    assert result.confidence["incoterm"] == "edited"
    assert result.confidence["mode"] == "stated"
