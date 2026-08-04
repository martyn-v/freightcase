import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from freightcase.contracts import SpecialistResult
from freightcase.execution import StubToolExecutor, ToolExecutorError
from freightcase.graph import State, build_graph, execute
from freightcase.schemas import QuoteRequest

FIXTURES = Path(__file__).parent / "fixtures" / "emails"
ROAD_EML = str(FIXTURES / "quote_road_plain_es.eml")


def extraction_json(*, exclude: tuple[str, ...] = ()) -> str:
    """A complete road-freight extraction (the Rotterdam lathe); `exclude`
    removes dotted paths to create the specific gap a test is about."""
    data: dict = {
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
    for path in exclude:
        target = data
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        del target[parts[-1]]
    return json.dumps(data)


def fake_graph(
    *model_responses: str,
    classification: str = '{"classification": "quote_request"}',
):
    """A graph with a scripted model and its own stub executor: fully
    deterministic, no LLM, no shared state between tests. The classify node
    consumes the model's first response, so a classification is prepended;
    override it to script the unknown/dead-letter route."""
    executor = StubToolExecutor()
    model = GenericFakeChatModel(messages=iter([classification, *model_responses]))
    graph = build_graph(
        executor=executor, checkpointer=InMemorySaver(), model=model
    )
    return graph, executor


@pytest.fixture
def live_graph():
    """A graph against the real local model, with its own executor."""
    executor = StubToolExecutor()
    graph = build_graph(executor=executor, checkpointer=InMemorySaver())
    return graph, executor


def config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid4())}}


# --- Journey tests: state flowing through the wiring, one flow per test. ---


def test_approval_journey_from_eml_to_executed_quote(live_graph):
    """The happy path, end to end against the live local model:
    intake -> classify -> extract -> pause at confirm -> human approves
    with edits filling the gaps -> execute writes to the (stub) TMS ->
    finalize graduates the result. Extraction quality is pinned in
    test_extraction.py; assertions here stick to wiring and unambiguous
    fields.
    """
    graph, executor = live_graph
    cfg = config()

    # Stage 1: invoke on the .eml — the run must pause at the HITL gate.
    paused = graph.invoke({"eml_file_path": ROAD_EML}, config=cfg)

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

    # Stage 2: the human approves, filling every gap via edits ('confirmed'
    # implies complete — approving with gaps open would bounce back).
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
        config=cfg,
    )

    # Stage 3: the run is finished — verify what each hop left behind.

    # Intake decoded the email (charset survived the graph plumbing).
    intake = done["intake"]
    assert intake is not None
    assert intake.body_text.startswith("Buenas tardes")

    # The finalized result: executed, complete, carrying the TMS reference.
    current = done["results"][0]
    assert current.function == "quote_request"
    assert current.status == "executed"
    assert current.execution_ref == "Q-STUB-1"
    assert current.missing == []

    # Execute called the tool the human approved, with canonical values.
    assert len(executor.calls) == 1
    tool, sent = executor.calls[0]
    assert tool == "create_quote"
    assert sent["cargo"][0]["weight"]["kg"] == 8400  # canonicals reach the TMS
    assert sent["origin"]["locode"] == "COBOG"  # human edit reached the TMS

    # The extraction itself survived confirm/execute untouched (spot checks).
    extraction = current.output
    assert extraction is not None
    assert extraction.mode == "road"
    assert extraction.incoterm is not None
    assert extraction.incoterm.rule == "DAP"

    # Finalize cleared the in-flight slot.
    assert done["current"] is None


def test_rejection_journey_records_the_decline_and_writes_nothing(live_graph):
    """Extract -> pause -> human rejects -> the result graduates as
    'rejected' with the extraction preserved for audit, and no TMS write
    ever happens."""
    graph, executor = live_graph
    cfg = config()

    # Stage 1: run to the HITL gate.
    paused = graph.invoke({"eml_file_path": ROAD_EML}, config=cfg)
    assert "__interrupt__" in paused

    # Stage 2: the human declines.
    done = graph.invoke(Command(resume={"approved": False}), config=cfg)

    # Stage 3: rejection is a recorded outcome, not a deletion — and it
    # must not look like an approval to the TMS.
    current = done["results"][0]
    assert current.status == "rejected"
    assert current.output is not None  # the extraction is preserved
    assert executor.calls == []  # nothing was written anywhere
    assert done["current"] is None


def test_failed_extraction_journey_ends_in_dead_letter_not_crash():
    """Extract fails, repair round fails too -> the run still completes:
    no pause (there is nothing to confirm), no exception, a 'failed'
    result with a self-explanatory error in the results queue."""
    # Two bad model responses: the repair round must also fail.
    graph, executor = fake_graph("not json {{{", "still not json")

    # Single stage: the run goes straight through to finalize.
    done = graph.invoke({"eml_file_path": ROAD_EML}, config=config())

    assert "__interrupt__" not in done  # never asked a human to confirm nothing
    result = done["results"][0]
    assert result.status == "failed"
    assert result.error is not None
    assert "Model did not return valid JSON" in result.error
    assert executor.calls == []  # and never touched the TMS


def test_bad_edit_journey_reprompts_with_reason_then_accepts_fix():
    """Pause with one gap -> human approves but supplies an invalid edit ->
    re-prompt names the problem -> human fixes it -> executed with the
    corrected value. The Spanish unit in the fix also exercises alias
    normalization through the human input channel."""
    graph, _ = fake_graph(extraction_json(exclude=("cargo.0.weight",)))
    cfg = config()

    # Stage 1: pause; the payload shows the gap and no problems yet.
    paused = graph.invoke({"eml_file_path": ROAD_EML}, config=cfg)
    (pause,) = paused["__interrupt__"]
    assert pause.value["missing"] == ["cargo.0.weight"]
    assert pause.value["problems"] == []

    # Stage 2: approve with a malformed weight — must re-prompt, not crash.
    reprompted = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"cargo.0.weight": {"value": 10, "unit": "quintales"}},
            }
        ),
        config=cfg,
    )
    (pause,) = reprompted["__interrupt__"]
    assert "quintales" in pause.value["problems"][0]  # the reason is named

    # Stage 3: a valid fix — the run completes with the edit applied.
    done = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"cargo.0.weight": {"value": 1.2, "unit": "toneladas"}},
            }
        ),
        config=cfg,
    )

    assert "__interrupt__" not in done
    result = done["results"][0]
    assert result.status == "executed"
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
    graph, _ = fake_graph(extraction_json())
    cfg = config()

    # Stage 1: pause normally.
    paused = graph.invoke({"eml_file_path": ROAD_EML}, config=cfg)
    assert "__interrupt__" in paused

    # Stage 2: resume with a wrong key — must re-prompt naming the field.
    reprompted = graph.invoke(Command(resume={"accepted": True}), config=cfg)
    payload = reprompted["__interrupt__"][0].value
    assert "approved" in payload["problems"][0]

    # Stage 3: the thread is still alive — a correct resume completes it.
    done = graph.invoke(Command(resume={"approved": True, "edits": {}}), config=cfg)
    assert "__interrupt__" not in done
    assert done["results"][0].status == "executed"


def test_repair_warning_reaches_payload_and_final_result():
    """A repaired extraction must announce itself: the warning shows in the
    confirmation payload (human sees it before approving) and survives into
    the finalized result (ops/audit sees it after)."""
    graph, _ = fake_graph("not json {{{", extraction_json())
    cfg = config()

    # Stage 1: first model response is garbage, the repair round succeeds;
    # the pause payload must already announce that repair happened.
    paused = graph.invoke({"eml_file_path": ROAD_EML}, config=cfg)

    payload = paused["__interrupt__"][0].value
    assert "attempts" in payload["warnings"][0]
    # Confidence is derived at extraction and reaches the human untouched.
    assert payload["confidence"]["mode"] == "stated"
    assert payload["confidence"]["cargo.0.weight.unit"] == "stated"

    # Stage 2: approve — the warning survives into the finalized result.
    done = graph.invoke(Command(resume={"approved": True, "edits": {}}), config=cfg)
    assert done["results"][0].warnings == payload["warnings"]


def test_confirm_remediates_gaps_across_rounds():
    """Approve-with-gaps re-prompts ('confirmed' implies complete), and valid
    edits stick between rounds: fixing one gap per resume must converge, not
    demand the human re-send everything each time."""
    graph, _ = fake_graph(
        extraction_json(exclude=("incoterm", "cargo.0.weight"))
    )
    cfg = config()

    # Stage 1: pause with two gaps.
    paused = graph.invoke({"eml_file_path": ROAD_EML}, config=cfg)
    assert paused["__interrupt__"][0].value["missing"] == [
        "incoterm",
        "cargo.0.weight",
    ]

    # Stage 2 (round 1): fix only the weight — approval must bounce,
    # incoterm is still open.
    reprompted = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"cargo.0.weight": {"value": 950, "unit": "kg"}},
            }
        ),
        config=cfg,
    )

    payload = reprompted["__interrupt__"][0].value
    assert payload["problems"] == ["Still missing: incoterm"]
    assert payload["missing"] == ["incoterm"]
    # The weight fix is visible in the re-prompt: fields, summary and
    # provenance all show the human's accepted progress.
    assert payload["fields"]["cargo"][0]["weight"]["kg"] == 950
    assert "950 kg" in payload["summary"]
    assert payload["confidence"]["cargo.0.weight"] == "edited"
    assert "cargo.0.weight.unit" not in payload["confidence"]

    # Stage 3 (round 2): fix only the incoterm — the weight edit from
    # round 1 must have stuck.
    done = graph.invoke(
        Command(
            resume={
                "approved": True,
                "edits": {"incoterm": {"rule": "EXW", "named_place": "Rotterdam"}},
            }
        ),
        config=cfg,
    )

    # Stage 4: done — both human contributions present in one result.
    assert "__interrupt__" not in done
    result = done["results"][0]
    assert result.status == "executed"
    assert result.output is not None
    assert result.output.incoterm is not None
    assert result.output.incoterm.rule == "EXW"
    # Provenance accumulated across both rounds survives into the final result.
    assert result.confidence["cargo.0.weight"] == "edited"
    assert result.confidence["incoterm"] == "edited"
    assert result.confidence["mode"] == "stated"


def test_spam_email_journey_dead_letters_via_live_classifier(live_graph):
    """The real classifier on a real spam .eml: marketing mail must route to
    the dead-letter queue without extraction, pause, or TMS write. Distinct
    fixture failure mode: an email that should never resolve to any function."""
    graph, executor = live_graph

    done = graph.invoke(
        {"eml_file_path": str(FIXTURES / "spam_marketing_en.eml")}, config=config()
    )

    assert "__interrupt__" not in done
    result = done["results"][0]
    assert result.function is None
    assert result.status == "failed"
    assert executor.calls == []
    assert done["classification"].classification == "unknown"


def test_unclassifiable_email_journey_dead_letters_without_pausing():
    """Intake -> classify says 'unknown' -> straight to finalize: a recorded
    failed outcome carrying the classifier's reason. No extraction, no
    interrupt (nothing to confirm), no TMS write; the classifier's verdict
    is kept in state for audit."""
    graph, executor = fake_graph(
        classification='{"classification": "unknown", "reason": "marketing newsletter"}',
    )

    done = graph.invoke({"eml_file_path": ROAD_EML}, config=config())

    assert "__interrupt__" not in done
    result = done["results"][0]
    assert result.function is None  # no specialist was assigned
    assert result.status == "failed"
    assert result.error is not None
    assert "marketing newsletter" in result.error
    assert executor.calls == []  # never touched the TMS
    # The classifier's verdict is auditable in state.
    assert done["classification"].classification == "unknown"


def test_garbage_classifier_output_degrades_to_unknown_not_crash():
    """A classifier that emits garbage must not kill the run: the email
    dead-letters as unknown with the classifier failure as the reason."""
    graph, executor = fake_graph(classification="not json {{{")

    done = graph.invoke({"eml_file_path": ROAD_EML}, config=config())

    assert "__interrupt__" not in done
    result = done["results"][0]
    assert result.function is None
    assert result.status == "failed"
    assert result.error is not None
    assert "Classifier failed" in result.error
    assert executor.calls == []


# --- Node tests: direct calls, no graph machinery. ---


class RaisingToolExecutor:
    def execute(self, tool: str, payload: dict):
        raise ToolExecutorError("TMS unreachable")


def confirmed_result() -> SpecialistResult:
    return SpecialistResult(
        function="quote_request",
        output=QuoteRequest.model_validate(json.loads(extraction_json())),
        status="confirmed",
    )


def test_execute_node_degrades_tool_failure_to_failed_result():
    """A TMS failure after human approval must become a dead-letter entry,
    not an exception (which would wedge the checkpointed thread)."""
    state: State = {"eml_file_path": ROAD_EML, "current": confirmed_result()}

    out = execute(state, executor=RaisingToolExecutor())

    updated = out["current"]
    assert updated.status == "failed"
    assert updated.error is not None
    assert "TMS unreachable" in updated.error


def test_execute_node_refuses_unconfirmed_results():
    """Routing invariant kept loud: execute on anything but 'confirmed' is a
    wiring bug, not a business outcome."""
    state: State = {
        "eml_file_path": ROAD_EML,
        "current": confirmed_result().model_copy(update={"status": "extracted"}),
    }

    with pytest.raises(ValueError, match="status 'extracted'"):
        execute(state, executor=StubToolExecutor())
