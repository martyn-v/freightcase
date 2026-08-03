import json
from pathlib import Path
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from freightcase.graph import build_graph

FIXTURES = Path(__file__).parent / "fixtures" / "emails"

# One checkpointed graph for the module; per-test thread_ids keep runs isolated.
graph = build_graph(checkpointer=InMemorySaver())


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
    assert payload["missing"] == ["cargo.0.dimensions"]

    # Phase 2: human approves; the run finishes.
    done = graph.invoke(Command(resume={"approved": True, "edits": {}}), config=config)

    intake = done["intake"]
    assert intake is not None
    assert intake.body_text.startswith("Buenas tardes")

    current = done["results"][0]
    assert current.function == "quote_request"
    assert current.status == "executed"
    # Email states an incoterm but no dimensions: road mode requires dims.
    assert current.missing == ["cargo.0.dimensions"]

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

    fake = GenericFakeChatModel(messages=iter(["not json {{{"]))
    graph = build_graph(checkpointer=InMemorySaver(), model=fake)

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
            "origin": {"name": "Rotterdam"},
            "destination": {"name": "Houston"},
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
    graph = build_graph(checkpointer=InMemorySaver(), model=fake)

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
