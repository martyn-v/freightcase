from __future__ import annotations

import operator
from functools import partial
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from mcp import StdioServerParameters
from pydantic import ValidationError

from freightcase.classification import (
    ClassificationError,
    ClassificationOutcome,
    classify_email,
)
from freightcase.contracts import (
    TOOL_FOR_FUNCTION,
    ConfirmationPayload,
    ConfirmationResume,
    Confirmed,
    Rejected,
    SpecialistResult,
    overlay_edited,
    process_resume,
)
from freightcase.execution import (
    MCPToolExecutor,
    ToolExecutor,
    ToolExecutorError,
)
from freightcase.extraction import ExtractionError, extract_quote_request
from freightcase.intake import EmailAddress, EmailAttachment, IntakeResult, parse_eml
from freightcase.schemas import (
    CargoLine,
    Dimensions,
    Incoterm,
    Location,
    QuoteRequest,
    Weight,
)
from freightcase.validation import summarize_validation_error


class State(TypedDict):
    eml_file_path: str
    intake: NotRequired[IntakeResult]  # The result of parsing the .eml file
    current: NotRequired[SpecialistResult | None]  # The current specialist result
    results: NotRequired[Annotated[list[SpecialistResult], operator.add]]
    classification: NotRequired[ClassificationOutcome]


def require_current(state: State) -> SpecialistResult:
    if "current" not in state or state["current"] is None:
        raise ValueError(
            "Current specialist result is missing. Please run extract_quote first."
        )
    return state["current"]


def extract_quote(state: State, *, model: BaseChatModel | None = None) -> dict:
    """Extracts the quote request from the intake result and stores it in the state."""

    if "intake" not in state or state["intake"] is None:
        raise ValueError("Intake result is missing. Please run intake first.")

    try:
        result = extract_quote_request(
            state["intake"].body_text, model=model, max_repairs=1
        )

    except ExtractionError as e:
        return {
            "current": SpecialistResult(
                function="quote_request", status="failed", error=str(e), output=None
            )
        }

    warnings = []
    if result.attempts > 1:
        warnings.append(
            f"Model required {result.attempts} attempts to produce a valid quote request."
        )

    return {
        "current": SpecialistResult(
            function="quote_request",
            output=result.request,
            status="extracted",
            missing=result.request.missing_for_quoting(),
            confidence=result.request.confidence(result.raw),
            warnings=warnings,
        )
    }


def route_after_extract(state: State) -> Literal["confirm", "__end__"]:
    """If the extraction is complete, go to confirm; if not, leave the subgraph
    (the parent's finalize graduates current either way)."""
    current = require_current(state)
    return "__end__" if current.status == "failed" else "confirm"


def confirm(state: State) -> dict:
    """The HITL gate: pause with a ConfirmationPayload, loop until the human
    either rejects or approves a *complete* request (possibly after edits).

    Thin wrapper per the repo rule: all decision logic lives in the pure
    `process_resume` (contracts.py); this function only owns the interrupt
    mechanics, which have two non-obvious properties:

    - Replay on resume: every resume re-runs this function from the top.
      Earlier `interrupt()` calls return their cached answers instantly and
      the newest call pauses again — so everything before the loop must stay
      cheap and side-effect-free, and the loop naturally steps one human
      answer per resume.
    - The loop is bounded by the human, not a counter: Rejected and Confirmed
      are the only exits, and reject is always available whatever state the
      edits are in.
    """
    current = require_current(state)
    output = current.output
    if output is None:
        # Routing guarantees failed results never reach confirm; this guard
        # keeps the invariant loud if the wiring ever changes.
        raise ValueError("Cannot confirm a result with no extraction output.")

    # First prompt: no problems, payload reflects the extraction as-is.
    payload = ConfirmationPayload.from_result(current)
    # Edit paths the human has successfully applied across rounds; their
    # confidence entries become "edited" (provenance: human, not email).
    applied: set[str] = set()

    while True:
        # Pause here. The resume value is untrusted human input: validate it
        # like any other (rule 1). Crucially, an invalid resume must RE-PROMPT,
        # never raise: LangGraph caches the resume value in the checkpoint and
        # replays it on retries, so an exception here would wedge the thread
        # permanently — no later, corrected resume could ever get through.
        raw = interrupt(payload.model_dump())
        try:
            resume = ConfirmationResume.model_validate(raw)
        except ValidationError as e:
            payload = payload.model_copy(
                update={
                    "problems": [f"Invalid response: {summarize_validation_error(e)}"]
                }
            )
            continue

        decision = process_resume(output, resume)

        if isinstance(decision, Rejected):
            return {"current": current.model_copy(update={"status": "rejected"})}

        if isinstance(decision, Confirmed):
            # `confirmed` implies complete (process_resume enforces it), so
            # missing is [] by construction and execute needs no re-check.
            applied.update(decision.edited_paths)
            return {
                "current": current.model_copy(
                    update={
                        "status": "confirmed",
                        "output": decision.edited,
                        "missing": [],
                        "confidence": overlay_edited(current.confidence, applied),
                    }
                )
            }

        # Reprompt: carry forward whatever process_resume decided survives
        # (original output if the edits were bad, edited output if they were
        # valid but incomplete), and rebuild the payload from it so the human
        # sees their accepted progress — fields, summary, missing and
        # confidence update; problems says why they're being asked again.
        output = decision.output
        applied.update(decision.edited_paths)
        payload = ConfirmationPayload.from_result(
            current.model_copy(
                update={
                    "output": output,
                    "missing": output.missing_for_quoting(),
                    "confidence": overlay_edited(current.confidence, applied),
                }
            )
        ).model_copy(update={"problems": decision.problems})


def route_after_confirm(state: State) -> Literal["execute", "__end__"]:
    """Approved -> execute; rejected -> leave the subgraph (the parent's
    finalize graduates current either way)."""
    current = require_current(state)
    return "execute" if current.status == "confirmed" else "__end__"


def execute(state: State, *, executor: ToolExecutor) -> dict:
    current = require_current(state)
    if current.status != "confirmed":
        raise ValueError(
            f"Cannot execute quote request with status {current.status!r}. "
            "Please confirm the quote request first."
        )

    if current.output is None:
        # Routing invariant: confirm never produces a confirmed result
        # without output. Loud failure means the wiring broke, not the email.
        raise ValueError("Confirmed result has no output; graph wiring is broken.")

    if current.function is None:
        raise ValueError("No confirmable action for an unrouted email")

    try:
        # TOOL_FOR_FUNCTION is the same mapping from_result used to build the
        # displayed action: what the human approved is what runs.
        result = executor.execute(
            TOOL_FOR_FUNCTION[current.function], current.output.model_dump()
        )
        return {
            "current": current.model_copy(
                update={"status": "executed", "execution_ref": result.reference}
            )
        }
    except ToolExecutorError as e:
        return {
            "current": current.model_copy(
                update={"status": "failed", "error": f"Execution failed: {e}"}
            )
        }


def finalize(state: State) -> dict:
    current = require_current(state)

    return {"results": [current], "current": None}


def intake(state: State) -> dict:
    """Intake function to parse the .eml file and store the result in the state."""

    with open(state["eml_file_path"], "rb") as f:
        raw = f.read()

    result = parse_eml(raw)

    return {
        "intake": result,
    }


def classify(
    state: State, *, model=None
) -> Command[Literal["finalize", "quote_specialist"]]:
    if "intake" not in state or state["intake"] is None:
        raise ValueError("Intake result is missing. Please run intake first.")
    intake = state["intake"]

    try:
        outcome = classify_email(
            intake.body_text, intake.subject or "No subject extracted", model=model
        )
    except ClassificationError as e:
        outcome = ClassificationOutcome(
            classification="unknown", reason=f"Classifier failed: {e}"
        )

    if outcome.classification == "unknown":
        return Command(
            update={
                "classification": outcome,
                "current": SpecialistResult(
                    function=None,
                    output=None,
                    status="failed",
                    error=f"No specialist for this email: {outcome.reason}",
                ),
            },
            goto="finalize",  # straight to the dead-letter queue
        )
    return Command(update={"classification": outcome}, goto="quote_specialist")


# Our pydantic models that legitimately live inside checkpoints — the
# allowlist is exact (module, class) pairs, so every nested model must be
# listed; a type left out silently deserializes as a plain dict. LangGraph
# warns on (and will eventually block) unregistered types; anyone
# constructing a checkpointer for this graph should pass
# `serde=checkpoint_serde()` so strict mode stays green.
CHECKPOINT_ALLOWED_TYPES: list[type] = [
    ClassificationOutcome,
    EmailAddress,
    EmailAttachment,
    IntakeResult,
    SpecialistResult,
    # Everything reachable from SpecialistResult.output:
    QuoteRequest,
    CargoLine,
    Weight,
    Dimensions,
    Location,
    Incoterm,
]


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_TYPES)


def build_graph(
    executor: ToolExecutor,
    checkpointer: BaseCheckpointSaver | None = None,
    model: BaseChatModel | None = None,
):
    """Compile the graph. Tests/CLI pass a checkpointer (required for the HITL
    interrupt to resume); the LangGraph API server injects its own, so the
    module-level `graph` compiles bare."""

    quote_specialist_builder = StateGraph(State)
    quote_specialist_builder.add_node("extract", partial(extract_quote, model=model))
    quote_specialist_builder.add_node("confirm", confirm)
    quote_specialist_builder.add_node("execute", partial(execute, executor=executor))
    quote_specialist_builder.add_edge(START, "extract")
    quote_specialist_builder.add_conditional_edges("extract", route_after_extract)
    quote_specialist_builder.add_conditional_edges("confirm", route_after_confirm)
    quote_specialist_builder.add_edge("execute", END)
    quote_specialist = quote_specialist_builder.compile()

    builder = StateGraph(State)
    builder.add_node("intake", intake)
    # partial() hides classify's Command[Literal[...]] return annotation from
    # LangGraph's destination inference; declare the edges explicitly so the
    # rendered graph stays truthful.
    builder.add_node(
        "classify",
        partial(classify, model=model),
        destinations=("quote_specialist", "finalize"),
    )
    builder.add_node("quote_specialist", quote_specialist)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_edge("quote_specialist", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph(
    executor=MCPToolExecutor(
        server_command=StdioServerParameters(
            command="uv", args=["run", "python", "-m", "freightcase.tms_stub"]
        )
    )
)
