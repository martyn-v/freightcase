from __future__ import annotations
from functools import partial

from pydantic import ValidationError
from freightcase.contracts import (
    ConfirmationPayload,
    ConfirmationResume,
    EditError,
    SpecialistResult,
    apply_edits,
)
from freightcase.extraction import (
    ExtractionError,
    extract_quote_request,
    summarize_validation_error,
)
from freightcase.intake import parse_eml
import operator
from typing import Annotated, NotRequired, TypedDict, Literal
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, interrupt
from langgraph.graph import END, START, StateGraph
from freightcase.intake import IntakeResult
from freightcase.schemas import QuoteRequest


class State(TypedDict):
    eml_file_path: str
    intake: NotRequired[IntakeResult]  # The result of parsing the .eml file
    current: NotRequired[SpecialistResult | None]  # The current specialist result
    results: NotRequired[Annotated[list[SpecialistResult], operator.add]]


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

    # TODO: repair loop on ExtractionError (.raw / .validation_error) instead of failing the run
    try:
        result = extract_quote_request(state["intake"].body_text, model=model)
    except ExtractionError as e:
        return {
            "current": SpecialistResult(
                function="quote_request", status="failed", error=str(e), output=None
            )
        }

    return {
        "current": SpecialistResult(
            function="quote_request",
            output=result,
            status="extracted",
            missing=result.missing_for_quoting(),
        )
    }


def route_after_extract(state: State) -> Literal["confirm", "__end__"]:
    """If the extraction is complete, go to confirm; if not, leave the subgraph
    (the parent's finalize graduates current either way)."""
    current = require_current(state)
    return "__end__" if current.status == "failed" else "confirm"


def confirm(state: State) -> dict:
    current = require_current(state)

    payload = ConfirmationPayload.from_result(current)
    output: QuoteRequest = current.output  # type: ignore (we know it's not None because from_result would have raised otherwise)

    while True:
        resume = ConfirmationResume.model_validate(interrupt(payload.model_dump()))
        if not resume.approved:
            return {"current": current.model_copy(update={"status": "rejected"})}

        try:
            edited = apply_edits(output, resume.edits)
        except EditError as e:
            payload = payload.model_copy(update={"problems": [str(e)]})
            continue
        except ValidationError as e:
            payload = payload.model_copy(
                update={"problems": [summarize_validation_error(e)]}
            )
            continue

        # TODO: check if edited has missing fields for quoting; if so, return to the user for further edits instead of confirming

        return {
            "current": current.model_copy(
                update={
                    "status": "confirmed",
                    "output": edited,
                    "missing": edited.missing_for_quoting(),
                }
            )
        }


def route_after_confirm(state: State) -> Literal["execute", "__end__"]:
    """Approved -> execute; rejected -> leave the subgraph (the parent's
    finalize graduates current either way)."""
    current = require_current(state)
    return "execute" if current.status == "confirmed" else "__end__"


def execute(state: State) -> dict:
    current = require_current(state)
    if current.status != "confirmed":
        raise ValueError(
            f"Cannot execute quote request with status {current.status!r}. "
            "Please confirm the quote request first."
        )

    # TODO: Implementation of the actual execution logic (e.g., sending the quote request to an external system)

    return {"current": current.model_copy(update={"status": "executed"})}


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


def classify(state: State) -> Command[Literal["quote_specialist"]]:
    return Command(goto="quote_specialist")


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None, model: BaseChatModel | None = None
):
    """Compile the graph. Tests/CLI pass a checkpointer (required for the HITL
    interrupt to resume); the LangGraph API server injects its own, so the
    module-level `graph` compiles bare."""

    quote_specialist_builder = StateGraph(State)
    quote_specialist_builder.add_node("extract", partial(extract_quote, model=model))
    quote_specialist_builder.add_node("confirm", confirm)
    quote_specialist_builder.add_node("execute", execute)
    quote_specialist_builder.add_edge(START, "extract")
    quote_specialist_builder.add_conditional_edges("extract", route_after_extract)
    quote_specialist_builder.add_conditional_edges("confirm", route_after_confirm)
    quote_specialist_builder.add_edge("execute", END)
    quote_specialist = quote_specialist_builder.compile()

    builder = StateGraph(State)
    builder.add_node("intake", intake)
    builder.add_node("classify", classify)
    builder.add_node("quote_specialist", quote_specialist)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_edge("quote_specialist", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
