from __future__ import annotations
from freightcase.contracts import (
    ConfirmationPayload,
    ConfirmationResume,
    SpecialistResult,
)
from freightcase.extraction import extract_quote_request
from freightcase.intake import parse_eml
import operator
from typing import Annotated, NotRequired, TypedDict, Literal
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, interrupt
from langgraph.graph import END, START, StateGraph
from freightcase.intake import IntakeResult


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


def extract_quote(state: State) -> dict:
    """Extracts the quote request from the intake result and stores it in the state."""

    if "intake" not in state or state["intake"] is None:
        raise ValueError("Intake result is missing. Please run intake first.")

    # TODO: repair loop on ExtractionError (.raw / .validation_error) instead of failing the run
    result = extract_quote_request(state["intake"].body_text)

    return {
        "current": SpecialistResult(
            function="quote_request",
            output=result,
            status="extracted",
            missing=result.missing_for_quoting(),
        )
    }


def confirm(state: State) -> dict:
    current = require_current(state)
    payload = ConfirmationPayload.from_result(current)

    confirmation = ConfirmationResume.model_validate(interrupt(payload.model_dump()))

    # TODO: apply confirmation.edits (re-validate via QuoteRequest.model_validate)

    status = "confirmed" if confirmation.approved else "rejected"

    return {"current": current.model_copy(update={"status": status})}


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


quote_specialist_builder = StateGraph(State)
quote_specialist_builder.add_node("extract", extract_quote)
quote_specialist_builder.add_node("confirm", confirm)
quote_specialist_builder.add_node("execute", execute)
quote_specialist_builder.add_edge(START, "extract")
quote_specialist_builder.add_edge("extract", "confirm")
quote_specialist_builder.add_conditional_edges("confirm", route_after_confirm)
quote_specialist_builder.add_edge("execute", END)
quote_specialist = quote_specialist_builder.compile()


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


builder = StateGraph(State)
builder.add_node("intake", intake)
builder.add_node("classify", classify)
builder.add_node("quote_specialist", quote_specialist)
builder.add_node("finalize", finalize)

builder.add_edge(START, "intake")
builder.add_edge("intake", "classify")
builder.add_edge("quote_specialist", "finalize")
builder.add_edge("finalize", END)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the graph. Tests/CLI pass a checkpointer (required for the HITL
    interrupt to resume); the LangGraph API server injects its own, so the
    module-level `graph` compiles bare."""
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
