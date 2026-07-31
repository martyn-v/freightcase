from __future__ import annotations
from freightcase.extraction import extract_quote_request
from freightcase.intake import parse_eml
import operator
from typing import Annotated, NotRequired, TypedDict, Literal
from langgraph.types import Command
from langgraph.graph import END, START, StateGraph
from freightcase.intake import IntakeResult
from freightcase.schemas import QuoteRequest


class State(TypedDict):
    eml_file_path: str
    intake_result: NotRequired[
        IntakeResult | None
    ]  # The result of parsing the .eml file
    extraction: NotRequired[QuoteRequest | None]  # The extracted quote request data
    warnings: NotRequired[Annotated[list[str], operator.add]]


def extract_quote(state: State) -> dict:
    """Extracts the quote request from the intake result and stores it in the state."""

    if "intake_result" not in state or state["intake_result"] is None:
        raise ValueError("Intake result is missing. Please run intake first.")

    # TODO: repair loop on ExtractionError (.raw / .validation_error) instead of failing the run
    result = extract_quote_request(state["intake_result"].body_text)

    return {"extraction": result}


quote_agent_builder = StateGraph(State)
quote_agent_builder.add_node("extract", extract_quote)
quote_agent_builder.add_edge(START, "extract")
quote_agent_builder.add_edge("extract", END)
quote_agent = quote_agent_builder.compile()


def intake(state: State) -> dict:
    """Intake function to parse the .eml file and store the result in the state."""

    with open(state["eml_file_path"], "rb") as f:
        raw = f.read()

    result = parse_eml(raw)

    return {
        "intake_result": result,
        "warnings": result.intake_warnings,
    }


def classify(state: State) -> Command[Literal["quote_agent"]]:
    return Command(goto="quote_agent")


builder = StateGraph(State)
builder.add_node("intake", intake)
builder.add_node("classify", classify)
builder.add_node("quote_agent", quote_agent)

builder.add_edge(START, "intake")
builder.add_edge("intake", "classify")
builder.add_edge("quote_agent", END)
graph = builder.compile()
