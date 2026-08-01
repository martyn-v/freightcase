from __future__ import annotations
from freightcase.contracts import SpecialistResult
from freightcase.extraction import extract_quote_request
from freightcase.intake import parse_eml
import operator
from typing import Annotated, NotRequired, TypedDict, Literal
from langgraph.types import Command
from langgraph.graph import END, START, StateGraph
from freightcase.intake import IntakeResult


class State(TypedDict):
    eml_file_path: str
    intake: NotRequired[IntakeResult | None]  # The result of parsing the .eml file
    results: NotRequired[Annotated[list[SpecialistResult], operator.add]]


def extract_quote(state: State) -> dict:
    """Extracts the quote request from the intake result and stores it in the state."""

    if "intake" not in state or state["intake"] is None:
        raise ValueError("Intake result is missing. Please run intake first.")

    # TODO: repair loop on ExtractionError (.raw / .validation_error) instead of failing the run
    result = extract_quote_request(state["intake"].body_text)

    return {
        "results": [
            SpecialistResult(
                function="quote_request",
                output=result,
                status="extracted",
                missing=result.missing_for_quoting(),
            )
        ]
    }


quote_specialist_builder = StateGraph(State)
quote_specialist_builder.add_node("extract", extract_quote)
quote_specialist_builder.add_edge(START, "extract")
quote_specialist_builder.add_edge("extract", END)
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

builder.add_edge(START, "intake")
builder.add_edge("intake", "classify")
builder.add_edge("quote_specialist", END)
graph = builder.compile()
