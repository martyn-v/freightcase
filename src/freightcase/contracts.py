from typing import Any, Literal

from pydantic import BaseModel

from freightcase.schemas import Location, QuoteRequest


class SpecialistResult(BaseModel):
    function: Literal["quote_request"]
    output: QuoteRequest | None
    missing: list[str] = []
    confidence: dict[str, float] = {}
    status: Literal["extracted", "confirmed", "executed", "rejected", "failed"]
    error: str | None = None


FieldConfidence = Literal["stated", "normalized", "missing"]


class ConfirmationAction(BaseModel):
    tool: Literal["create_quote"]
    function: Literal["quote_request"]


def _location_label(location: Location) -> str:
    return location.name or location.locode or location.iata or "unknown"


def _summarize_quote_request(request: QuoteRequest) -> str:
    """One sentence stating what confirming will execute. Composed here so
    every surface shows the same truth; totals use canonical kg. Unstated
    fields are named as gaps, never omitted — the human must see them."""
    mode = request.mode or "mode not stated"
    stated_pieces = [line.pieces for line in request.cargo if line.pieces is not None]
    pieces = f"{sum(stated_pieces)} pieces" if stated_pieces else "pieces not stated"
    stated_kg = [line.weight.kg for line in request.cargo if line.weight is not None]
    kg = f"{sum(stated_kg):g} kg" if stated_kg else "weight not stated"
    incoterm = (
        f"{request.incoterm.rule} {request.incoterm.named_place}"
        if request.incoterm is not None
        else "incoterm not stated"
    )
    return (
        f"Create a quote request: {mode}, "
        f"{_location_label(request.origin)} → {_location_label(request.destination)}, "
        f"{pieces}, {kg}, {incoterm}."
    )


class ConfirmationPayload(BaseModel):
    """The payload for a Human-in-the-Loop (HITL) task."""

    action: ConfirmationAction
    summary: str
    fields: dict[str, Any]
    confidence: dict[str, FieldConfidence] = {}
    missing: list[str] = []
    warnings: list[str] = []

    @staticmethod
    def from_result(result: SpecialistResult) -> "ConfirmationPayload":
        if result.output is None:
            raise ValueError(
                "Cannot build a confirmation payload without extraction output "
                f"(status={result.status!r}); failed results have nothing to confirm."
            )
        return ConfirmationPayload(
            action=ConfirmationAction(tool="create_quote", function=result.function),
            summary=_summarize_quote_request(result.output),
            fields=result.output.model_dump(),
            missing=result.missing,
        )


class ConfirmationResume(BaseModel):
    """The payload for resuming a Human-in-the-Loop (HITL) task."""

    approved: bool
    edits: dict[str, Any] = {}
