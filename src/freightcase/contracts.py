from typing import Any, Literal

from pydantic import BaseModel

from freightcase.schemas import QuoteRequest


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


class ConfirmationPayload(BaseModel):
    """The payload for a Human-in-the-Loop (HITL) task."""

    action: ConfirmationAction
    fields: dict[str, Any]
    confidence: dict[str, FieldConfidence] = {}
    missing: list[str] = []
    warnings: list[str] = []

    @staticmethod
    def from_result(result: SpecialistResult) -> "ConfirmationPayload":
        return ConfirmationPayload(
            action=ConfirmationAction(tool="create_quote", function=result.function),
            fields=result.output.model_dump() if result.output is not None else {},
            missing=result.missing,
        )


class ConfirmationResume(BaseModel):
    """The payload for resuming a Human-in-the-Loop (HITL) task."""

    approved: bool
    edits: dict[str, Any] = {}
