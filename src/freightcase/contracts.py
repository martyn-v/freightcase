from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from freightcase.extraction import summarize_validation_error
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


class EditError(ValueError):
    """A human edit referenced a path that doesn't exist in the request."""


def _set_path(data: Any, path: str, value: Any) -> None:
    """Set `value` at a dotted field path ('incoterm', 'cargo.0.weight') inside
    a model_dump dict, in place. Unknown paths fail loudly — a typo'd edit
    must never be silently ignored."""
    target = data
    parts = path.split(".")
    for depth, part in enumerate(parts):
        last = depth == len(parts) - 1
        if isinstance(target, list):
            try:
                index = int(part)
            except ValueError:
                raise EditError(
                    f"Edit path {path!r}: expected a list index, got {part!r}"
                ) from None
            if not 0 <= index < len(target):
                raise EditError(f"Edit path {path!r}: index {index} out of range")
            if last:
                target[index] = value
            else:
                target = target[index]
        elif isinstance(target, dict):
            if part not in target:
                raise EditError(f"Edit path {path!r}: unknown field {part!r}")
            if last:
                target[part] = value
            else:
                target = target[part]
        else:
            parent = ".".join(parts[:depth])
            raise EditError(
                f"Edit path {path!r}: {parent!r} is not set; edit {parent!r} itself instead"
            )


def apply_edits(original: QuoteRequest, edits: dict[str, Any]) -> QuoteRequest:
    """Apply human edits keyed by dotted field paths — the same vocabulary as
    missing_for_quoting() and confidence — then re-validate the whole object.
    The human is untrusted input like the model (rule 1): a bad value raises
    ValidationError, a bad path raises EditError; nothing is silently dropped."""
    data = original.model_dump()
    for path, value in edits.items():
        _set_path(data, path, value)
    return QuoteRequest.model_validate(data)


@dataclass(frozen=True)
class Rejected:
    """The human declined; the run records the extraction as rejected."""


@dataclass(frozen=True)
class Reprompt:
    """Ask the human again. `problems` says why; `output` is the request the
    next round starts from (original if the edits were rejected wholesale,
    edited if they were valid but left gaps)."""

    problems: list[str]
    output: QuoteRequest


@dataclass(frozen=True)
class Confirmed:
    """Approved and complete; `edited` is the final validated request."""

    edited: QuoteRequest


ConfirmDecision = Rejected | Reprompt | Confirmed


def process_resume(
    output: QuoteRequest, resume: ConfirmationResume
) -> ConfirmDecision:
    """Decide what one human answer means for the confirmation loop.

    Pure function — no interrupt, no state — so every branch is unit-testable.
    Invariants it enforces:
    - Rejection always exits, whatever the edits contain.
    - Edits are all-or-nothing per answer: a bad path or bad value discards
      the whole batch (the human retries from what they saw, no partial state).
    - Valid-but-incomplete answers keep the applied edits (`output` in the
      Reprompt) so multi-round remediation converges.
    - Confirmed implies complete: missing_for_quoting() must be empty.
    """
    if not resume.approved:
        return Rejected()

    try:
        edited = apply_edits(output, resume.edits)
    except EditError as e:
        return Reprompt(problems=[str(e)], output=output)
    except ValidationError as e:
        return Reprompt(problems=[summarize_validation_error(e)], output=output)

    missing = edited.missing_for_quoting()
    if missing:
        return Reprompt(
            problems=[f"Still missing: {', '.join(missing)}"], output=edited
        )

    return Confirmed(edited=edited)


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
    # Why the human is being asked again (invalid edit, remaining gaps).
    # Empty on the first interrupt; populated on re-prompts.
    problems: list[str] = []

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
