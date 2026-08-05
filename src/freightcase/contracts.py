from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, SerializeAsAny, ValidationError, model_validator

from freightcase.registry import REGISTRY, Specialization
from freightcase.specialists.base import FieldConfidence, SpecialistSchema
from freightcase.validation import summarize_validation_error


class SpecialistResult(BaseModel):
    function: Specialization | None
    # SerializeAsAny: a field typed by the abstract base would otherwise
    # serialize only base-class fields (i.e. nothing) — dump the concrete
    # instance's full data instead.
    output: SerializeAsAny[SpecialistSchema] | None
    missing: list[str] = []
    confidence: dict[str, FieldConfidence] = {}
    status: Literal["extracted", "confirmed", "executed", "rejected", "failed"]
    warnings: list[str] = []
    error: str | None = None
    execution_ref: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _rehydrate_output(cls, data: Any) -> Any:
        """Checkpoint restore hands us dumped dicts, and pydantic can't pick
        a concrete class for the abstract `output` field. The envelope carries
        its own discriminator — `function` — so resolve the schema through
        the registry and validate the dict back into the real model."""
        if isinstance(data, dict):
            output = data.get("output")
            function = data.get("function")
            if isinstance(output, dict) and function in REGISTRY:
                data = {
                    **data,
                    "output": REGISTRY[function].schema.model_validate(output),
                }
        return data


class ConfirmationAction(BaseModel):
    tool: str
    function: Specialization


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


def apply_edits[S: SpecialistSchema](original: S, edits: dict[str, Any]) -> S:
    """Apply human edits keyed by dotted field paths — the same vocabulary as
    missing_for_execution() and confidence — then re-validate the whole object.
    The human is untrusted input like the model (rule 1): a bad value raises
    ValidationError, a bad path raises EditError; nothing is silently dropped."""
    data = original.model_dump()
    for path, value in edits.items():
        _set_path(data, path, value)
    return type(original).model_validate(data)


@dataclass(frozen=True)
class Rejected:
    """The human declined; the run records the extraction as rejected."""


@dataclass(frozen=True)
class Reprompt[S: SpecialistSchema]:
    """Ask the human again. `problems` says why; `output` is the request the
    next round starts from (original if the edits were rejected wholesale,
    edited if they were valid but left gaps). `edited_paths` are the edit
    paths actually applied this round — empty when edits were rejected."""

    problems: list[str]
    output: S
    edited_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Confirmed[S: SpecialistSchema]:
    """Approved and complete; `edited` is the final validated request and
    `edited_paths` the edit paths applied in the approving answer."""

    edited: S
    edited_paths: tuple[str, ...] = ()


type ConfirmDecision[S: SpecialistSchema] = Rejected | Reprompt[S] | Confirmed[S]


def process_resume[S: SpecialistSchema](
    output: S, resume: ConfirmationResume
) -> ConfirmDecision[S]:
    """Decide what one human answer means for the confirmation loop.

    Pure function — no interrupt, no state — so every branch is unit-testable.
    Invariants it enforces:
    - Rejection always exits, whatever the edits contain.
    - Edits are all-or-nothing per answer: a bad path or bad value discards
      the whole batch (the human retries from what they saw, no partial state).
    - Valid-but-incomplete answers keep the applied edits (`output` in the
      Reprompt) so multi-round remediation converges.
    - Confirmed implies complete: missing_for_execution() must be empty.
    """
    if not resume.approved:
        return Rejected()

    try:
        edited = apply_edits(output, resume.edits)
    except EditError as e:
        return Reprompt(problems=[str(e)], output=output)
    except ValidationError as e:
        return Reprompt(problems=[summarize_validation_error(e)], output=output)

    missing = edited.missing_for_execution()
    if missing:
        return Reprompt(
            problems=[f"Still missing: {', '.join(missing)}"],
            output=edited,
            edited_paths=tuple(resume.edits),
        )

    return Confirmed(edited=edited, edited_paths=tuple(resume.edits))


def overlay_edited(
    confidence: dict[str, FieldConfidence], edited_paths: Iterable[str]
) -> dict[str, FieldConfidence]:
    """Mark human-supplied paths in a provenance map. Entries at or beneath
    each edited path are collapsed into one container-level "edited" entry —
    leaf provenance describes the model's transcription, which no longer
    applies to a value the human replaced. Returns a new dict."""
    paths = tuple(edited_paths)
    out: dict[str, FieldConfidence] = {
        key: value
        for key, value in confidence.items()
        if not any(key == p or key.startswith(f"{p}.") for p in paths)
    }
    for p in paths:
        out[p] = "edited"
    return out


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
    def from_result(result: SpecialistResult) -> ConfirmationPayload:
        if result.output is None:
            raise ValueError(
                "Cannot build a confirmation payload without extraction output "
                f"(status={result.status!r}); failed results have nothing to confirm."
            )
        if result.function is None:
            raise ValueError(
                "Cannot build a confirmation payload for an unrouted email: "
                "no function means no confirmable action."
            )
        return ConfirmationPayload(
            action=ConfirmationAction(
                tool=REGISTRY[result.function].tool, function=result.function
            ),
            summary=result.output.summarize(),
            fields=result.output.model_dump(),
            missing=result.missing,
            confidence=result.confidence,
            warnings=result.warnings,
        )


class ConfirmationResume(BaseModel):
    """The payload for resuming a Human-in-the-Loop (HITL) task."""

    approved: bool
    edits: dict[str, Any] = {}
