from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

# Per-field provenance: what the human can trust about each extracted value.
# Deterministic, not model-emitted (rule 1): "normalized" means a validator
# changed what the model transcribed; "edited" means the value came from the
# human at the confirmation gate, not from the email at all.
FieldConfidence = Literal["stated", "normalized", "missing", "edited"]


class SpecialistSchema(BaseModel, ABC):
    @abstractmethod
    def summarize(self) -> str: ...

    @abstractmethod
    def missing_for_execution(self) -> list[str]: ...

    def confidence(self, raw_model_output: dict) -> dict[str, FieldConfidence]:
        """Exhaustive per-field provenance, derived by walking every schema
        field (computed fields excluded) of this validated request and
        comparing it to the raw pre-validation dict it was built from
        (rule 1: the comparison is code, never model judgment).

        Leaf fields get entries ("cargo.0.weight.unit"); a None field is
        reported "missing" at its own path with no entries beneath it.
        `raw_model_output` must be the dict this instance was validated
        from (ExtractionOutcome.raw)."""
        conf: dict[str, FieldConfidence] = {}
        _walk_confidence(self, raw_model_output, "", conf)
        return conf


def _walk_confidence(
    model: BaseModel, raw: dict, prefix: str, conf: dict[str, FieldConfidence]
) -> None:
    """Recursive worker for QuoteRequest.confidence(). Declared fields only:
    model_fields excludes computed fields, which don't exist in raw output."""
    for name in type(model).model_fields:
        path = f"{prefix}{name}"
        value = getattr(model, name)
        raw_value = raw.get(name) if isinstance(raw, dict) else None

        if value is None:
            conf[path] = "missing"
        elif isinstance(value, BaseModel):
            _walk_confidence(
                value,
                raw_value if isinstance(raw_value, dict) else {},
                f"{path}.",
                conf,
            )
        elif isinstance(value, list):
            for i, item in enumerate(value):
                raw_item = (
                    raw_value[i]
                    if isinstance(raw_value, list) and i < len(raw_value)
                    else {}
                )
                if isinstance(item, BaseModel):
                    _walk_confidence(item, raw_item, f"{path}.{i}.", conf)
                else:
                    conf[f"{path}.{i}"] = "stated" if item == raw_item else "normalized"
        else:
            conf[path] = "stated" if value == raw_value else "normalized"
