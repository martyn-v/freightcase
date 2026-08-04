"""Shared validation utilities, dependency-free within the package."""

from pydantic import ValidationError


def summarize_validation_error(ve: ValidationError) -> str:
    """Compact field-path summary, e.g. 'cargo.0.weight: Field required'.
    Used wherever a ValidationError must become a human-readable one-liner:
    dead-letter errors, HITL re-prompt problems, classification failures."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in ve.errors()
    )
