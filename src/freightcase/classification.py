import json
from typing import Literal

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from freightcase.llm import default_model
from freightcase.registry import REGISTRY, Specialization
from freightcase.validation import summarize_validation_error


class ClassificationOutcome(BaseModel):
    classification: Specialization | Literal["unknown"]
    reason: str | None = None


class ClassificationError(Exception):
    """Raised when the classification model fails to produce a valid output."""

    def __init__(
        self,
        message: str,
        *,
        raw: str | None = None,
        validation_error: ValidationError | None = None,
    ):
        super().__init__(message)
        self.raw = raw
        self.validation_error = validation_error


SYSTEM_PROMPT_TEMPLATE = """You are an inbox assistant that classifies emails into exactly one category:

{options}

Return a single JSON object matching this JSON Schema exactly. No prose, no markdown fences.

{schema}
"""


def classification_system_prompt() -> str:
    """Composed from the registry: registering a specialist teaches the
    classifier its name (via the Specialization type in the schema enum)
    and its meaning (via the description here) — no classifier edits."""
    options = "\n".join(
        f'- "{function}": {entry.description}' for function, entry in REGISTRY.items()
    )
    options += (
        '\n- "unknown": none of the above apply; explain briefly in the `reason` field.'
    )
    schema = json.dumps(ClassificationOutcome.model_json_schema())
    return SYSTEM_PROMPT_TEMPLATE.format(options=options, schema=schema)


def classify_email(
    email_body: str, email_subject: str, model: BaseChatModel | None = None
) -> ClassificationOutcome:
    model = model or default_model()

    messages = [
        SystemMessage(content=classification_system_prompt()),
        HumanMessage(
            content=f"Email subject: {email_subject}\nEmail body: {email_body}"
        ),
    ]

    response = model.invoke(messages)
    raw = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ClassificationError(
            f"Model did not return valid JSON: {e}", raw=raw
        ) from e

    try:
        return ClassificationOutcome.model_validate(data)
    except ValidationError as e:
        raise ClassificationError(
            f"Model did not return valid classification: {summarize_validation_error(e)}",
            raw=raw,
            validation_error=e,
        ) from e
