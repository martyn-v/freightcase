import json
from typing import Literal

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from freightcase.llm import default_model
from freightcase.validation import summarize_validation_error


class ClassificationOutcome(BaseModel):
    classification: Literal["quote_request", "unknown"]
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


SYSTEM_PROMPT_TEMPLATE = """You are an inbox assistant that classifies emails. Classify the email as either a "quote_request" or "unknown". If the email is classified as "unknown", provide a brief reason for the classification.
Return it as a single JSON object matching this JSON Schema exactly. No prose, no markdown fences.

{schema}
"""


def classify_email(
    email_body: str, email_subject: str, model: BaseChatModel | None = None
) -> ClassificationOutcome:
    model = model or default_model()
    schema = json.dumps(ClassificationOutcome.model_json_schema())

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(schema=schema)),
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
