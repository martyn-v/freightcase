import json

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from freightcase.llm import default_model
from freightcase.schemas import QuoteRequest


class ExtractionError(Exception):
    """Extraction produced no valid QuoteRequest. Carries structured detail for the repair loop / HITL confidence."""

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


def summarize_validation_error(ve: ValidationError) -> str:
    """Compact field-path summary, e.g. 'cargo.0.weight: Field required'.
    This lands in SpecialistResult.error, the only detail an ops human sees."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in ve.errors()
    )


SYSTEM_PROMPT_TEMPLATE = """You are an inbox assistant that extracts structured information from emails. Extract the relevant information and return it as a single JSON object matching this JSON Schema exactly. No prose, no markdown fences.

{schema}

Extract values exactly as stated in the email. Do not convert units or reformat identifiers; report what the email says.
Output compact single-line JSON with no whitespace. Omit optional fields that are not present in the email rather than emitting null.
"""


def extract_quote_request(
    email_content: str, model: BaseChatModel | None = None
) -> QuoteRequest:
    """Extract a structured QuoteRequest from raw email text.

    The model is instructed to transcribe values exactly as stated in the
    email (no unit conversion, no reformatting); all normalization and
    interpretation happens deterministically in the schema's validators.
    The JSON Schema is supplied in the system prompt and the response is
    parsed and validated by us, rather than delegating to
    `with_structured_output`, so validation failures surface with their
    structured field errors intact for the repair loop and per-field
    HITL confidence.

    Args:
        email_content: Plain-text email body (untrusted input; passed as
            the human message, never interpolated into instructions).
        model: Any LangChain chat model. Defaults to the local Ollama
            model from AGENT_MODEL; inject an API model for eval runs or
            a fake for tests.

    Returns:
        A validated QuoteRequest.

    Raises:
        ExtractionError: If the response is not valid JSON (`.raw` set)
            or fails schema validation (`.validation_error` carries the
            Pydantic errors).
    """

    model = model or default_model()
    schema = json.dumps(QuoteRequest.model_json_schema())

    messages = [
        SystemMessage(SYSTEM_PROMPT_TEMPLATE.format(schema=schema)),
        HumanMessage(email_content),
    ]

    response = model.invoke(messages)
    raw = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Model did not return valid JSON: {e}", raw=raw) from e

    try:
        return QuoteRequest.model_validate(data)
    except ValidationError as e:
        raise ExtractionError(
            f"Extraction failed schema validation: {summarize_validation_error(e)}",
            raw=raw,
            validation_error=e,
        ) from e
