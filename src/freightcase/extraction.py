import json

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from freightcase.llm import default_model
from freightcase.schemas import QuoteRequest
from freightcase.validation import summarize_validation_error


class ExtractionOutcome(BaseModel):
    """A successful extraction, possibly after repair rounds.

    `raw` is the parsed dict of the *successful* attempt, pre-validation —
    what the model literally said before validators normalized it. Comparing
    it path-by-path against `request` is what powers per-field confidence
    ("normalized" = validator changed it) and transcription proofs in tests.
    (Failed attempts carry their unparseable output as ExtractionError.raw,
    a string — different thing.)
    """

    request: QuoteRequest
    raw: dict
    attempts: int  # total model invocations; 1 = first try succeeded


class ExtractionError(Exception):
    """Extraction produced no valid QuoteRequest. Carries structured detail
    for the repair loop: `raw` is the model's unparseable output as a string
    (unlike ExtractionOutcome.raw, which is a parsed dict)."""

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


SYSTEM_PROMPT_TEMPLATE = """You are an inbox assistant that extracts structured information from emails. Extract the relevant information and return it as a single JSON object matching this JSON Schema exactly. No prose, no markdown fences.

{schema}

Extract values exactly as stated in the email. Do not convert units or reformat identifiers; report what the email says.
Output compact single-line JSON with no whitespace. Omit optional fields that are not present in the email rather than emitting null.
"""


def _attempt(messages: list, model: BaseChatModel) -> tuple[dict, QuoteRequest]:
    """One model invocation -> validated QuoteRequest. Raises ExtractionError
    on invalid JSON (`.raw`) or schema failure (`.validation_error`)."""
    response = model.invoke(messages)
    raw = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Model did not return valid JSON: {e}", raw=raw) from e

    try:
        return data, QuoteRequest.model_validate(data)
    except ValidationError as e:
        raise ExtractionError(
            f"Extraction failed schema validation: {summarize_validation_error(e)}",
            raw=raw,
            validation_error=e,
        ) from e


def _repair_messages(raw: str, error: str) -> list:
    """The repair turn: the model's failed output as its own AI turn, then the
    correction request. The original email is already in the message history."""
    return [
        AIMessage(raw),
        HumanMessage(
            f"That output was rejected: {error}. "
            "Return a corrected JSON object matching the schema. "
            "JSON only, no prose."
        ),
    ]


def extract_quote_request(
    email_content: str, model: BaseChatModel | None = None, max_repairs: int = 1
) -> ExtractionOutcome:
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
        max_repairs: Failed attempts are retried with the rejected output
            and its errors appended to the conversation, up to this many
            repair rounds. 0 disables repair (eval switch: measures raw
            model quality instead of pipeline quality).

    Returns:
        An ExtractionOutcome containing the validated QuoteRequest and metadata.

    Raises:
        ExtractionError: The last attempt's failure, once repairs are
            exhausted — invalid JSON (`.raw` set) or schema validation
            (`.validation_error` carries the Pydantic errors).
    """

    model = model or default_model()
    schema = json.dumps(QuoteRequest.model_json_schema())

    messages = [
        SystemMessage(SYSTEM_PROMPT_TEMPLATE.format(schema=schema)),
        HumanMessage(email_content),
    ]

    attempts = 0
    while True:
        try:
            raw, request = _attempt(messages, model)
            return ExtractionOutcome(request=request, raw=raw, attempts=attempts + 1)
        except ExtractionError as e:
            attempts += 1
            if attempts > max_repairs:
                raise

            messages.extend(_repair_messages(e.raw or "", str(e)))
