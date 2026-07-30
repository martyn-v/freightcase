from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from freightcase.schemas import QuoteRequest

load_dotenv()

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gemma4:31b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are an inbox assistant that extracts structured information from emails. Extract the relevant information and return it as a JSON object that conforms to the QuoteRequest schema.
Extract values exactly as stated in the email. Do not convert units or reformat identifiers; report what the email says."""


def default_model() -> BaseChatModel:
    return ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_BASE_URL)


def extract_quote_request(
    email_content: str, model: BaseChatModel | None = None
) -> QuoteRequest:
    model = model or default_model()
    model_with_structured_output = model.with_structured_output(
        QuoteRequest, method="json_schema"
    )

    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(email_content),
    ]

    result = model_with_structured_output.invoke(messages)
    assert isinstance(result, QuoteRequest)

    return result
