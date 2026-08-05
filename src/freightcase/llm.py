import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

load_dotenv()

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gemma4:31b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def default_model() -> BaseChatModel:
    return ChatOllama(
        model=DEFAULT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        format="json",
        reasoning=False,
    )


def model_from_spec(spec: str = "") -> BaseChatModel:
    """Resolve a model spec for evals/CLI use, preserving comparability.

    ""                -> default_model() (AGENT_MODEL with house tuning)
    "ollama:<name>"   -> a local challenger with the SAME house tuning
                         (reasoning off, JSON format, temperature 0) so
                         model comparisons are apples-to-apples
    anything else     -> init_chat_model(spec) with temperature pinned to 0;
                         reasoning/format are Ollama-specific knobs
    """
    if not spec:
        return default_model()
    if spec.startswith("ollama:"):
        return ChatOllama(
            model=spec.removeprefix("ollama:"),
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            reasoning=False,
        )
    try:
        from langchain.chat_models import init_chat_model

        return init_chat_model(spec, temperature=0)
    except ValueError as e:
        raise ValueError(
            f"Could not resolve model spec {spec!r}: {e} — use an "
            "init_chat_model string like 'ollama:qwen3.5:9b' or "
            "'anthropic:claude-haiku-4-5'."
        ) from e
    except ImportError as e:
        raise ValueError(
            f"Model spec {spec!r} needs a langchain provider package that is "
            f"not installed: {e}"
        ) from e
