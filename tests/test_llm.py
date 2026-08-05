"""model_from_spec: every path must preserve measurement integrity — local
challengers get the same house tuning as the default model, API models get
pinned temperature, and bad specs fail with guidance. No model is contacted:
constructing a ChatOllama does not connect."""

import pytest
from langchain_ollama import ChatOllama

from freightcase.llm import DEFAULT_MODEL, default_model, model_from_spec


def test_empty_spec_is_the_default_model():
    model = model_from_spec("")

    assert isinstance(model, ChatOllama)
    assert model.model == DEFAULT_MODEL
    assert (model.temperature, model.format, model.reasoning) == (0, "json", False)


def test_ollama_spec_keeps_house_tuning_for_fair_comparison():
    model = model_from_spec("ollama:qwen3.5:9b")

    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3.5:9b"  # second colon belongs to the model name
    default = default_model()
    assert isinstance(default, ChatOllama)
    assert (model.temperature, model.format, model.reasoning) == (
        default.temperature,
        default.format,
        default.reasoning,
    )


def test_slash_spec_fails_with_guidance():
    with pytest.raises(ValueError, match="ollama:qwen3.5:9b"):
        model_from_spec("ollama/qwen3.5:9b")


def test_missing_provider_package_fails_with_guidance():
    with pytest.raises(ValueError, match="provider package"):
        model_from_spec("groq:some-model")
