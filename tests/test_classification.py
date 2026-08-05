import pytest
from langchain_core.language_models import GenericFakeChatModel

from freightcase.classification import (
    ClassificationError,
    classification_system_prompt,
    classify_email,
)
from freightcase.registry import REGISTRY


def test_system_prompt_is_built_from_the_registry():
    """Registering a specialist must teach the classifier its name AND its
    meaning: every registry function and its description appear in the
    prompt, plus the unknown escape hatch. No classifier edits per specialist."""
    prompt = classification_system_prompt()

    for function, entry in REGISTRY.items():
        assert f'"{function}"' in prompt
        assert entry.description in prompt
    assert '"unknown"' in prompt


CASES = {
    "quote": {
        "subject": "Request for quote",
        "body": "Please provide a quote for shipping 10 boxes from New York to Mexico City. Each box weighs 25 kg.",
        "classification": "quote_request",
    },
    "status_update": {
        "subject": "Order status update",
        "body": "Your order has been shipped and is on its way.",
        "classification": "unknown",
    },
}


@pytest.mark.parametrize("name", CASES.keys(), ids=CASES.keys())
def test_classification(name: str):
    """Tests the classify_email function against a set of email fixtures and expected outputs."""
    email = CASES[name]
    result = classify_email(email["body"], email["subject"])
    assert result.classification == email["classification"]


def test_classification_with_fake():
    """Tests the classify_email function with a fake model that returns a valid classification."""
    fake = GenericFakeChatModel(messages=iter(['{"classification": "quote_request"}']))
    result = classify_email("irrelevant email body", "irrelevant subject", model=fake)
    assert result.classification == "quote_request"


def test_classification_with_fake_unknown():
    """Tests the classify_email function with a fake model that returns a valid classification of 'unknown'."""
    fake = GenericFakeChatModel(
        messages=iter(
            [
                '{"classification": "unknown", "reason": "No relevant information found."}'
            ]
        )
    )
    result = classify_email("irrelevant email body", "irrelevant subject", model=fake)
    assert result.classification == "unknown"
    assert result.reason == "No relevant information found."


def test_fails_with_invalid_json():
    """Tests that the classify_email function raises a ClassificationError when the model returns invalid JSON."""
    fake = GenericFakeChatModel(messages=iter(["not json {{{"]))

    with pytest.raises(ClassificationError) as excinfo:
        classify_email("irrelevant email body", "irrelevant subject", model=fake)
    assert "Model did not return valid JSON" in str(excinfo.value)
    assert excinfo.value.raw == "not json {{{"  # debugging payload attached


def test_fails_with_invalid_classification():
    """Tests that the classify_email function raises a ClassificationError when the model returns valid JSON that does not match the expected schema."""
    fake = GenericFakeChatModel(messages=iter(['{"classification": "invalid_value"}']))

    with pytest.raises(ClassificationError) as excinfo:
        classify_email("irrelevant email body", "irrelevant subject", model=fake)
    assert "Model did not return valid classification" in str(excinfo.value)
