import json
from pathlib import Path

import pytest
from langchain_core.language_models import GenericFakeChatModel

from freightcase.extraction import (
    ExtractionError,
    _repair_messages,
    extract_specialist_schema,
)
from freightcase.specialists.quote import QuoteRequest

MINIMAL_VALID_JSON = json.dumps(
    {
        "mode": "road",
        "origin": {"name": "Bogota"},
        "destination": {"name": "Medellin"},
        "cargo": [{"description": "Packed foodstuffs"}],
    }
)


FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def load(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


# Partial expectations: assert what matters per scenario, not full object equality
CASES = {
    "quote_simple_en": {
        "mode": "ocean_fcl",
        "origin_locode": "CNSHA",
        "origin_iata": None,
        "origin_name": "Shanghai, China",
        "destination_locode": None,
        "destination_iata": None,
        "destination_name": "Cartagena, Colombia",
        "incoterm_rule": "FOB",
        "cargo_0_pieces": 18,
        "cargo_0_kg": pytest.approx(997.903),
        "cargo_0_dimensions": None,
    },
    "quote_es419_mixed_units": {
        "origin_locode": None,
        "origin_iata": None,
        "origin_name": "Valparaíso, Chile",
        "destination_locode": None,
        "destination_iata": None,
        "destination_name": "Guayaquil, Ecuador",
        "cargo_0_pieces": 4,
        "cargo_0_kg": pytest.approx(1500.0),
        "cargo_0_dimensions_height": pytest.approx(85.0),
        "cargo_0_dimensions_height_cm": pytest.approx(85.0),
        "cargo_0_dimensions_length": pytest.approx(110.0),
        "cargo_0_dimensions_length_cm": pytest.approx(110.0),
        "cargo_0_dimensions_width": pytest.approx(90.0),
        "cargo_0_dimensions_width_cm": pytest.approx(90.0),
        "cargo_0_dimensions_unit": "cm",
        "cargo_unit_dimensions_volume_m3": pytest.approx(0.842),
    },
    "quote_forwarded_chain_en": {
        "origin_locode": None,
        "origin_iata": "BOG",
        "origin_name": "Bogota",
        "destination_locode": None,
        "destination_iata": "PTY",
        "destination_name": "Panama City",
        "cargo_0_pieces": 6,
        "cargo_0_kg": pytest.approx(480),
        "cargo_0_dimensions_height": pytest.approx(75.0),
        "cargo_0_dimensions_height_cm": pytest.approx(75.0),
        "cargo_0_dimensions_length": pytest.approx(120.0),
        "cargo_0_dimensions_length_cm": pytest.approx(120.0),
        "cargo_0_dimensions_width": pytest.approx(80.0),
        "cargo_0_dimensions_width_cm": pytest.approx(80.0),
        "cargo_0_dimensions_unit": "cm",
        "cargo_unit_dimensions_volume_m3": pytest.approx(0.72),
    },
    # Distinct failure mode: no incoterm stated — the model must report absence,
    # not fabricate one (rule 2). Dimensions present, so incoterm is the only gap.
    "quote_no_incoterm_en": {
        "mode": "air",
        "incoterm": None,
        "origin_iata": "MDE",
        "destination_iata": "MEX",
        "cargo_0_pieces": 10,
        "cargo_0_kg": pytest.approx(250.0),
        "missing_for_quoting": ["origin.locode", "destination.locode", "incoterm"],
    },
    "quote_imperial_units": {
        "origin_locode": None,
        "origin_iata": "MIA",
        "origin_name": "Miami, FL",
        "destination_locode": None,
        "destination_iata": "PTY",
        "destination_name": "Panama City",
        "cargo_0_pieces": 6,
        "cargo_0_kg": pytest.approx(217.724),
        "cargo_0_dimensions_height": pytest.approx(29.0),
        "cargo_0_dimensions_height_cm": pytest.approx(73.66),
        "cargo_0_dimensions_length": pytest.approx(47.0),
        "cargo_0_dimensions_length_cm": pytest.approx(119.38),
        "cargo_0_dimensions_width": pytest.approx(32.0),
        "cargo_0_dimensions_width_cm": pytest.approx(81.28),
        "cargo_0_dimensions_unit": "in",
        "cargo_unit_dimensions_volume_m3": pytest.approx(0.72),
    },
}


@pytest.mark.parametrize("name", CASES.keys(), ids=CASES.keys())
def test_extraction(name: str):
    """Tests extract_specialist_schema against a set of email fixtures and expected outputs."""
    result = extract_specialist_schema(load(name), schema=QuoteRequest).request

    expected = CASES[name]
    if "mode" in expected:
        assert result.mode == expected["mode"]
    if "origin_locode" in expected:
        assert result.origin.locode == expected["origin_locode"]
    if "origin_iata" in expected:
        assert result.origin.iata == expected["origin_iata"]
    if "origin_name" in expected:
        assert result.origin.name == expected["origin_name"]
    if "destination_locode" in expected:
        assert result.destination.locode == expected["destination_locode"]
    if "destination_iata" in expected:
        assert result.destination.iata == expected["destination_iata"]
    if "destination_name" in expected:
        assert result.destination.name == expected["destination_name"]
    if "incoterm" in expected:
        assert result.incoterm == expected["incoterm"]
    if "incoterm_rule" in expected:
        assert result.incoterm is not None
        assert result.incoterm.rule == expected["incoterm_rule"]
    if "missing_for_quoting" in expected:
        assert result.missing_for_execution() == expected["missing_for_quoting"]
    if "cargo_0_kg" in expected:
        weight = result.cargo[0].weight
        assert weight is not None
        assert weight.kg == expected["cargo_0_kg"]
    if "cargo_0_pieces" in expected:
        assert result.cargo[0].pieces == expected["cargo_0_pieces"]
    if "cargo_0_dimensions" in expected:
        assert result.cargo[0].dimensions == expected["cargo_0_dimensions"]


def test_missing_weights_degrade_not_fail():
    """Design reversal (was: fatal): absence is degraded, not an error. The
    email states no weight or dimensions; extraction succeeds with the gaps
    recorded so HITL can remedy them (rule 3: fatal is for malformed, not
    incomplete)."""
    result = extract_specialist_schema(
        load("quote_missing_weights_en"), schema=QuoteRequest
    ).request

    assert result.cargo[0].weight is None
    missing = result.missing_for_execution()
    assert "cargo.0.weight" in missing
    assert "cargo.0.dimensions" in missing


def test_validation_error_message_names_the_failing_fields():
    """The error string must be self-explanatory: it feeds SpecialistResult.error,
    which is all an ops human sees in the dead-letter queue. No LLM: a fake
    model returns valid JSON with a malformed value (unknown weight unit) —
    a form violation, which stays fatal.
    """
    bad = json.dumps(
        {
            "mode": "air",
            "origin": {"name": "Bogota"},
            "destination": {"name": "Miami"},
            "cargo": [
                {
                    "description": "Cut flowers",
                    "pieces": 1,
                    "weight": {"value": 10, "unit": "quintales"},
                }
            ],
        }
    )
    fake = GenericFakeChatModel(messages=iter([bad]))

    with pytest.raises(ExtractionError) as exc_info:
        extract_specialist_schema(
            "irrelevant email body", schema=QuoteRequest, model=fake, max_repairs=0
        )

    assert "cargo.0.weight.unit" in str(exc_info.value)


def test_repair_loop_recovers_from_bad_first_attempt():
    """First response is garbage; the repair round succeeds. The two-message
    fake also proves exactly two invocations happen."""
    fake = GenericFakeChatModel(messages=iter(["not json {{{", MINIMAL_VALID_JSON]))

    outcome = extract_specialist_schema(
        "irrelevant email body", schema=QuoteRequest, model=fake
    )

    assert outcome.request.mode == "road"
    assert outcome.request.origin.name == "Bogota"
    assert outcome.attempts == 2  # first try failed, repair succeeded
    assert outcome.raw["mode"] == "road"  # the successful attempt's raw dict


def test_repair_exhaustion_raises_last_error():
    """Persistent garbage exhausts repairs and surfaces the final failure."""
    fake = GenericFakeChatModel(messages=iter(["not json {{{", "still not json"]))

    with pytest.raises(ExtractionError, match="valid JSON"):
        extract_specialist_schema(
            "irrelevant email body", schema=QuoteRequest, model=fake, max_repairs=1
        )


def test_repair_messages_put_failed_output_in_ai_turn():
    """The rejected output goes in an AI message (the model's own turn), the
    correction request in a human message naming the field errors."""
    messages = _repair_messages('{"bad": 1}', "cargo.0.weight.unit: unrecognized")

    ai, human = messages
    assert ai.content == '{"bad": 1}'
    assert "cargo.0.weight.unit" in human.content
    assert "corrected" in human.content.lower()
