from pathlib import Path
import pytest

from freightcase.graph import extract_quote_request

FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def load(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


# Partial expectations: assert what matters per scenario, not full object equality
CASES = {
    "quote_simple_en": {
        "mode": "ocean_fcl",
        "origin_locode": "CNSHA",
        "incoterm_rule": "FOB",
        "cargo_0_kg": pytest.approx(997.903),
    }
}


@pytest.mark.parametrize("name", CASES.keys(), ids=CASES.keys())
def test_extraction(name: str):
    """Tests the extract_quote_request function against a set of email fixtures and expected outputs."""
    result = extract_quote_request(load(name))
    expected = CASES[name]
    if "mode" in expected:
        assert result.mode == expected["mode"]
    if "cargo_0_kg" in expected:
        assert result.cargo_lines[0].weight.kg == expected["cargo_0_kg"]
