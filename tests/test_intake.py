import pytest
from pathlib import Path

from freightcase.intake import parse_eml

FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def load(name: str) -> bytes:
    return (FIXTURES / f"{name}.eml").read_bytes()


CASES = {"quote_road_plain_es": {}, "quote_lcl_htmlonly_en": {}}


@pytest.mark.parametrize("name", CASES.keys(), ids=CASES.keys())
def test_parse_eml(name):
    raw = load(name)
    print(parse_eml(raw))
    # Here you would call your parse_eml function and perform assertions
    # For example:
    # result = parse_eml(raw)
    # assert result.subject == CASES[name]["subject"]
