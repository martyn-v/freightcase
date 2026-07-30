import pytest
from pydantic import ValidationError

from freightcase.schemas import Weight


class TestWeight:
    @pytest.mark.parametrize("raw_unit", ["quintales", "stone", "", "kgz", "™"])
    def test_rejects_unknown_unit(self, raw_unit: str):
        with pytest.raises(ValidationError, match="Unrecognized weight unit"):
            Weight.model_validate({"value": 10, "unit": raw_unit})

    @pytest.mark.parametrize(
        ("raw_unit", "expected_unit"),
        [
            ("kg", "kg"),
            ("KGS", "kg"),
            ("lbs", "lb"),
            ("LBS.", "lb"),
            ("Pounds", "lb"),
            ("MT", "t"),
            ("tonnes", "t"),
            ("gr", "g"),
        ],
    )
    def test_normalizes_alias(self, raw_unit: str, expected_unit: str):
        w = Weight.model_validate({"value": 2200, "unit": raw_unit})
        assert w.unit == expected_unit

    @pytest.mark.parametrize(
        ("raw_unit", "expected_kg"),
        [
            ("kg", 2200),
            ("lbs", 997.903),
            ("t", 2200000),
            ("g", 2.2),
        ],
    )
    def test_kg_property(self, raw_unit: str, expected_kg: float):
        w = Weight.model_validate({"value": 2200, "unit": raw_unit})
        assert w.kg == pytest.approx(expected_kg, rel=1e-3)
