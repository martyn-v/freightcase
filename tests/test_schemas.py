import pytest
from pydantic import ValidationError

from freightcase.schemas import Dimensions, Weight, Location


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


class TestDimensions:
    @pytest.mark.parametrize("raw_unit", ["yard", "yards", "furlong", "parsec"])
    def test_rejects_unknown_dimension_unit(self, raw_unit: str):
        with pytest.raises(ValidationError, match="Unrecognized dimension unit"):
            Dimensions.model_validate(
                {"length": 10, "width": 5, "height": 2, "unit": raw_unit}
            )

    @pytest.mark.parametrize(
        ("raw_unit", "expected_unit"),
        [
            ("cm", "cm"),
            ("CM.", "cm"),
            ("centimeters", "cm"),
            ("m", "m"),
            ("meters", "m"),
            ("inches", "in"),
            ("in.", "in"),
            ("ft", "ft"),
            ("feet", "ft"),
        ],
    )
    def test_normalizes_dimension_alias(self, raw_unit: str, expected_unit: str):
        d = Dimensions.model_validate(
            {"length": 10, "width": 5, "height": 2, "unit": raw_unit}
        )
        assert d.unit == expected_unit

    @pytest.mark.parametrize(
        ("raw_unit", "expected_cm"),
        [
            ("cm", 10),
            ("m", 1000),
            ("in", 25.4),
            ("ft", 304.8),
        ],
    )
    def test_cm_property(self, raw_unit: str, expected_cm: float):
        d = Dimensions.model_validate(
            {"length": 10, "width": 5, "height": 2, "unit": raw_unit}
        )
        assert d.length_cm == pytest.approx(expected_cm, rel=1e-3)
        assert d.width_cm == pytest.approx(expected_cm / 2, rel=1e-3)
        assert d.height_cm == pytest.approx(expected_cm / 5, rel=1e-3)

    @pytest.mark.parametrize(
        ("length", "width", "height", "unit", "expected_volume_m3"),
        [
            (100, 50, 25, "cm", 0.125),
            (1, 0.5, 0.25, "m", 0.125),
            (39.37, 19.685, 9.843, "in", 0.125),
            (3.281, 1.641, 0.820, "ft", 0.125),
        ],
    )
    def test_volume_m3_property(
        self,
        length: float,
        width: float,
        height: float,
        unit: str,
        expected_volume_m3: float,
    ):
        d = Dimensions.model_validate(
            {"length": length, "width": width, "height": height, "unit": unit}
        )
        assert d.volume_m3 == pytest.approx(expected_volume_m3, rel=1e-3)


class TestLocation:
    def test_uppercases_locode_and_iata(self):
        loc = Location.model_validate(
            {"name": "Cartagena, Colombia", "locode": "usnyc", "iata": "bog"}
        )
        assert loc.locode == "USNYC"
        assert loc.iata == "BOG"

    def test_at_least_one_of_locode_or_iata_or_name(self):
        with pytest.raises(
            ValidationError,
            match="Location requires at least one of: name, locode, iata",
        ):
            Location.model_validate({})
