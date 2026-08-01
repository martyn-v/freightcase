import pytest

from freightcase.contracts import ConfirmationPayload, SpecialistResult
from freightcase.schemas import (
    CargoLine,
    Incoterm,
    Location,
    QuoteRequest,
    Weight,
    Dimensions,
)


class TestConfirmationPayload:
    def test_from_result(self):
        quote_request = QuoteRequest(
            mode="road",
            origin=Location(locode="CNSHA", iata=None, name="Shanghai"),
            destination=Location(locode="COBOG", iata=None, name="Bogota"),
            incoterm=Incoterm(rule="DAP", named_place="Shanghai"),
            cargo=[
                CargoLine(
                    description="Test cargo",
                    hs_code_hint=None,
                    pieces=3,
                    weight=Weight.model_validate({"value": 8400, "unit": "kgs"}),
                    dimensions=Dimensions.model_validate(
                        {"length": 120, "width": 80, "height": 60, "unit": "inches"}
                    ),
                )
            ],
        )

        result = SpecialistResult(
            function="quote_request",
            output=quote_request,
            status="extracted",
        )

        payload = ConfirmationPayload.from_result(result)

        assert payload.action.tool == "create_quote"
        assert payload.action.function == "quote_request"

        assert payload.fields["mode"] == "road"
        assert payload.fields["origin"]["locode"] == "CNSHA"
        assert payload.fields["destination"]["locode"] == "COBOG"

    def test_from_result_carries_missing(self):
        quote_request = QuoteRequest(
            mode="road",
            origin=Location(name="Bogota"),
            destination=Location(name="Medellin"),
            incoterm=None,
            cargo=[
                CargoLine(
                    description="Packed foodstuffs",
                    hs_code_hint=None,
                    pieces=12,
                    weight=Weight(value=8400, unit="kg"),
                    dimensions=None,
                )
            ],
        )

        result = SpecialistResult(
            function="quote_request",
            output=quote_request,
            missing=quote_request.missing_for_quoting(),
            status="extracted",
        )

        payload = ConfirmationPayload.from_result(result)

        assert payload.missing == ["incoterm", "cargo.0.dimensions"]

    def test_from_result_composes_summary(self):
        quote_request = QuoteRequest(
            mode="road",
            origin=Location(name="Bogota"),
            destination=Location(name="Medellin"),
            incoterm=Incoterm(rule="DAP", named_place="Medellin"),
            cargo=[
                CargoLine(
                    description="Packed foodstuffs",
                    hs_code_hint=None,
                    pieces=12,
                    weight=Weight(value=8.4, unit="t"),
                    dimensions=None,
                )
            ],
        )
        result = SpecialistResult(
            function="quote_request", output=quote_request, status="extracted"
        )

        summary = ConfirmationPayload.from_result(result).summary

        # Facts a human needs before confirming; canonical kg, not "8.4 t".
        assert "road" in summary
        assert "Bogota" in summary
        assert "Medellin" in summary
        assert "12 pieces" in summary
        assert "8400 kg" in summary
        assert "DAP Medellin" in summary

    def test_summary_names_unstated_incoterm(self):
        quote_request = QuoteRequest(
            mode="road",
            origin=Location(name="Bogota"),
            destination=Location(name="Medellin"),
            incoterm=None,
            cargo=[
                CargoLine(
                    description="Packed foodstuffs",
                    hs_code_hint=None,
                    pieces=12,
                    weight=Weight(value=8400, unit="kg"),
                    dimensions=None,
                )
            ],
        )
        result = SpecialistResult(
            function="quote_request", output=quote_request, status="extracted"
        )

        summary = ConfirmationPayload.from_result(result).summary

        assert "incoterm not stated" in summary

    def test_from_result_requires_output(self):
        result = SpecialistResult(
            function="quote_request", output=None, status="failed"
        )

        with pytest.raises(ValueError, match="without extraction output"):
            ConfirmationPayload.from_result(result)
