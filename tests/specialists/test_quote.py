from typing import Any

from freightcase.specialists.common import (
    CargoLine,
    Dimensions,
    Incoterm,
    Location,
    Weight,
)
from freightcase.specialists.quote import QuoteRequest


def make_quote_request(**overrides: Any) -> QuoteRequest:
    """A complete, quotable request; tests override the field under test."""
    fields: dict[str, Any] = {
        "mode": "air",
        "origin": Location(name="Bogota", locode="COBOG", iata="BOG"),
        "destination": Location(name="Panama City", locode="PAPTY", iata="PTY"),
        "incoterm": Incoterm(rule="DAP", named_place="Panama City"),
        "cargo": [
            CargoLine(
                description="Packed foodstuffs",
                hs_code_hint=None,
                pieces=6,
                weight=Weight(value=480, unit="kg"),
                dimensions=Dimensions(length=120, width=80, height=75, unit="cm"),
            )
        ],
    }
    fields.update(overrides)
    return QuoteRequest(**fields)


class TestMissingForQuoting:
    def test_complete_request_has_nothing_missing(self):
        assert make_quote_request().missing_for_execution() == []

    def test_absent_incoterm_is_flagged(self):
        qr = make_quote_request(incoterm=None)
        assert qr.missing_for_execution() == ["incoterm"]

    def test_absent_dimensions_flagged_with_cargo_line_index(self):
        qr = make_quote_request(
            cargo=[
                CargoLine(
                    description="Packed foodstuffs",
                    hs_code_hint=None,
                    pieces=6,
                    weight=Weight(value=480, unit="kg"),
                    dimensions=Dimensions(length=120, width=80, height=75, unit="cm"),
                ),
                CargoLine(
                    description="Machine parts",
                    hs_code_hint=None,
                    pieces=2,
                    weight=Weight(value=900, unit="kg"),
                    dimensions=None,
                ),
            ]
        )
        assert qr.missing_for_execution() == ["cargo.1.dimensions"]

    def test_fcl_does_not_require_dimensions(self):
        # FCL is priced per container; dimensions are implied by the box.
        qr = make_quote_request(
            mode="ocean_fcl",
            cargo=[
                CargoLine(
                    description="Packed foodstuffs",
                    hs_code_hint=None,
                    pieces=18,
                    weight=Weight(value=8400, unit="kg"),
                    dimensions=None,
                )
            ],
        )
        assert qr.missing_for_execution() == []

    def test_absent_mode_is_flagged(self):
        qr = make_quote_request(mode=None)
        assert qr.missing_for_execution() == ["mode"]

    def test_absent_locodes_are_flagged(self):
        # The TMS write needs canonical locations; name alone can't execute.
        qr = make_quote_request(
            origin=Location(name="Bogota"),
            destination=Location(name="Panama City"),
        )
        assert qr.missing_for_execution() == ["origin.locode", "destination.locode"]

    def test_absent_pieces_and_weight_flagged_per_line(self):
        qr = make_quote_request(
            cargo=[
                CargoLine(
                    description="Crated lathe",
                    pieces=None,
                    weight=None,
                    dimensions=Dimensions(length=120, width=80, height=75, unit="cm"),
                )
            ]
        )
        assert qr.missing_for_execution() == ["cargo.0.pieces", "cargo.0.weight"]

    def test_multiple_gaps_reported_in_field_order(self):
        qr = make_quote_request(
            incoterm=None,
            cargo=[
                CargoLine(
                    description="Machine parts",
                    hs_code_hint=None,
                    pieces=2,
                    weight=Weight(value=900, unit="kg"),
                    dimensions=None,
                )
            ],
        )
        assert qr.missing_for_execution() == ["incoterm", "cargo.0.dimensions"]


class TestSummarize:
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
        summary = quote_request.summarize()

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

        summary = quote_request.summarize()

        assert "incoterm not stated" in summary

    def test_summary_survives_unstated_fields(self):
        quote_request = QuoteRequest(
            mode=None,
            origin=Location(name="Rotterdam"),
            destination=Location(name="Houston"),
            incoterm=Incoterm(rule="EXW", named_place="Rotterdam"),
            cargo=[
                CargoLine(
                    description="Crated lathe",
                    pieces=1,
                    weight=None,
                    dimensions=None,
                )
            ],
        )

        summary = quote_request.summarize()

        assert "mode not stated" in summary
        assert "weight not stated" in summary
        assert "Rotterdam" in summary
