import pytest
from pydantic import ValidationError

from freightcase.contracts import (
    Confirmed,
    ConfirmationPayload,
    ConfirmationResume,
    EditError,
    Rejected,
    Reprompt,
    SpecialistResult,
    apply_edits,
    process_resume,
)
from freightcase.schemas import (
    CargoLine,
    Incoterm,
    Location,
    QuoteRequest,
    Weight,
    Dimensions,
)


def incomplete_request() -> QuoteRequest:
    """Rotterdam lathe scenario: weight, dimensions and mode unstated."""
    return QuoteRequest(
        mode=None,
        origin=Location(name="Rotterdam"),
        destination=Location(name="Houston"),
        incoterm=None,
        cargo=[CargoLine(description="Crated lathe", pieces=1)],
    )


class TestApplyEdits:
    def test_dotted_path_fills_gap_and_missing_shrinks(self):
        edited = apply_edits(
            incomplete_request(),
            {"cargo.0.weight": {"value": 1.2, "unit": "toneladas"}},
        )

        assert edited.cargo[0].weight is not None
        assert edited.cargo[0].weight.kg == 1200  # human input hits the validators too
        assert "cargo.0.weight" not in edited.missing_for_quoting()

    def test_top_level_path(self):
        edited = apply_edits(incomplete_request(), {"mode": "ocean_lcl"})
        assert edited.mode == "ocean_lcl"

    def test_invalid_value_fails_loudly(self):
        with pytest.raises(ValidationError):
            apply_edits(
                incomplete_request(),
                {"cargo.0.weight": {"value": 10, "unit": "quintales"}},
            )

    def test_unknown_path_fails_loudly(self):
        with pytest.raises(EditError, match="incotrem"):
            apply_edits(incomplete_request(), {"incotrem": {"rule": "EXW"}})

    def test_out_of_range_index_fails_loudly(self):
        with pytest.raises(EditError, match="out of range"):
            apply_edits(incomplete_request(), {"cargo.5.pieces": 2})

    def test_path_through_unset_parent_fails_loudly(self):
        # incoterm is None: you can't edit incoterm.rule, only incoterm itself.
        with pytest.raises(EditError, match="incoterm"):
            apply_edits(incomplete_request(), {"incoterm.rule": "EXW"})

    def test_original_is_not_mutated(self):
        original = incomplete_request()
        apply_edits(original, {"mode": "air"})
        assert original.mode is None


class TestProcessResume:
    """The confirm node's pure per-resume decision logic, tested without any
    graph machinery. The node itself is a thin interrupt loop over this."""

    def test_rejection_wins_even_with_edits_attached(self):
        resume = ConfirmationResume(approved=False, edits={"mode": "air"})
        decision = process_resume(incomplete_request(), resume)
        assert isinstance(decision, Rejected)

    def test_unknown_edit_path_reprompts_and_discards_edits(self):
        original = incomplete_request()
        resume = ConfirmationResume(approved=True, edits={"incotrem": {}})
        decision = process_resume(original, resume)

        assert isinstance(decision, Reprompt)
        assert "incotrem" in decision.problems[0]
        assert decision.output is original  # rejected wholesale, nothing sticks

    def test_invalid_edit_value_reprompts_and_discards_edits(self):
        original = incomplete_request()
        resume = ConfirmationResume(
            approved=True,
            edits={"cargo.0.weight": {"value": 10, "unit": "quintales"}},
        )
        decision = process_resume(original, resume)

        assert isinstance(decision, Reprompt)
        assert "quintales" in decision.problems[0]
        assert decision.output is original

    def test_valid_edit_with_remaining_gaps_reprompts_but_sticks(self):
        resume = ConfirmationResume(
            approved=True,
            edits={"cargo.0.weight": {"value": 950, "unit": "kg"}},
        )
        decision = process_resume(incomplete_request(), resume)

        assert isinstance(decision, Reprompt)
        assert decision.problems == [
            "Still missing: mode, incoterm, cargo.0.dimensions"
        ]
        # The valid edit is retained for the next round.
        assert decision.output.cargo[0].weight is not None
        assert decision.output.cargo[0].weight.kg == 950

    def test_all_gaps_filled_confirms(self):
        resume = ConfirmationResume(
            approved=True,
            edits={
                "mode": "ocean_lcl",
                "incoterm": {"rule": "EXW", "named_place": "Rotterdam"},
                "cargo.0.weight": {"value": 950, "unit": "kg"},
                "cargo.0.dimensions": {
                    "length": 200,
                    "width": 120,
                    "height": 150,
                    "unit": "cm",
                },
            },
        )
        decision = process_resume(incomplete_request(), resume)

        assert isinstance(decision, Confirmed)
        assert decision.edited.missing_for_quoting() == []
        assert decision.edited.mode == "ocean_lcl"

    def test_complete_request_approves_without_edits(self):
        complete = process_resume(
            incomplete_request(),
            ConfirmationResume(
                approved=True,
                edits={
                    "mode": "road",
                    "incoterm": {"rule": "EXW", "named_place": "Rotterdam"},
                    "cargo.0.weight": {"value": 950, "unit": "kg"},
                    "cargo.0.dimensions": {
                        "length": 200,
                        "width": 120,
                        "height": 150,
                        "unit": "cm",
                    },
                },
            ),
        )
        assert isinstance(complete, Confirmed)

        decision = process_resume(
            complete.edited, ConfirmationResume(approved=True, edits={})
        )
        assert isinstance(decision, Confirmed)


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

    def test_from_result_carries_warnings(self):
        quote_request = QuoteRequest(
            mode="road",
            origin=Location(name="Bogota"),
            destination=Location(name="Medellin"),
            incoterm=Incoterm(rule="DAP", named_place="Medellin"),
            cargo=[
                CargoLine(
                    description="Packed foodstuffs",
                    pieces=12,
                    weight=Weight(value=8400, unit="kg"),
                    dimensions=Dimensions(length=120, width=80, height=60, unit="cm"),
                )
            ],
        )
        result = SpecialistResult(
            function="quote_request",
            output=quote_request,
            status="extracted",
            warnings=["Model required 2 attempts to produce a valid quote request."],
        )

        payload = ConfirmationPayload.from_result(result)

        assert payload.warnings == result.warnings

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
        result = SpecialistResult(
            function="quote_request", output=quote_request, status="extracted"
        )

        summary = ConfirmationPayload.from_result(result).summary

        assert "mode not stated" in summary
        assert "weight not stated" in summary
        assert "Rotterdam" in summary

    def test_from_result_requires_output(self):
        result = SpecialistResult(
            function="quote_request", output=None, status="failed"
        )

        with pytest.raises(ValueError, match="without extraction output"):
            ConfirmationPayload.from_result(result)
