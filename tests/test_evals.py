"""The eval harness's deterministic parts: case loading, stage selection,
and scoring. No LLM, no eval framework — freightcase.evals is the contract."""

import json
from pathlib import Path

import pytest

from freightcase.evals import (
    CaseResult,
    EvalCase,
    cases_for_stage,
    load_cases,
    score_case,
)
from freightcase.specialists.quote import QuoteRequest


def write_case(folder: Path, name: str, suffix: str, expected: dict) -> None:
    (folder / f"{name}{suffix}").write_text("body", encoding="utf-8")
    (folder / f"{name}.expected.json").write_text(json.dumps(expected))


class TestLoadCases:
    def test_pairs_email_files_with_sidecars(self, tmp_path: Path):
        write_case(tmp_path, "a", ".txt", {"fields": {"mode": "road"}})
        write_case(tmp_path, "b", ".eml", {"classification": "unknown"})

        cases = load_cases(tmp_path)

        assert [(c.name, c.kind) for c in cases] == [("a", "txt"), ("b", "eml")]
        assert cases[0].expected == {"fields": {"mode": "road"}}

    def test_email_without_sidecar_fails_loudly(self, tmp_path: Path):
        (tmp_path / "lonely.txt").write_text("body")

        with pytest.raises(FileNotFoundError, match="lonely"):
            load_cases(tmp_path)

    def test_orphan_sidecar_fails_loudly(self, tmp_path: Path):
        write_case(tmp_path, "a", ".txt", {})
        (tmp_path / "typo.expected.json").write_text("{}")

        with pytest.raises(FileNotFoundError, match="typo"):
            load_cases(tmp_path)

    def test_empty_folder_fails_loudly(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .eml/.txt"):
            load_cases(tmp_path)


def case(kind: str = "txt", **expected) -> EvalCase:
    return EvalCase(name="t", path=Path(f"t.{kind}"), kind=kind, expected=expected)


class TestCasesForStage:
    def test_stage_participation_follows_expectations_and_kind(self):
        labelled_eml = case(kind="eml", classification="unknown")
        fields_txt = case(kind="txt", fields={"mode": "road"})
        fields_eml = case(
            kind="eml", classification="quote_request", fields={"mode": "road"}
        )
        bare_eml = case(kind="eml")

        cases = [labelled_eml, fields_txt, fields_eml, bare_eml]

        assert cases_for_stage(cases, "classification") == [labelled_eml, fields_eml]
        assert cases_for_stage(cases, "extraction") == [fields_txt, fields_eml]
        # pipeline takes every full email, labelled or not
        assert cases_for_stage(cases, "pipeline") == [
            labelled_eml,
            fields_eml,
            bare_eml,
        ]

    def test_txt_never_classifies(self):
        mislabelled_txt = case(kind="txt", classification="quote_request")
        assert cases_for_stage([mislabelled_txt], "classification") == []


def extracted() -> QuoteRequest:
    return QuoteRequest.model_validate(
        {
            "mode": "road",
            "origin": {"name": "Rotterdam", "locode": "NLRTM"},
            "destination": {"name": "Houston", "locode": "USHOU"},
            "incoterm": {"rule": "EXW", "named_place": "Rotterdam"},
            "cargo": [
                {
                    "description": "Lathe",
                    "pieces": 1,
                    "weight": {"value": 8.4, "unit": "t"},
                }
            ],
        }
    )


class TestScoring:
    def test_all_expectations_met_scores_one(self):
        c = case(
            fields={
                "mode": "road",
                "cargo.0.weight.kg": 8400,  # canonical computed field via dump
                "origin.locode": "NLRTM",
            },
            missing=["cargo.0.dimensions"],
        )
        score = score_case(c, CaseResult(extracted=extracted()), "extraction")

        assert score.value == 1.0
        assert score.failures == []

    def test_numeric_tolerance_is_relative(self):
        c = case(fields={"cargo.0.weight.kg": 8400.01})
        result = CaseResult(extracted=extracted())
        assert score_case(c, result, "extraction").value == 1.0

        c = case(fields={"cargo.0.weight.kg": 9000})
        assert score_case(c, result, "extraction").value == 0.0

    def test_wrong_value_names_the_path(self):
        c = case(fields={"mode": "air"})
        score = score_case(c, CaseResult(extracted=extracted()), "extraction")

        assert score.value == 0.0
        assert "fields.mode" in score.failures[0]
        assert "road" in score.failures[0]

    def test_unknown_expected_path_is_a_failure_not_a_crash(self):
        c = case(fields={"cargo.0.wieght.kg": 8400})
        score = score_case(c, CaseResult(extracted=extracted()), "extraction")

        assert score.value == 0.0
        assert "wieght" in score.failures[0]

    def test_extraction_failure_zeroes_field_checks_with_the_error(self):
        c = case(fields={"mode": "road"}, missing=["incoterm"])
        score = score_case(c, CaseResult(error="extraction failed: boom"), "extraction")

        assert score.value == 0.0
        assert all("boom" in f for f in score.failures)

    def test_missing_is_membership_not_equality(self):
        # actual missing_for_execution() also contains mode etc.; expecting a
        # subset must pass.
        request = QuoteRequest.model_validate(
            {
                "origin": {"name": "Rotterdam"},
                "destination": {"name": "Houston"},
                "cargo": [{"description": "Lathe"}],
            }
        )
        c = case(missing=["cargo.0.weight", "incoterm"])
        score = score_case(c, CaseResult(extracted=request), "extraction")

        assert score.value == 1.0

    def test_classification_stage_scores_the_label_only(self):
        c = case(kind="eml", classification="unknown", fields={"mode": "road"})

        score = score_case(c, CaseResult(classification="unknown"), "classification")
        assert score.value == 1.0
        assert len(score.checks) == 1  # fields ignored in this stage

        score = score_case(
            c, CaseResult(classification="quote_request"), "classification"
        )
        assert score.value == 0.0

    def test_pipeline_stage_scores_everything(self):
        c = case(kind="eml", classification="quote_request", fields={"mode": "road"})
        result = CaseResult(classification="quote_request", extracted=extracted())

        score = score_case(c, result, "pipeline")

        assert score.value == 1.0
        assert len(score.checks) == 2  # label + field

    def test_pipeline_misroute_zeroes_extraction_checks(self):
        # Classifier said unknown -> no extraction happened; both the label
        # and the field expectations fail, attributably.
        c = case(kind="eml", classification="quote_request", fields={"mode": "road"})
        result = CaseResult(classification="unknown")

        score = score_case(c, result, "pipeline")

        assert score.value == 0.0
        assert len(score.checks) == 2
