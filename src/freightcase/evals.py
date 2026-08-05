"""Eval-harness core: case loading, staged pipeline runners, deterministic
scoring. Framework-free on purpose — the runners on top (the local script,
the LangSmith glue) are thin and disposable; this module is the contract.

Case format — a folder of sidecar pairs, drop-in by convention:

    my-evals/
      rotterdam_lathe.eml            # full email (usable by all stages)
      rotterdam_lathe.expected.json
      simple_quote.txt               # bare body (extraction stage only)
      simple_quote.expected.json

expected.json (every key optional):

    {
      "function": "quote_request",        # extraction dispatch; default quote_request
      "classification": "quote_request",  # classification-stage label (.eml only)
      "fields": {"mode": "road", "cargo.0.weight.kg": 8400},
      "missing": ["incoterm"]             # membership in missing_for_execution()
    }

`fields` uses the same dotted-path vocabulary as `missing`, `confidence`,
and HITL edits. Numbers compare with relative tolerance; everything else
compares exactly.

Stages — separated so failures attribute to the component that caused them:

    classification  one call per .eml with a `classification` label; the
                    cheap, high-N measurement of routing quality
    extraction      dispatched by the sidecar's `function` (the classifier
                    is bypassed by design); fields/missing expectations
    pipeline        classify -> extract end to end on .eml cases; the
                    compounding measurement, scored on all expectations
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel

from freightcase.classification import ClassificationError, classify_email
from freightcase.extraction import ExtractionError, extract_specialist_schema
from freightcase.intake import parse_eml
from freightcase.registry import REGISTRY
from freightcase.specialists.base import SpecialistSchema

Stage = Literal["classification", "extraction", "pipeline"]

STAGES: tuple[Stage, ...] = ("classification", "extraction", "pipeline")


@dataclass(frozen=True)
class EvalCase:
    name: str
    path: Path  # the .eml or .txt file
    kind: str  # "eml" | "txt"
    expected: dict[str, Any]


@dataclass
class CaseResult:
    classification: str | None = None
    extracted: SpecialistSchema | None = None
    attempts: int = 0
    error: str | None = None


@dataclass
class CaseScore:
    """One (check name, passed, detail) row per expectation checked."""

    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    def extend(self, other: CaseScore) -> None:
        self.checks.extend(other.checks)

    @property
    def value(self) -> float:
        if not self.checks:
            return 1.0
        return sum(1 for _, passed, _ in self.checks if passed) / len(self.checks)

    @property
    def failures(self) -> list[str]:
        return [
            f"{name}: {detail}" for name, passed, detail in self.checks if not passed
        ]


def load_cases(folder: Path) -> list[EvalCase]:
    """Discover sidecar pairs. Every .eml/.txt must have its .expected.json
    and vice versa — a stray file is a mistake, and mistakes fail loudly."""
    cases: list[EvalCase] = []
    emails = sorted(
        p for p in folder.iterdir() if p.suffix in (".eml", ".txt") and p.is_file()
    )
    expected_files = set(folder.glob("*.expected.json"))

    for email_path in emails:
        expected_path = email_path.parent / f"{email_path.stem}.expected.json"
        if not expected_path.exists():
            raise FileNotFoundError(
                f"Eval case {email_path.name} has no {expected_path.name} sidecar"
            )
        expected_files.discard(expected_path)
        cases.append(
            EvalCase(
                name=email_path.stem,
                path=email_path,
                kind=email_path.suffix.lstrip("."),
                expected=json.loads(expected_path.read_text(encoding="utf-8")),
            )
        )

    if expected_files:
        orphans = ", ".join(sorted(p.name for p in expected_files))
        raise FileNotFoundError(
            f"expected.json sidecars without email files: {orphans}"
        )
    if not cases:
        raise FileNotFoundError(f"No .eml/.txt eval cases found in {folder}")
    return cases


def cases_for_stage(cases: list[EvalCase], stage: Stage) -> list[EvalCase]:
    """A case participates in a stage when it carries that stage's
    expectations (and, for classification/pipeline, is a full email)."""
    if stage == "classification":
        return [c for c in cases if c.kind == "eml" and "classification" in c.expected]
    if stage == "extraction":
        return [
            c for c in cases if c.expected.get("fields") or c.expected.get("missing")
        ]
    return [c for c in cases if c.kind == "eml"]


# --- stage runners -----------------------------------------------------------


def _email_body(case: EvalCase) -> tuple[str, str]:
    """(body, subject); .txt cases have no headers, subject is empty."""
    if case.kind == "eml":
        intake = parse_eml(case.path.read_bytes())
        return intake.body_text, intake.subject or ""
    return case.path.read_text(encoding="utf-8"), ""


def run_classification_case(
    case: EvalCase, model: BaseChatModel | None = None
) -> CaseResult:
    """One classifier call. A ClassificationError counts as 'unknown' with
    the error recorded — a routing failure, not a crash."""
    body, subject = _email_body(case)
    result = CaseResult()
    try:
        outcome = classify_email(body, subject, model=model)
        result.classification = outcome.classification
    except ClassificationError as e:
        result.classification = "unknown"
        result.error = f"classification failed: {e}"
    return result


def run_extraction_case(
    case: EvalCase, model: BaseChatModel | None = None, max_repairs: int = 1
) -> CaseResult:
    """Extraction in isolation: dispatched by the sidecar's `function`,
    deliberately bypassing the classifier so scores attribute to extraction."""
    body, _ = _email_body(case)
    function = case.expected.get("function", "quote_request")
    result = CaseResult()

    entry = REGISTRY.get(function)
    if entry is None:
        result.error = f"no registry entry for function {function!r}"
        return result

    try:
        extraction = extract_specialist_schema(
            body, schema=entry.schema, model=model, max_repairs=max_repairs
        )
    except ExtractionError as e:
        result.error = f"extraction failed: {e}"
        return result

    result.extracted = extraction.request
    result.attempts = extraction.attempts
    return result


def run_pipeline_case(
    case: EvalCase, model: BaseChatModel | None = None, max_repairs: int = 1
) -> CaseResult:
    """Classify then extract, as production does: extraction follows the
    classifier's verdict, so this measures the compounding behavior."""
    result = run_classification_case(case, model=model)
    if result.classification == "unknown":
        return result

    extraction = run_extraction_case(
        EvalCase(
            name=case.name,
            path=case.path,
            kind=case.kind,
            expected={**case.expected, "function": result.classification},
        ),
        model=model,
        max_repairs=max_repairs,
    )
    extraction.classification = result.classification
    return extraction


RUNNERS = {
    "classification": run_classification_case,
    "extraction": run_extraction_case,
    "pipeline": run_pipeline_case,
}


# --- scoring -----------------------------------------------------------------


def _get_path(data: Any, path: str) -> Any:
    """Read a dotted field path from a model_dump dict; failures become
    ValueError with the path named."""
    target = data
    for part in path.split("."):
        if isinstance(target, list):
            index = int(part)
            if not 0 <= index < len(target):
                raise ValueError(f"{path}: index {index} out of range")
            target = target[index]
        elif isinstance(target, dict):
            if part not in target:
                raise ValueError(f"{path}: unknown field {part!r}")
            target = target[part]
        else:
            # TRY004 suppressed: ValueError is deliberate — scoring catches
            # it to record a failed check; TypeError would crash the run.
            raise ValueError(  # noqa: TRY004
                f"{path}: cannot descend into {type(target).__name__}"
            )
    return target


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(expected, actual, rel_tol=1e-3)
    return expected == actual


def score_classification(case: EvalCase, result: CaseResult) -> CaseScore:
    score = CaseScore()
    if "classification" in case.expected and case.kind == "eml":
        score.add(
            "classification",
            result.classification == case.expected["classification"],
            f"expected {case.expected['classification']!r}, "
            f"got {result.classification!r}",
        )
    return score


def score_extraction(case: EvalCase, result: CaseResult) -> CaseScore:
    """Field/missing expectations. Pipeline failures zero the checks they
    block but are themselves just rows — adopters want failure *rates*."""
    score = CaseScore()
    fields: dict[str, Any] = case.expected.get("fields", {})
    missing_expected: list[str] = case.expected.get("missing", [])
    if not fields and not missing_expected:
        return score

    if result.extracted is None:
        for path in fields:
            score.add(f"fields.{path}", False, result.error or "no extraction")
        for path in missing_expected:
            score.add(f"missing.{path}", False, result.error or "no extraction")
        return score

    dump = result.extracted.model_dump()
    for path, want in fields.items():
        try:
            got = _get_path(dump, path)
        except ValueError as e:
            score.add(f"fields.{path}", False, str(e))
            continue
        score.add(
            f"fields.{path}",
            _values_match(want, got),
            f"expected {want!r}, got {got!r}",
        )

    actual_missing = set(result.extracted.missing_for_execution())
    for path in missing_expected:
        score.add(
            f"missing.{path}",
            path in actual_missing,
            f"not in missing_for_execution() (actual: {sorted(actual_missing)})",
        )
    return score


def score_case(case: EvalCase, result: CaseResult, stage: Stage) -> CaseScore:
    if stage == "classification":
        return score_classification(case, result)
    if stage == "extraction":
        return score_extraction(case, result)
    score = score_classification(case, result)
    score.extend(score_extraction(case, result))
    return score
