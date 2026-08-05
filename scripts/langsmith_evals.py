"""LangSmith eval runner: same cases, same scorers, ecosystem-native results.

One experiment per stage; runs appear in LangSmith with full langchain
traces (model, tokens, prompts) because tracing is already on for the
pipeline. Requires LANGSMITH_API_KEY (and tracing env) — see .env.

    uv run python scripts/langsmith_evals.py --stage classification
    uv run python scripts/langsmith_evals.py --model anthropic:claude-haiku-4-5

Note: this uploads case emails + expectations to LangSmith as datasets.
For data that must stay local, use scripts/run_evals.py instead.
"""

import argparse
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langsmith import Client

from freightcase.evals import (
    RUNNERS,
    STAGES,
    CaseResult,
    EvalCase,
    Stage,
    cases_for_stage,
    load_cases,
    score_case,
)
from freightcase.llm import model_from_spec
from freightcase.registry import REGISTRY


def sync_dataset(client: Client, name: str, cases: list[EvalCase]) -> None:
    """Recreate the dataset when the case set changed; examples mirror the
    sidecar files (inputs = the email, outputs = the expectations)."""
    if client.has_dataset(dataset_name=name):
        existing = list(client.list_examples(dataset_name=name))
        if len(existing) == len(cases):
            return
        client.delete_dataset(dataset_name=name)

    dataset = client.create_dataset(dataset_name=name)
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {
                "name": c.name,
                "kind": c.kind,
                "email": c.path.read_text(encoding="utf-8", errors="replace"),
            }
            for c in cases
        ],
        outputs=[c.expected for c in cases],
    )


def make_stage_fns(
    stage: Stage,
    stage_cases: list[EvalCase],
    model: BaseChatModel | None,
    max_repairs: int,
):
    """Target + evaluator for one stage. A factory (not loop closures) so
    each experiment binds its own stage/cases — the B023 late-binding trap."""
    by_name = {c.name: c for c in stage_cases}
    runner = RUNNERS[stage]
    kwargs = {} if stage == "classification" else {"max_repairs": max_repairs}

    def target(inputs: dict) -> dict:
        result = runner(by_name[inputs["name"]], model=model, **kwargs)
        return {
            "classification": result.classification,
            "extracted": result.extracted.model_dump() if result.extracted else None,
            "attempts": result.attempts,
            "error": result.error,
        }

    def expectation_score(inputs: dict, outputs: dict, reference_outputs: dict):
        """Rehydrate the run result and re-score with the shared scorer —
        the registry resolves the concrete schema, as everywhere else."""
        case = by_name[inputs["name"]]
        extracted = None
        if outputs.get("extracted") is not None:
            function = reference_outputs.get("function", "quote_request")
            extracted = REGISTRY[function].schema.model_validate(outputs["extracted"])
        result = CaseResult(
            classification=outputs.get("classification"),
            extracted=extracted,
            attempts=outputs.get("attempts", 0),
            error=outputs.get("error"),
        )
        score = score_case(case, result, stage)
        return {
            "key": "expectation_score",
            "score": score.value,
            "comment": "; ".join(score.failures) or "all expectations met",
        }

    return target, expectation_score


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", default=Path(__file__).resolve().parent.parent / "evals" / "cases"
    )
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--model", default="", help="init_chat_model string")
    parser.add_argument("--max-repairs", type=int, default=1)
    args = parser.parse_args()

    model = model_from_spec(args.model)
    model_label = args.model or "local-default"

    client = Client()
    all_cases = load_cases(Path(args.cases))
    stages: tuple[Stage, ...] = (
        STAGES if args.stage == "all" else (cast(Stage, args.stage),)
    )

    for stage in stages:
        stage_cases = cases_for_stage(all_cases, stage)
        if not stage_cases:
            print(f"{stage}: no cases")
            continue

        dataset_name = f"freightcase-{stage}"
        sync_dataset(client, dataset_name, stage_cases)
        target, expectation_score = make_stage_fns(
            stage, stage_cases, model, args.max_repairs
        )

        experiment = client.evaluate(
            target,
            data=dataset_name,
            evaluators=[expectation_score],
            experiment_prefix=f"{stage}-{model_label}",
            metadata={"model": model_label, "max_repairs": args.max_repairs},
        )
        print(f"{stage}: {experiment.experiment_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
