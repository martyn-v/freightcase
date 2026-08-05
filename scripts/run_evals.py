"""Local eval runner: zero infrastructure, nothing leaves the machine.

    uv run python scripts/run_evals.py                        # all stages, local model
    uv run python scripts/run_evals.py --stage classification
    uv run python scripts/run_evals.py --cases /path/to/my-evals
    uv run python scripts/run_evals.py --model anthropic:claude-haiku-4-5
    uv run python scripts/run_evals.py --max-repairs 0 --json results.json

--model takes any langchain init_chat_model string (the matching provider
package and API key are yours to supply); omitted = the local Ollama default.
"""

import argparse
import json
import time
from pathlib import Path

from freightcase.evals import (
    RUNNERS,
    STAGES,
    cases_for_stage,
    load_cases,
    score_case,
)
from freightcase.llm import model_from_spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", default=Path(__file__).resolve().parent.parent / "evals" / "cases"
    )
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--model", default="", help="init_chat_model string")
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--json", dest="json_path", default="", help="artifact path")
    args = parser.parse_args()

    model = model_from_spec(args.model)
    model_label = args.model or "local default"

    cases = load_cases(Path(args.cases))
    stages = STAGES if args.stage == "all" else (args.stage,)
    artifact: dict = {
        "model": model_label,
        "max_repairs": args.max_repairs,
        "stages": {},
    }

    for stage in stages:
        stage_cases = cases_for_stage(cases, stage)
        if not stage_cases:
            print(f"\n== {stage}: no cases ==")
            continue

        runner = RUNNERS[stage]
        kwargs = {} if stage == "classification" else {"max_repairs": args.max_repairs}
        rows = []
        print(f"\n== {stage} ({len(stage_cases)} cases, model: {model_label}) ==")

        for case in stage_cases:
            started = time.perf_counter()
            result = runner(case, model=model, **kwargs)
            elapsed = time.perf_counter() - started
            score = score_case(case, result, stage)
            rows.append(
                {
                    "case": case.name,
                    "score": score.value,
                    "checks": len(score.checks),
                    "attempts": result.attempts,
                    "error": result.error,
                    "failures": score.failures,
                    "seconds": round(elapsed, 2),
                }
            )
            marker = "PASS" if score.value == 1.0 else "FAIL"
            print(f"  {marker}  {case.name:<32} {score.value:>5.2f}  {elapsed:5.1f}s")
            for failure in score.failures:
                print(f"        - {failure}")

        scored = [r for r in rows if r["checks"]]
        mean = sum(r["score"] for r in scored) / len(scored) if scored else 1.0
        failed_runs = sum(1 for r in rows if r["error"])
        repaired = sum(1 for r in rows if r["attempts"] > 1)
        print(
            f"  mean {mean:.3f} | run failures {failed_runs}/{len(rows)}"
            + (f" | repaired {repaired}" if stage != "classification" else "")
        )
        artifact["stages"][stage] = {"mean": round(mean, 4), "cases": rows}

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(artifact, indent=2))
        print(f"\nartifact: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
