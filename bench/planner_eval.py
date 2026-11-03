"""Planner comparison: JSON validity, plan accuracy and latency per model.

    python -m bench.planner_eval --models qwen3-vl:2b qwen3-vl:4b qwen3-vl:8b --out results/planners.json

The "heuristic" row is the regex fallback parser, i.e. what runs when no LLM is
reachable. A model row counts a case as a plan only if the LLM answered with
valid JSON (no fallback); a fallback answer is scored as a failure there.
"""

import argparse
import json
import time

from agent.planner import BasePlanner, OllamaPlanner
from bench.planner_cases import CASES, score


def eval_model(model: str, repeats: int = 1):
    planner = OllamaPlanner(model=model) if model != "heuristic" else BasePlanner()
    rows = []
    for text, gold, any_order in CASES:
        for _ in range(repeats):
            t0 = time.time()
            if model == "heuristic":
                plan, valid = planner._create_fallback_plan(text), True
            else:
                plan = planner.decompose(text, max_retries=1)
                valid = not planner.used_fallback
            rows.append({"instruction": text, "valid_json": valid,
                         "correct": bool(valid and score(plan, gold, any_order)),
                         "latency_s": time.time() - t0})
    n = len(rows)
    lat = sorted(r["latency_s"] for r in rows)
    return {"model": model, "n": n,
            "valid_json": sum(r["valid_json"] for r in rows) / n,
            "plan_accuracy": sum(r["correct"] for r in rows) / n,
            "median_latency_s": lat[n // 2], "p90_latency_s": lat[int(n * 0.9) - 1],
            "cases": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["heuristic", "qwen3-vl:4b"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="results/planners.json")
    args = ap.parse_args()
    out = []
    for m in args.models:
        r = eval_model(m, args.repeats)
        print(f"{m:16s} valid={r['valid_json']:.0%} accuracy={r['plan_accuracy']:.0%} "
              f"median={r['median_latency_s']:.2f}s", flush=True)
        out.append(r)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
