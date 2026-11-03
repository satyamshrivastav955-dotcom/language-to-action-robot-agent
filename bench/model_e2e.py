"""End to end, per planner model: instruction -> LLM plan -> closed loop -> ground-truth score.

    python -m bench.model_e2e --models heuristic qwen3-vl:2b qwen3-vl:4b --seeds 10 --profiles slippery biased

Plans are generated once per (model, task, seed) and reused across arms, so the
arms differ only in what the robot does after the plan. A plan that is wrong
(or a clarification question) simply fails the episode, as it should.
"""

import argparse
import json
import os
import time

from agent.planner import BasePlanner, OllamaPlanner
from bench.episode import run_episode
from bench.stats import wilson
from bench.tasks import TASKS, make_task
from env.so101_env import create_environment


class FixedPlanner:
    used_fallback = False
    backend_name = "fixed"

    def __init__(self, plan):
        self.plan = plan

    def decompose(self, instruction):
        return self.plan


def make_plan(model: str, instruction: str):
    if model == "heuristic":
        return BasePlanner()._create_fallback_plan(instruction), True, 0.0
    planner = OllamaPlanner(model=model)
    t0 = time.time()
    plan = planner.decompose(instruction, max_retries=2)
    return plan, not planner.used_fallback, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["heuristic", "qwen3-vl:4b"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--profiles", nargs="+", default=["slippery", "biased"])
    ap.add_argument("--arms", nargs="+", default=["no_verifier", "verifier_corrected"])
    ap.add_argument("--out", default="results/e2e.json")
    args = ap.parse_args()

    env = create_environment()
    out = []
    if os.path.exists(args.out):  # resume: keep finished (model, profile) pairs
        with open(args.out, encoding="utf-8") as f:
            out = json.load(f)
    done = {(r["model"], r["profile"]) for r in out}
    for model in args.models:
        if all((model, p) in done for p in args.profiles):
            continue
        plans = {}
        for task in TASKS:
            for seed in range(args.seeds):
                plans[(task, seed)] = make_plan(model, make_task(task, seed).instruction)
        valid = sum(1 for p in plans.values() if p[1]) / len(plans)
        latency = sorted(p[2] for p in plans.values())[len(plans) // 2]
        for profile in args.profiles:
            if (model, profile) in done:
                continue
            arms = {}
            for arm in args.arms:
                wins = n = 0
                for (task, seed), (plan, _, _) in plans.items():
                    r = run_episode(task, seed, profile, arm, env=env, planner=FixedPlanner(plan))
                    wins += bool(r["oracle_success"])
                    n += 1
                lo, hi = wilson(wins, n)
                arms[arm] = {"n": n, "k": wins, "rate": wins / n, "lo": lo, "hi": hi}
            out.append({"model": model, "profile": profile, "llm_valid_rate": valid,
                        "median_plan_latency_s": latency, "arms": arms})
            print(model, profile, {a: f"{v['k']}/{v['n']}" for a, v in arms.items()}, flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
