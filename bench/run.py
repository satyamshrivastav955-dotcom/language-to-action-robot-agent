"""Run the ablation grid in parallel and write results JSON.

    python -m bench.run --episodes 50 --workers 12 --out results/ablation.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time

from bench.episode import ARMS
from bench.tasks import TASKS
from env.faults import PROFILES

_ENV = None


def _init():
    global _ENV
    from env.so101_env import create_environment
    _ENV = create_environment()


def _work(job):
    from bench.episode import run_episode
    task, seed, profile, arm = job
    try:
        return run_episode(task, seed, profile, arm, env=_ENV)
    except Exception as e:  # keep the grid going; the failure is recorded, not hidden
        return {"task": task, "seed": seed, "profile": profile, "arm": arm,
                "oracle_success": False, "believed_success": False, "attempts": 0,
                "retries": 0, "rows": [], "wall_s": 0.0, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50, help="seeds per (task, profile, arm)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", default="results/ablation.json")
    args = ap.parse_args()

    jobs = [(t, s, p, a) for t in args.tasks for p in args.profiles
            for a in args.arms for s in range(args.episodes)]
    # Resumable: each finished episode is appended to <out>.jsonl.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    part = args.out + "l"
    results = []
    if os.path.exists(part):
        with open(part, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
    done = {(r["task"], r["seed"], r["profile"], r["arm"]) for r in results}
    jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already done) on {args.workers} workers", flush=True)
    t0 = time.time()
    sink = open(part, "a", encoding="utf-8")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            results.append(r)
            sink.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            sink.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    sink.close()
    meta = {"episodes": args.episodes, "attempt_budget": 3, "wall_s": time.time() - t0,
            "profiles": {k: vars(v) if hasattr(v, "__dict__") else str(v) for k, v in PROFILES.items()}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "episodes": results}, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    errs = sum(1 for r in results if r.get("error"))
    print(f"done in {time.time() - t0:.0f}s, {errs} episodes with errors -> {args.out}")


if __name__ == "__main__":
    main()
