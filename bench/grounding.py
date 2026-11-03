"""Perception accuracy: colour segmentation vs VLM point grounding, against ground truth.

    python -m bench.grounding --scenes 8 --models qwen3-vl:2b qwen3-vl:4b --out results/grounding.json

For each random scene the same RGB-D frame goes to every estimator. Error is the
3-D distance (mm) between the estimated block centre and the simulator's true
centre. Ground truth is read here, by the evaluator, and nowhere in perception/.
"""

import argparse
import json
import time

import numpy as np

from bench.tasks import make_task
from env.so101_env import create_environment
from perception import SegmentationEstimator
from perception.vlm import OllamaVLMGrounder


def summarise(errs, found, total, latency):
    e = np.array(errs) if errs else np.array([np.nan])
    return {"detect_rate": found / total, "median_err_mm": float(np.nanmedian(e)),
            "p90_err_mm": float(np.nanpercentile(e, 90)), "max_err_mm": float(np.nanmax(e)),
            "frac_within_10mm": float(np.mean(e < 10)) if errs else 0.0,
            "median_latency_s": float(np.median(latency)) if latency else 0.0, "n": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out", default="results/grounding.json")
    args = ap.parse_args()

    env = create_environment()
    estimators = {"segmentation": SegmentationEstimator()}
    estimators.update({m: OllamaVLMGrounder(model=m) for m in args.models})
    stats = {k: {"errs": [], "found": 0, "total": 0, "lat": []} for k in estimators}

    for seed in range(args.scenes):
        spec = make_task("pick_place", seed)
        env.reset(spec.block_positions)
        env.park_arm()
        rgbd = env.capture_rgbd(rng=np.random.default_rng(seed))
        for name, est in estimators.items():
            t0 = time.time()
            dets = est.estimate(rgbd)
            lat = time.time() - t0
            st = stats[name]
            st["lat"].append(lat)
            for blk in env.object_names:
                st["total"] += 1
                if blk in dets:
                    st["found"] += 1
                    st["errs"].append(1000 * float(np.linalg.norm(dets[blk].xyz - env.get_object_pose(blk))))
        print(f"scene {seed} done", flush=True)

    out = {k: summarise(v["errs"], v["found"], v["total"], v["lat"]) for k, v in stats.items()}
    for k, v in out.items():
        print(f"{k:14s} detect={v['detect_rate']:.0%} median={v['median_err_mm']:.1f}mm "
              f"within10mm={v['frac_within_10mm']:.0%} latency/scene={v['median_latency_s']:.1f}s")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
