"""The verifier ablation on ManiSkill3's PickCube-v1 (official env, official success metric).

    docker build -f docker/maniskill/Dockerfile -t robot-agent-maniskill .
    docker run --rm -v "$PWD/results:/work/results" robot-agent-maniskill maniskill_bench.run \
        --episodes 30 --out results/maniskill.json

Same protocol as the MuJoCo study, on a standard task:
  * the agent localises the cube from the RGB-D camera only (colour mask + depth);
  * the controller is ManiSkill's own motion planner (mplib), with faults injected
    into where it grasps / where it holds the cube (same profiles as env/faults.py);
  * arms: no verifier / camera verifier + blind retry / camera verifier + corrected retry;
  * success = the environment's own `evaluate()["success"]` at the end.
The goal position is the task specification (an input), not a perception output.
"""

import argparse
import json
import os
import time
from typing import Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
import sapien

# SAPIEN renders on the software Vulkan device (llvmpipe) only through its default
# device; ManiSkill asks for "cpu" by name, which SAPIEN rejects. Ignore the name.
_RenderSystem = sapien.render.RenderSystem
sapien.render.RenderSystem = lambda *a, **k: _RenderSystem()

import mani_skill.envs  # noqa: E402,F401  (registers environments)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver  # noqa: E402

from maniskill_bench.stats import mcnemar_exact, wilson  # noqa: E402

PROFILES = {
    "nominal": dict(grasp_noise=0.0, place_noise=0.0, bias=(0.0, 0.0)),
    "noisy": dict(grasp_noise=0.020, place_noise=0.020, bias=(0.0, 0.0)),
    "biased": dict(grasp_noise=0.0, place_noise=0.0, bias=(0.045, -0.030)),
}
ARMS = {
    "no_verifier": dict(verify=False, corrected=False),
    "verifier_blind": dict(verify=True, corrected=False),
    "verifier_corrected": dict(verify=True, corrected=True),
}
CUBE_HALF = 0.02
GOAL_TOL = 0.025  # the environment's own placement tolerance
ALPHA = 0.85
BUDGET = 3


# ---- perception: RGB-D only -------------------------------------------------
def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def capture(env) -> Dict:
    # Stepping in obs_mode="state" keeps the planner fast; render only when we look.
    obs = env.unwrapped._get_obs_with_sensor_data(env.unwrapped.get_info(), apply_texture_transforms=True)
    cam = obs["sensor_data"]["base_camera"]
    par = obs["sensor_param"]["base_camera"]
    return {"rgb": _np(cam["rgb"])[0].astype(np.uint8),
            "depth": _np(cam["depth"])[0, ..., 0].astype(np.float32) / 1000.0,  # mm -> m
            "K": _np(par["intrinsic_cv"])[0], "E": _np(par["extrinsic_cv"])[0]}


def locate_cube(frame: Dict, rng: np.random.Generator, rgb_noise: float = 2.0) -> Optional[Dict]:
    """Cube centre and yaw from a red colour mask + depth. None if not visible."""
    rgb = np.clip(frame["rgb"].astype(np.float32) + rng.normal(0, rgb_noise, frame["rgb"].shape), 0, 255)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = ((r > 140) & (g < 0.35 * r) & (b < 0.35 * r)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 12:
        return None
    vs, us = np.nonzero(labels == k)
    d = frame["depth"][vs, us]
    ok = d > 0
    us, vs, d = us[ok], vs[ok], d[ok]
    K, E = frame["K"], frame["E"]
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    cam_pts = np.stack([x, y, d], axis=1)
    R, t = E[:, :3], E[:, 3]
    world = (cam_pts - t) @ R          # row-wise R^T (X - t)
    top = world[:, 2] >= np.percentile(world[:, 2], 70)   # highest 30%: the top face
    top_pts = world[top]
    centre = np.array([top_pts[:, 0].mean(), top_pts[:, 1].mean(), top_pts[:, 2].mean() - CUBE_HALF])
    (_, _), (_, _), ang = cv2.minAreaRect(top_pts[:, :2].astype(np.float32))
    return {"xyz": centre, "yaw": np.deg2rad(ang), "n_pixels": int(len(us))}


# ---- one episode -------------------------------------------------------------
def run_episode(env, seed: int, profile: str, arm: str) -> Dict:
    prof, cfg = PROFILES[profile], ARMS[arm]
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    goal = _np(u.goal_site.pose.p)[0].astype(float)
    planner = PandaArmMotionPlanningSolver(env, debug=False, vis=False, base_pose=u.agent.robot.pose,
                                           visualize_target_grasp_pose=False, print_env_info=False)
    sensor_rng = np.random.default_rng([seed, 11])
    correction = np.zeros(2)
    log = []
    believed = False
    try:
        for attempt in range(1, BUDGET + 1):
            rng = np.random.default_rng([seed, 5, attempt])          # faults: identical across arms
            grasp_err = rng.standard_normal(2) * prof["grasp_noise"]
            place_err = rng.standard_normal(2) * prof["place_noise"] + np.array(prof["bias"])

            planner.open_gripper()
            frame = capture(env)
            det = locate_cube(frame, sensor_rng)
            if det is None:
                log.append({"attempt": attempt, "verdict": False, "reason": "cube not visible"})
                continue
            c = det["xyz"]
            closing = np.array([np.cos(det["yaw"]), np.sin(det["yaw"]), 0.0])
            grasp_c = np.array([c[0] + grasp_err[0], c[1] + grasp_err[1], c[2]])
            grasp_pose = u.agent.build_grasp_pose(np.array([0, 0, -1.0]), closing, grasp_c)
            planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, -0.08]))
            planner.move_to_pose_with_screw(grasp_pose)
            planner.close_gripper()
            tgt = goal.copy()
            tgt[:2] += correction + place_err
            planner.move_to_pose_with_screw(sapien.Pose(tgt, grasp_pose.q))

            entry = {"attempt": attempt, "correction": correction.tolist()}
            if not cfg["verify"]:
                believed = True
                entry["verdict"] = None
                log.append(entry)
                break
            det2 = locate_cube(capture(env), sensor_rng)
            if det2 is None:
                entry.update(verdict=False, reason="cube not visible after the attempt")
            else:
                err = det2["xyz"] - goal
                ok = float(np.linalg.norm(err)) < GOAL_TOL
                entry.update(verdict=bool(ok), error_cm=[100 * float(err[0]), 100 * float(err[1])])
                if ok:
                    believed = True
                elif cfg["corrected"] and np.linalg.norm(err[:2]) < 0.12:
                    correction = correction - ALPHA * err[:2]
            log.append(entry)
            if entry["verdict"]:
                break
        success = bool(_np(u.evaluate()["success"])[0])
    finally:
        planner.close()
    return {"seed": seed, "profile": profile, "arm": arm, "success": success, "believed": believed,
            "attempts": len(log), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PickCube-v1")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--size", type=int, default=512, help="camera width/height in pixels")
    ap.add_argument("--out", default="results/maniskill.json")
    args = ap.parse_args()

    env = gym.make(args.env, num_envs=1, obs_mode="state", control_mode="pd_joint_pos",
                   render_mode=None, sensor_configs=dict(width=args.size, height=args.size))
    results, t0 = [], time.time()
    for profile in args.profiles:
        for arm in args.arms:
            for seed in range(args.episodes):
                try:
                    r = run_episode(env, seed, profile, arm)
                except Exception as e:  # recorded, not hidden
                    r = {"seed": seed, "profile": profile, "arm": arm, "success": False, "believed": False,
                         "attempts": 0, "log": [], "error": f"{type(e).__name__}: {e}"}
                results.append(r)
            ep = [r for r in results if r["profile"] == profile and r["arm"] == arm]
            k = sum(r["success"] for r in ep)
            lo, hi = wilson(k, len(ep))
            print(f"{profile:8s} {arm:20s} {k}/{len(ep)} [{lo:.0%}-{hi:.0%}]  {time.time() - t0:.0f}s", flush=True)

    summary = {}
    for profile in args.profiles:
        by = {a: {r["seed"]: r["success"] for r in results if r["profile"] == profile and r["arm"] == a}
              for a in args.arms}
        row = {}
        for a in args.arms:
            k, n = sum(by[a].values()), len(by[a])
            lo, hi = wilson(k, n)
            row[a] = {"k": k, "n": n, "rate": k / n if n else 0, "lo": lo, "hi": hi,
                      "false_success": sum(1 for r in results if r["profile"] == profile and r["arm"] == a
                                           and r["believed"] and not r["success"])}
        pairs = {}
        for a, b in (("verifier_blind", "no_verifier"), ("verifier_corrected", "verifier_blind")):
            if a in by and b in by:
                oa = sum(1 for s in by[a] if by[a][s] and not by[b].get(s))
                ob = sum(1 for s in by[a] if by[b].get(s) and not by[a][s])
                pairs[f"{a}_vs_{b}"] = {"only_first": oa, "only_second": ob, "p": mcnemar_exact(oa, ob)}
        summary[profile] = {"arms": row, "paired": pairs}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"env": args.env, "episodes": args.episodes, "attempt_budget": BUDGET,
                   "summary": summary, "results": results}, f, default=float)


if __name__ == "__main__":
    main()
