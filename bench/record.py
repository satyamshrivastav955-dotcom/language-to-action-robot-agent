"""Record chosen episodes as replays for the static web viewer.

    python -m bench.record --results results/ablation.json --out web/data

Each episode is re-run deterministically with video on, per-attempt H.264 clips
and an annotated "what the verifier saw" image are written under
<out>/episodes/<id>/, plus <id>.json and an index.json. The recorded outcome is
checked against the benchmark result for the same (task, seed, profile, arm); a
mismatch is reported, not hidden.
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import cv2

from agent.task_agent import RobotTaskAgent
from agent.verifier import PerceptionVerifier, Verifier
from bench.episode import ARMS, ATTEMPT_BUDGET
from bench.tasks import StaticPlanner, make_task
from env.so101_env import create_environment
from perception.overlay import annotate


def record_episode(task: str, seed: int, profile: str, arm: str, out_dir: str, ep_id: str, env,
                   max_retries: int = ATTEMPT_BUDGET) -> Dict:
    spec = make_task(task, seed)
    ep_dir = os.path.join(out_dir, "episodes", ep_id)
    shutil.rmtree(ep_dir, ignore_errors=True)
    os.makedirs(ep_dir)

    agent = RobotTaskAgent(env=env, planner=StaticPlanner(spec.subtasks), console_output=False,
                           fault_profile=profile, seed=seed, record_frames=True, **ARMS[arm])
    agent.executor.video_dir = ep_dir

    images: Dict[Tuple[int, int], str] = {}
    counters: Dict[int, int] = {}
    if isinstance(agent.verifier, PerceptionVerifier):
        inner = agent.verifier.verify

        def verify_and_snapshot(subtask, execution_result):
            result = inner(subtask, execution_result)
            sid = subtask.get("id", 0)
            counters[sid] = counters.get(sid, 0) + 1
            target = result.get("target_position")
            img = annotate(agent.perception.last_rgbd, agent.perception.last,
                            target, subtask.get("object"))
            name = f"verify_{sid}_{counters[sid]}.jpg"
            cv2.imwrite(os.path.join(ep_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            images[(sid, counters[sid])] = name
            return result
        agent.verifier.verify = verify_and_snapshot

    outcome = agent.execute_instruction(spec.instruction, max_retries=max_retries, save_videos=True,
                                        block_positions=spec.block_positions)
    truth = Verifier(agent.env)
    final = [bool(truth.verify(st, {})["success"]) for st in spec.subtasks]

    subtasks = []
    for row, st in zip(outcome.get("results", []), spec.subtasks):
        attempts = []
        for a in row["attempt_log"]:
            sid, n = row["subtask_id"], a["attempt"]
            vid = f"subtask_{sid}_attempt_{n}.mp4"
            err = a.get("error_vector")
            corr = a.get("correction_applied")
            nxt = a.get("correction_next")
            attempts.append({
                "attempt": n,
                "video": f"episodes/{ep_id}/{vid}" if os.path.exists(os.path.join(ep_dir, vid)) else None,
                "verify_image": f"episodes/{ep_id}/{images[(sid, n)]}" if (sid, n) in images else None,
                "verdict": a["verdict"], "oracle_success": a["oracle_success"],
                "observed_by": a["observed_by"], "reason": a["reason"],
                "error_cm": None if err is None else [round(100 * v, 1) for v in err],
                "correction_applied_cm": None if not corr else [round(100 * v, 1) for v in corr[:2]],
                "correction_next_cm": None if not nxt else [round(100 * v, 1) for v in nxt[:2]],
                "grasp_detected": a["grasp_detected"], "perception_failure": a["perception_failure"],
            })
        subtasks.append({"id": row["subtask_id"], "action": st["action"], "object": st["object"],
                         "target": st["target"], "attempts": attempts, "final_oracle": final[len(subtasks)]})
    doc = {"id": ep_id, "task": task, "seed": seed, "profile": profile, "arm": arm,
           "instruction": spec.instruction, "oracle_success": all(final),
           "believed_success": bool(outcome.get("success")),
           "attempts": sum(len(s["attempts"]) for s in subtasks), "subtasks": subtasks}
    with open(os.path.join(out_dir, "episodes", f"{ep_id}.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return doc


def pick_showcase(episodes: List[Dict]) -> List[Tuple[str, str, int, str, str]]:
    """Choose (id, task, seed, profile, arm) tuples that tell the story, from benchmark results."""
    by = {}
    for e in episodes:
        by[(e["task"], e["seed"], e["profile"], e["arm"])] = e
    picks: List[Tuple[str, str, int, str, str]] = []

    def find(profile, task, pred, label):
        for seed in range(200):
            trio = [by.get((task, seed, profile, a)) for a in
                    ("no_verifier", "verifier_blind", "verifier_corrected")]
            if all(trio) and pred(*trio):
                for arm, ep in zip(("no_verifier", "verifier_blind", "verifier_corrected"), trio):
                    picks.append((f"{label}-{arm}", task, seed, profile, arm))
                return True
        return False

    # Silent failure -> still failing with blind retry -> fixed by the corrected retry.
    find("biased", "stack",
         lambda n, b, c: n["believed_success"] and not n["oracle_success"]
         and not b["oracle_success"] and c["oracle_success"] and c["attempts"] >= 2, "biased-fix")
    # Dropped mid-carry: the verifier notices, the retry re-grasps where the block fell.
    find("slippery", "stack",
         lambda n, b, c: not n["oracle_success"] and b["oracle_success"] and b["attempts"] >= 2, "slip-recover")
    # Plain success, for contrast.
    for seed in range(50):
        e = by.get(("two_step", seed, "nominal", "verifier_corrected"))
        if e and e["oracle_success"] and e["attempts"] == 2:
            picks.append(("nominal-twostep-verifier_corrected", "two_step", seed, "nominal", "verifier_corrected"))
            break
    # An honest failure: the verifier is on and the budget still runs out.
    for seed in range(200):
        e = by.get(("pick_place", seed, "noisy", "verifier_corrected"))
        if e and not e["oracle_success"] and e["attempts"] == 3:
            picks.append(("budget-exhausted-verifier_corrected", "pick_place", seed, "noisy", "verifier_corrected"))
            break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation.json")
    ap.add_argument("--out", default="web/data")
    args = ap.parse_args()
    with open(args.results, encoding="utf-8") as f:
        bench = {(e["task"], e["seed"], e["profile"], e["arm"]): e for e in json.load(f)["episodes"]}
    picks = pick_showcase(list(bench.values()))
    env = create_environment()
    index, mismatches = [], []
    os.makedirs(args.out, exist_ok=True)
    for ep_id, task, seed, profile, arm in picks:
        doc = record_episode(task, seed, profile, arm, args.out, ep_id, env)
        ref = bench.get((task, seed, profile, arm))
        if ref and (ref["oracle_success"] != doc["oracle_success"] or ref["attempts"] != doc["attempts"]):
            mismatches.append(ep_id)
        index.append({k: doc[k] for k in ("id", "task", "seed", "profile", "arm", "instruction",
                                          "oracle_success", "believed_success", "attempts")})
        print(f"recorded {ep_id}: oracle={doc['oracle_success']} attempts={doc['attempts']}", flush=True)
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": index, "replay_mismatches": mismatches}, f, indent=1)
    if mismatches:
        print("WARNING: replays differed from benchmark runs:", mismatches)


if __name__ == "__main__":
    main()

