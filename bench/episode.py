"""Run one episode of (task, seed, fault profile, ablation arm)."""

import time
from typing import Dict

from agent.task_agent import RobotTaskAgent
from agent.verifier import Verifier
from bench.tasks import StaticPlanner, make_task

ARMS: Dict[str, Dict] = {
    # Single attempt; the agent believes whatever happened worked.
    "no_verifier": dict(verifier_mode="none", retry_strategy="blind"),
    # Camera-based verifier; retry = same plan, objects re-perceived.
    "verifier_blind": dict(verifier_mode="perception", retry_strategy="blind"),
    # Camera-based verifier; retry corrects the release from the measured error.
    "verifier_corrected": dict(verifier_mode="perception", retry_strategy="corrected"),
    # Same, but only 40% of the measured error is fed back (damped against noise).
    "verifier_damped": dict(verifier_mode="perception", retry_strategy="corrected", retry_alpha=0.4),
    # Upper bound on verification quality: judge from simulator truth.
    "oracle_verifier_corrected": dict(verifier_mode="oracle", retry_strategy="corrected"),
}
ATTEMPT_BUDGET = 3


def run_episode(task: str, seed: int, profile: str, arm: str, env=None,
                record_frames: bool = False, save_videos: bool = False,
                planner=None, max_retries: int = ATTEMPT_BUDGET) -> Dict:
    spec = make_task(task, seed)
    t0 = time.time()
    agent = RobotTaskAgent(env=env, planner=planner or StaticPlanner(spec.subtasks), console_output=False,
                           fault_profile=profile, seed=seed, record_frames=record_frames,
                           **ARMS[arm])
    try:
        outcome = agent.execute_instruction(spec.instruction, max_retries=max_retries,
                                            save_videos=save_videos,
                                            block_positions=spec.block_positions)
        # Ground truth, read once, after the episode, by the evaluator only.
        truth = Verifier(agent.env)
        final = [bool(truth.verify(st, {})["success"]) for st in spec.subtasks]
        rows = outcome.get("results", [])
        return {
            "task": task, "seed": seed, "profile": profile, "arm": arm,
            "instruction": spec.instruction, "subtasks": spec.subtasks,
            "oracle_success": all(final),
            "oracle_per_subtask": final,
            "believed_success": bool(outcome.get("success")),
            "attempts": sum(r["attempts"] for r in rows),
            "retries": sum(r["retries"] for r in rows),
            "rows": rows,
            "wall_s": time.time() - t0,
            "error": outcome.get("error"),
        }
    finally:
        if env is None:
            agent.close()
