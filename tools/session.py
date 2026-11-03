"""A persistent robot session behind the typed tools."""

from typing import Dict, Optional

from agent.task_agent import RobotTaskAgent
from bench.tasks import StaticPlanner, make_task
from env.so101_env import create_environment


class RobotSession:
    def __init__(self, fault_profile: str = "nominal", seed: int = 0, planner=None):
        self.env = create_environment()
        self.fault_profile = fault_profile
        self.seed = seed
        self._llm_planner = planner
        self.agent: Optional[RobotTaskAgent] = None
        self.reset_scene()

    def _make_agent(self, planner) -> RobotTaskAgent:
        return RobotTaskAgent(env=self.env, planner=planner, console_output=False,
                              fault_profile=self.fault_profile, seed=self.seed, record_frames=False)

    # --- tools -----------------------------------------------------------
    def reset_scene(self, seed: Optional[int] = None) -> Dict:
        positions = make_task("pick_place", seed).block_positions if seed is not None else None
        self.env.reset(positions)
        self.env.set_faults(self.fault_profile)
        return {"ok": True, "seed": seed}

    def set_fault_profile(self, profile: str) -> Dict:
        self.fault_profile = profile
        self.env.set_faults(profile)
        return {"ok": True, "profile": profile}

    def get_scene(self) -> Dict:
        agent = self._make_agent(StaticPlanner([]))
        dets = agent.perception.perceive()
        return {"blocks": {n: [round(float(v), 4) for v in d.xyz] for n, d in dets.items()}}

    def _run(self, subtasks, instruction, max_attempts, planner=None) -> Dict:
        agent = self._make_agent(planner or StaticPlanner(subtasks))
        out = agent.execute_instruction(instruction, max_retries=max_attempts or 3,
                                        save_videos=False, reset_world=False)
        if out.get("clarification"):
            return {"success": False, "clarification": out["clarification"]}
        return {"success": out.get("success", False), "error": out.get("error"),
                "subtasks": [{"object": r["object"], "target": r["target"], "attempts": r["attempts"],
                              "verified": r["believed_success"], "attempt_log": [
                                  {k: a[k] for k in ("attempt", "verdict", "error_vector", "reason")}
                                  for a in r["attempt_log"]]} for r in out.get("results", [])]}

    def pick_and_place(self, object: str, target: str, max_attempts: int = 3) -> Dict:
        return self._run([dict(id=1, action="pick_and_place", object=object, target=target)],
                         f"pick_and_place {object} {target}", max_attempts)

    def stack(self, object: str, target: str, max_attempts: int = 3) -> Dict:
        return self._run([dict(id=1, action="stack", object=object, target=target)],
                         f"stack {object} on {target}", max_attempts)

    def run_instruction(self, instruction: str, max_attempts: int = 3) -> Dict:
        from agent.planner import create_planner
        planner = self._llm_planner or create_planner()
        return self._run(None, instruction, max_attempts, planner=planner)
