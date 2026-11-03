import numpy as np
import os
import time
from typing import Dict, Optional, List


from env.so101_env import SO101Environment
from configs.settings import (
    MAX_STEPS_PER_ROLLOUT,
    HOME_POSITION_DEG,
    TARGET_REGIONS,
    BLOCK_SIZE,
)


class MockSmolVLAPolicy:
    """Scripted stand-in for a VLA policy.

    IMPORTANT - this policy does not perceive or act. `begin_attempt` draws the
    outcome from a Bernoulli trial *before* the rollout starts, and the env then
    teleports the block accordingly. The joint commands returned by get_action
    are home-position noise and have no effect on success. Anything reported by
    a run using this policy is a simulation of an agent loop, not a measurement
    of manipulation skill. See `Executor.describe_policy`.
    """

    def __init__(self, device: str = "cpu", success_probability: float = 0.65,
                 retry_bonus: float = 0.1):
        self.device = device
        self.model_name = "mock_smolvla"
        self.success_probability = success_probability
        self.retry_bonus = retry_bonus
        self._will_succeed = False
        self._task_set = False
        self._last_probability = success_probability

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu"):
        print(f"[MockPolicy] Loading policy from {model_path}")
        return cls(device=device)

    def begin_attempt(self, retry_params: Optional[Dict] = None):
        """Draw this attempt's predetermined outcome.

        Retries get a modest bump so the diagnosis-then-adjust loop has a
        visible effect; without it, retrying is pure resampling and the retry
        controller's output is decorative.
        """
        probability = self.success_probability
        if retry_params:
            extra_attempts = max(0, int(retry_params.get("retry_count", 1)) - 1)
            probability = min(0.95, probability + extra_attempts * self.retry_bonus)

        self._last_probability = probability
        self._will_succeed = np.random.random() < probability
        self._task_set = True
        print(f"[MockPolicy] SCRIPTED outcome (p={probability:.2f}): "
              f"{'SUCCESS' if self._will_succeed else 'FAIL'}")

    def get_action(self, observation: np.ndarray, language_prompt: str) -> np.ndarray:
        base_angles = np.array(HOME_POSITION_DEG) * np.pi / 180.0
        perturbation = np.random.randn(6) * 0.02
        return base_angles + perturbation

    def will_succeed(self) -> bool:
        return self._will_succeed

    def last_probability(self) -> float:
        return self._last_probability

    def reset(self):
        self._task_set = False
        self._will_succeed = False


class ScriptedMuJoCoPolicy:
    """Waypoint controller that actually manipulates blocks in MuJoCo.

    Not learned, and it does not claim to be - but unlike MockSmolVLAPolicy the
    outcome is NOT predetermined. It drives the real arm through approach ->
    descend -> grasp -> lift -> transfer -> release, and whether the block ends
    up in the bin is decided by physics (contact, friction, gravity). The
    verifier then reads the resulting mjData position. A bad grasp genuinely
    fails.

    Stands in for SmolVLA until trained weights are plugged in.
    """

    APPROACH_Z = 0.20
    GRASP_Z_OFFSET = 0.005
    CARRY_Z = 0.24
    RELEASE_Z = 0.13
    # Joint-velocity norm (rad/s, summed over the 6 arm DOFs) below which the
    # arm counts as still. At the end of "lower" the measured norm is ~50, so
    # this is roughly two orders of magnitude down - low enough that opening
    # the gripper no longer imparts a kick to the block underneath it.
    SETTLE_SPEED = 1.0

    def __init__(self, env, settle_steps: int = 14):
        self.env = env
        self.model_name = "scripted_mujoco_waypoint"
        self.settle_steps = settle_steps
        self._plan: List[Dict] = []
        self._idx = 0
        self._phase_step = 0

    def reset(self):
        self._plan = []
        self._idx = 0
        self._phase_step = 0

    def begin_attempt(self, action: str, obj: str, target: str,
                      retry_params: Optional[Dict] = None, pose_fn=None):
        """Build the waypoint plan for this attempt.

        `pose_fn(name)` supplies object poses: estimates from perception in the
        default pipeline, simulator truth only when none is given. Injected
        faults (env.faults) perturb where the arm ACTUALLY grasps and releases,
        after the agent has decided where to; `place_correction` in
        retry_params shifts the commanded release to compensate.
        """
        retry_params = retry_params or {}
        pose_fn = pose_fn or self.env.get_object_pose
        correction = np.array(retry_params.get("place_correction", [0.0, 0.0, 0.0]), dtype=float)

        obj_pos = np.asarray(pose_fn(obj), dtype=float)

        # Fault draws happen in a fixed order so arms stay paired (see env.faults).
        faults = getattr(self.env, "faults", None)
        rng = getattr(self.env, "_fault_rng", np.random.default_rng(0))
        grasp_err = rng.standard_normal(2) * (faults.grasp_noise_std if faults else 0.0)
        place_err = rng.standard_normal(2) * (faults.place_noise_std if faults else 0.0)
        if faults:
            place_err = place_err + np.asarray(faults.place_bias, dtype=float)

        if target == "bin":
            place = np.asarray(TARGET_REGIONS["bin"], dtype=float).copy()
            release_z = self.RELEASE_Z
        else:
            tgt = np.asarray(pose_fn(target), dtype=float)
            place = tgt.copy()
            place[2] = tgt[2] + BLOCK_SIZE  # stack on top
            # The carried block hangs ~6mm below the grip site, so the site has
            # to sit that much higher than the intended resting height, plus a
            # few mm clearance so the block is set down rather than driven into
            # the tower (which flicks it off).
            release_z = tgt[2] + BLOCK_SIZE + 0.006 + 0.008
        self._stack_target = None if target == "bin" else target

        self.intended_place = place.copy()           # what the task asks for
        place[:2] = place[:2] + correction[:2]       # what the agent commands
        place[:2] = place[:2] + place_err            # where the arm actually goes
        grasp_xy = obj_pos[:2] + grasp_err

        self._plan = [
            {"name": "approach", "xyz": [grasp_xy[0], grasp_xy[1], self.APPROACH_Z],
             "grip": 0.0, "steps": 22},
            {"name": "align", "xyz": [grasp_xy[0], grasp_xy[1], 0.11],
             "grip": 0.0, "steps": 20},
            {"name": "descend", "xyz": [grasp_xy[0], grasp_xy[1], obj_pos[2] + self.GRASP_Z_OFFSET],
             "grip": 0.0, "steps": 22},
            {"name": "grasp", "xyz": [grasp_xy[0], grasp_xy[1], obj_pos[2] + self.GRASP_Z_OFFSET],
             "grip": 1.0, "steps": 22},
            {"name": "lift", "xyz": [grasp_xy[0], grasp_xy[1], self.CARRY_Z],
             "grip": 1.0, "steps": 26},
            {"name": "transfer", "xyz": [place[0], place[1], self.CARRY_Z],
             "grip": 1.0, "steps": 34},
            # Descend in two stages: get directly above the target first, then
            # drop straight down. A single diagonal move clips the target block
            # with the wrist and knocks it across the table before release.
            {"name": "descend_over", "xyz": [place[0], place[1], release_z + 0.075],
             "grip": 1.0, "steps": 20},
            {"name": "lower", "xyz": [place[0], place[1], release_z],
             "grip": 1.0, "steps": 55},
            # Hold the lowered pose, gripper still closed, until the arm stops
            # moving. The position controller reaches the target with a large
            # residual joint velocity still in the links; if the gripper opens
            # and the arm retreats while that energy is present, the open
            # fingers clip the just-placed block and fling it across the table.
            # How hard it gets clipped is chaotic - it depends on the exact
            # velocity phase, which differs between a first pick and a second
            # one - so a fixed settle count is unreliable; we wait on measured
            # joint velocity instead. See get_action's "settle" branch.
            # "hold" means: keep the joint targets lower ended on. Re-solving
            # IK here would throw away lower's closed-loop correction and
            # command the raw goal again, which walks the arm ~2cm back off
            # the target while the block is still gripped.
            {"name": "settle", "xyz": [place[0], place[1], release_z],
             "grip": 1.0, "steps": 60, "settle": True, "hold": True},
            {"name": "release", "xyz": [place[0], place[1], release_z],
             "grip": 0.0, "steps": 40, "hold": True},
            # Retreat lifts straight up from wherever the release actually
            # happened; a sideways move here drags the just-placed block.
            {"name": "retreat", "xyz": [place[0], place[1], self.CARRY_Z],
             "grip": 0.0, "steps": 16, "vertical_from_release": True},
        ]
        self._idx = 0
        self._phase_step = 0
        print(f"[ScriptedPolicy] Plan: grasp {obj} at "
              f"{np.round(obj_pos, 3).tolist()} -> place at {np.round(place, 3).tolist()}")

    def get_action(self, observation: np.ndarray, language_prompt: str = "") -> np.ndarray:
        """Return 7 values: 6 joint targets (rad) + gripper close fraction."""
        if not self._plan:
            return np.append(np.asarray(observation, dtype=float)[:6], 0.0)

        phase = self._plan[min(self._idx, len(self._plan) - 1)]

        # Cache IK per phase - solving every step is wasteful and jitters.
        if self._phase_step == 0 and not phase.get("hold"):
            if phase.get("vertical_from_release"):
                here = self.env.get_end_effector_pose()
                phase["xyz"] = [float(here[0]), float(here[1]), self.CARRY_Z]
            self._cached_q = self.env.solve_ik(phase["xyz"])

        # Position actuators settle short of the IK solution under gravity, by
        # ~2cm in most of the workspace but over 4cm near the base - enough to
        # set a block down off the edge of a stack target and tip it over. On
        # the placement phases, re-solve from where the arm ACTUALLY is and
        # correct the residual instead of trusting the one-shot solution.
        elif phase["name"] == "lower" and self._phase_step % 6 == 0:
            goal = np.asarray(phase["xyz"], dtype=float)
            # Deliberately NOT re-reading the stack target here. Tracking it
            # during descent becomes a shove: the carried block nudges the
            # target, the goal chases it, and both end up across the table.
            # The frozen plan pose plus closed-loop correction on our OWN
            # position is what actually lands the block.
            actual = self.env.get_end_effector_pose()
            err = goal - actual
            if np.linalg.norm(err) > 0.005:
                self._cached_q = self.env.solve_ik(goal + err * 0.9, restarts=0)

        self._phase_step += 1

        # The settle phase is the one phase with a data-driven end: it holds
        # until the arm is measurably still, and its step count is only an
        # upper bound. Everything else runs its full fixed budget.
        done = self._phase_step >= phase["steps"]
        if phase.get("settle") and self._phase_step >= 12 and not done:
            done = self._arm_speed() < self.SETTLE_SPEED

        if done and self._idx < len(self._plan) - 1:
            self._idx += 1
            self._phase_step = 0

        return np.append(self._cached_q[:6], phase["grip"])

    def _arm_speed(self) -> float:
        """Joint-velocity norm over the 6 arm DOFs, or 0.0 without physics.

        The mock environment has no qvel; returning 0 there makes the settle
        phase exit at its minimum hold, which is the right behaviour when
        there is no physics to settle.
        """
        data = getattr(self.env, "data", None)
        adr = getattr(self.env, "_arm_dof_adr", None)
        if data is None or adr is None:
            return 0.0
        return float(np.linalg.norm([data.qvel[a] for a in adr]))

    def is_done(self) -> bool:
        last = self._idx >= len(self._plan) - 1
        return last and self._phase_step >= self._plan[-1]["steps"] if self._plan else True

    def current_phase(self) -> str:
        if not self._plan:
            return "idle"
        return self._plan[min(self._idx, len(self._plan) - 1)]["name"]


class Executor:
    
    def __init__(self, policy: Optional[object] = None, env: Optional[SO101Environment] = None,
                 use_mock_policy: bool = True, perception=None, record_frames: bool = True):
        self.env = env if env else SO101Environment()
        # With a PerceptionSystem the executor acts on camera estimates; without
        # one it reads simulator poses (oracle mode, for upper-bound comparisons).
        self.perception = perception
        self.record_frames = record_frames
        self.video_dir: Optional[str] = None  # defaults to configs.settings.VIDEOS_DIR

        # policy_type must be set on EVERY branch - it drives the rollout path
        # below. Previously it was only assigned in the elif/else, so passing an
        # explicit policy raised AttributeError on the first step.
        if policy is not None:
            self.policy = policy
            self.policy_type = "mock" if isinstance(policy, MockSmolVLAPolicy) else "real"
            if isinstance(policy, ScriptedMuJoCoPolicy):
                self.policy_type = "scripted_sim"
        elif use_mock_policy:
            self.policy = MockSmolVLAPolicy()
            self.policy_type = "mock"
        elif getattr(self.env, "is_physics_backed", False):
            # Real run + real physics: drive the arm with the waypoint
            # controller. The trajectory is hand-written; the outcome is
            # still determined by the simulation, not predetermined.
            self.policy = ScriptedMuJoCoPolicy(self.env)
            self.policy_type = "scripted_sim"
            print("[Executor] Using scripted MuJoCo waypoint controller "
                  "(physics-determined outcomes)")
        else:
            # No physics available and mock not requested: fall back to the
            # mock policy rather than pretending a learned policy exists.
            print("[Executor] MuJoCo unavailable; falling back to mock policy")
            self.policy = MockSmolVLAPolicy()
            self.policy_type = "mock"

        self.current_frames: List[np.ndarray] = []
        self.video_path: Optional[str] = None

    def describe_policy(self) -> Dict:
        """How this run's outcomes are produced, for logs and the UI.

        A reader of a trace should never have to guess whether a SUCCESS came
        from a policy or from a coin flip.
        """
        if self.policy_type == "mock":
            return {
                "type": "mock",
                "name": getattr(self.policy, "model_name", "mock"),
                "outcomes_are_scripted": True,
                "success_probability": getattr(self.policy, "success_probability", None),
                "note": ("Outcomes are drawn from a Bernoulli trial before each rollout "
                         "and the block is teleported to match. Joint commands do not "
                         "affect success. Not a measurement of manipulation skill."),
            }

        if self.policy_type == "scripted_sim":
            return {
                "type": "scripted_sim",
                "name": getattr(self.policy, "model_name", "scripted_mujoco_waypoint"),
                # Trajectory is hand-written, but the OUTCOME is not: the block
                # moves only through contact, and a bad grasp really fails.
                "outcomes_are_scripted": False,
                "physics_backed": True,
                "note": ("Hand-written waypoint controller (not a learned policy) driving "
                         "the real MuJoCo arm. Success is determined by physics - contact, "
                         "friction and gravity - and read back from mjData, not predetermined."),
            }

        loaded = getattr(self.policy, "model", None) is not None
        return {
            "type": "real",
            "name": getattr(self.policy, "model_path", "external"),
            # An externally supplied policy whose load failed silently falls
            # back to noise actions, which is scripted-adjacent - say so
            # rather than claiming a real policy is driving.
            "outcomes_are_scripted": not loaded,
            "weights_loaded": loaded,
            "note": ("Externally supplied policy driving the rollout." if loaded else
                     "Model weights failed to load; actions are home-position noise."),
        }

    def create_subgoal_prompt(self, action: str, obj: str, target: str) -> str:
        action_templates = {
            "pick_and_place": f"Pick up the {obj} and place it in the {target}",
            "stack": f"Pick up the {obj} and stack it on top of the {target}",
            "push": f"Push the {obj} towards the {target}",
        }
        return action_templates.get(action, f"Move the {obj} to the {target}")
    
    def run_subtask(self, subtask: Dict, max_steps: int = MAX_STEPS_PER_ROLLOUT,
                    save_video: bool = False, video_name: Optional[str] = None,
                    retry_params: Optional[Dict] = None) -> Dict:
        start_time = time.time()

        action = subtask.get("action", "pick_and_place")
        obj = subtask.get("object", "red block")
        target = subtask.get("target", "bin")

        prompt = self.create_subgoal_prompt(action, obj, target)
        print(f"[Executor] Running subtask: {prompt}")
        if retry_params and retry_params.get("retry_count", 1) > 1:
            print(f"[Executor] Retry adjustments: {retry_params}")

        self.current_frames = []
        self._prev_obs = None
        self._settled_steps = 0

        # NOTE: no env.reset() here. The environment is reset once per
        # instruction by RobotTaskAgent.run; each subtask continues from the
        # world state the previous one produced, so multi-step instructions
        # genuinely compose ("put red in bin, THEN stack blue on green" runs
        # subtask 2 with red still in the bin). Retries of a failed subtask
        # are rewound by the caller via env.restore_state, not by a factory
        # reset. Only the ARM is re-homed between subtasks (blocks untouched):
        # each pick starts from the canonical configuration, like a real
        # manipulation cell returning to home between jobs.
        pose_fn = None
        target_pos_est = None
        if self.perception is not None and self.policy_type == "scripted_sim":
            # Look first (arm parked out of view), then act on what was seen.
            self.perception.perceive(park=True)
            pose_fn = self.perception.pose
            needed = [obj] + ([] if target == "bin" else [target])
            if any(self.perception.pose(n) is None for n in needed):
                return {"success": None, "frames": [], "video_path": None,
                        "initial_pos": None, "final_pos": None, "object_moved": False,
                        "grasp_detected": False, "perception_failure": True,
                        "duration": time.time() - start_time, "steps": 0, "prompt": prompt,
                        "retry_params": retry_params or {}, "outcome_source": "policy"}
            if target != "bin":
                target_pos_est = np.array(self.perception.pose(target))

        if hasattr(self.env, "reset_arm"):
            self.env.reset_arm()

        if hasattr(self.env, "grasp_attached"):
            self.env.grasp_attached = False

        if self.policy_type == "mock" and hasattr(self.policy, 'begin_attempt'):
            self.policy.begin_attempt(retry_params)
        elif self.policy_type == "scripted_sim":
            self.policy.begin_attempt(action, obj, target, retry_params, pose_fn=pose_fn)

        initial_object_pos = (np.array(pose_fn(obj)) if pose_fn else self.env.get_object_pose(obj))

        obs = self.env._get_observation()

        # A slower, more careful retry is modelled as a longer rollout budget.
        speed_factor = float((retry_params or {}).get("speed_factor", 1.0) or 1.0)
        effective_steps = max(1, int(max_steps / max(speed_factor, 0.1)))

        if self.policy_type == "scripted_sim":
            # The waypoint plan needs its full length to finish the place; a
            # short max_steps would cut the arm off mid-carry. 360 covers the
            # full 11-phase plan (~307 steps) with margin.
            effective_steps = max(effective_steps, 360)

        for step in range(effective_steps):
            if self.policy_type == "real":
                frame = self.env.render()
                action_cmd = self.policy.get_action(obs, frame, prompt)
            else:
                action_cmd = self.policy.get_action(obs, prompt)

            obs, reward, done, info = self.env.step(action_cmd)

            if self.record_frames:
                frame = self.env.render(
                    phase=self.policy.current_phase() if hasattr(self.policy, 'current_phase') else None)
                if frame is not None:
                    self.current_frames.append(frame)

            if self.policy_type == "scripted_sim":
                if self.policy.is_done():
                    break
            elif self._check_motion_converged(obs, info):
                break

        if self.policy_type == "scripted_sim":
            # Let the block settle after release before reading its pose,
            # otherwise the verifier measures it mid-fall.
            for i in range(40):
                self.env.step(np.append(obs[:6], 0.0))
                if self.record_frames and i % 4 == 0:
                    frame = self.env.render()
                    if frame is not None:
                        self.current_frames.append(frame)
        
        final_object_pos = self.env.get_object_pose(obj)

        will_succeed = False
        if self.policy_type == "mock" and hasattr(self.policy, 'will_succeed'):
            will_succeed = self.policy.will_succeed()

        policy_info = self.describe_policy()
        outcome_source = "scripted" if policy_info["outcomes_are_scripted"] else "policy"

        if policy_info["outcomes_are_scripted"] and hasattr(self.env, 'simulate_pick_and_place'):
            self.env.simulate_pick_and_place(obj, target, will_succeed)
            final_object_pos = self.env.get_object_pose(obj)

        for _ in range(5 if self.record_frames else 0):
            frame = self.env.render()
            if frame is not None:
                self.current_frames.append(frame)

        elapsed_time = time.time() - start_time

        video_path = None
        if save_video and self.current_frames:
            video_path = self._save_video(video_name or f"subtask_{subtask.get('id', 0)}")

        return {
            "success": None,
            "frames": self.current_frames,
            "video_path": video_path,
            "initial_pos": initial_object_pos,
            "final_pos": final_object_pos,  # simulator truth: evaluation/logging only
            # Gripper-closure feedback: did the grasp take hold of a block? The
            # verifier, not this flag, decides whether the TASK succeeded.
            "grasp_detected": bool(getattr(self.env, "grasp_attached", False)),
            "target_pos_est": target_pos_est,
            "intended_place": getattr(self.policy, "intended_place", None),
            "object_moved": not np.allclose(initial_object_pos, final_object_pos, atol=0.01),
            "duration": elapsed_time,
            "steps": len(self.current_frames),
            "perception_failure": False,
            "prompt": prompt,
            "retry_params": retry_params or {},
            # "scripted" means the block was teleported to a predetermined
            # result, not moved by the policy. Downstream logs surface this.
            "outcome_source": outcome_source,
        }
    
    def _check_motion_converged(self, obs: np.ndarray, info: Dict) -> bool:
        """True once joint motion has settled, to end a rollout early.

        Compares against the previous observation; requires several consecutive
        near-identical readings so a momentary pause does not end the rollout.
        """
        prev = getattr(self, "_prev_obs", None)
        self._prev_obs = np.asarray(obs, dtype=float).copy()

        if prev is None:
            self._settled_steps = 0
            return False

        if np.allclose(prev, self._prev_obs, atol=1e-4):
            self._settled_steps = getattr(self, "_settled_steps", 0) + 1
        else:
            self._settled_steps = 0

        return self._settled_steps >= 5
    
    def _save_video(self, name: str, fps: int = 25) -> Optional[str]:
        from configs.settings import VIDEOS_DIR

        if not self.current_frames:
            return None

        vdir = self.video_dir or VIDEOS_DIR
        os.makedirs(vdir, exist_ok=True)
        video_path = os.path.join(vdir, f"{name}.mp4")

        from env.video import write_mp4
        path = write_mp4(self.current_frames, video_path, fps=fps)
        if path:
            print(f"[Executor] Saved video ({len(self.current_frames)} frames) to {path}")
        return path

    def close(self):
        self.env.close()


def create_executor(use_mock_policy: bool = True) -> Executor:
    return Executor(use_mock_policy=use_mock_policy)


if __name__ == "__main__":
    print("Testing Executor...")
    
    executor = create_executor(use_mock_policy=True)
    executor.env.reset()
    
    test_subtask = {
        "id": 1,
        "action": "pick_and_place",
        "object": "red block",
        "target": "bin"
    }
    
    print(f"\nRunning test subtask: {test_subtask}")
    result = executor.run_subtask(test_subtask, max_steps=50, save_video=True, video_name="test_subtask")
    
    print("\nSubtask result:")
    print(f"  Duration: {result['duration']:.2f}s")
    print(f"  Steps: {result['steps']}")
    print(f"  Object moved: {result['object_moved']}")
    print(f"  Initial position: {result['initial_pos']}")
    print(f"  Final position: {result['final_pos']}")
    print(f"  Video saved: {result['video_path']}")
    
    executor.close()
    print("\nExecutor test complete!")
