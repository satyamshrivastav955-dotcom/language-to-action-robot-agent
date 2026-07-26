import numpy as np
import torch
import os
import time
from typing import Dict, Optional, List, Tuple
import cv2
import imageio

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                      retry_params: Optional[Dict] = None):
        """Build the waypoint plan for this attempt from live object poses."""
        retry_params = retry_params or {}
        offset = np.array(retry_params.get("position_offset", [0.0, 0.0, 0.0]), dtype=float)

        obj_pos = np.asarray(self.env.get_object_pose(obj), dtype=float)

        if target == "bin":
            place = np.asarray(TARGET_REGIONS["bin"], dtype=float).copy()
            release_z = self.RELEASE_Z
        else:
            tgt = np.asarray(self.env.get_object_pose(target), dtype=float)
            place = tgt.copy()
            place[2] = tgt[2] + BLOCK_SIZE  # stack on top
            # The carried block hangs ~6mm below the grip site, so the site has
            # to sit that much higher than the intended resting height, plus a
            # few mm clearance so the block is set down rather than driven into
            # the tower (which flicks it off).
            release_z = tgt[2] + BLOCK_SIZE + 0.006 + 0.008
        self._stack_target = None if target == "bin" else target

        # Retry nudges the grasp, not the goal: the target is ground truth.
        grasp_xy = obj_pos[:2] + offset[:2]

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
            {"name": "release", "xyz": [place[0], place[1], release_z],
             "grip": 0.0, "steps": 40},
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
        if self._phase_step == 0:
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
            # Re-read the stack target too: it may have been nudged since the
            # plan was built, and placing on its stale pose drops the block
            # onto empty table.
            goal = np.asarray(phase["xyz"], dtype=float)
            # Deliberately NOT re-reading the stack target here. Tracking it
            # during descent becomes a shove: the carried block nudges the
            # target, the goal chases it, and both end up across the table.
            # The frozen plan pose plus closed-loop correction on our OWN
            # position is what actually lands the block.
            actual = self.env.get_end_effector_pose()
            err = goal - actual
            if np.linalg.norm(err) > 0.005:
                self._cached_q = self.env.solve_ik(goal + err * 0.9)

        self._phase_step += 1

        if self._phase_step >= phase["steps"] and self._idx < len(self._plan) - 1:
            self._idx += 1
            self._phase_step = 0

        return np.append(self._cached_q[:6], phase["grip"])

    def is_done(self) -> bool:
        last = self._idx >= len(self._plan) - 1
        return last and self._phase_step >= self._plan[-1]["steps"] if self._plan else True

    def current_phase(self) -> str:
        if not self._plan:
            return "idle"
        return self._plan[min(self._idx, len(self._plan) - 1)]["name"]


class SmolVLAPolicy:
    
    def __init__(self, model_path: str = "lerobot/smolvla_base", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            print(f"[SmolVLA] Loading model from {self.model_path}...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("[SmolVLA] Model loaded successfully")
        except Exception as e:
            print(f"[SmolVLA] Failed to load real model: {e}")
            print("[SmolVLA] Falling back to mock policy")
            self.model = None
            self.processor = None
    
    @classmethod
    def from_pretrained(cls, model_path: str = "lerobot/smolvla_base", device: str = "cuda"):
        return cls(model_path=model_path, device=device)
    
    def get_action(self, observation: np.ndarray, image: np.ndarray, language_prompt: str) -> np.ndarray:
        if self.model is None:
            return self._mock_action(observation)
        
        try:
            inputs = self.processor(
                images=image,
                text=language_prompt,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50)
            
            action = self.processor.decode(outputs[0], skip_special_tokens=True)
            return self._parse_action(action)
        except Exception as e:
            print(f"[SmolVLA] Inference error: {e}")
            return self._mock_action(observation)
    
    def _mock_action(self, observation: np.ndarray) -> np.ndarray:
        base_angles = np.array(HOME_POSITION_DEG) * np.pi / 180.0
        perturbation = np.random.randn(6) * 0.02
        return base_angles + perturbation
    
    def _parse_action(self, action_str: str) -> np.ndarray:
        try:
            numbers = [float(x) for x in action_str.replace("[", "").replace("]", "").split(",")]
            return np.array(numbers[:6])
        except (ValueError, AttributeError):
            return np.array(HOME_POSITION_DEG) * np.pi / 180.0
    
    def reset(self):
        pass


class Executor:
    
    def __init__(self, policy: Optional[object] = None, env: Optional[SO101Environment] = None,
                 use_mock_policy: bool = True):
        self.env = env if env else SO101Environment()

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
            # controller so outcomes come from contact, not a coin flip.
            self.policy = ScriptedMuJoCoPolicy(self.env)
            self.policy_type = "scripted_sim"
            print("[Executor] Using scripted MuJoCo waypoint controller "
                  "(physics-determined outcomes)")
        else:
            try:
                self.policy = SmolVLAPolicy.from_pretrained()
                self.policy_type = "real"
            except Exception as e:
                print(f"[Executor] Could not load SmolVLA ({type(e).__name__}: {e}); using mock policy")
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
            "name": getattr(self.policy, "model_path", "smolvla"),
            # A SmolVLAPolicy whose load failed silently falls back to noise
            # actions, which is scripted-adjacent - say so rather than claiming
            # a real policy is driving.
            "outcomes_are_scripted": not loaded,
            "weights_loaded": loaded,
            "note": ("Real policy driving the rollout." if loaded else
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

        self.env.reset()

        if self.policy_type == "mock" and hasattr(self.policy, 'begin_attempt'):
            self.policy.begin_attempt(retry_params)
        elif self.policy_type == "scripted_sim":
            self.policy.begin_attempt(action, obj, target, retry_params)

        initial_object_pos = self.env.get_object_pose(obj)

        obs = self.env._get_observation()

        # A slower, more careful retry is modelled as a longer rollout budget.
        speed_factor = float((retry_params or {}).get("speed_factor", 1.0) or 1.0)
        effective_steps = max(1, int(max_steps / max(speed_factor, 0.1)))

        if self.policy_type == "scripted_sim":
            # The waypoint plan needs its full length to finish the place; a
            # short max_steps would cut the arm off mid-carry.
            effective_steps = max(effective_steps, 300)

        for step in range(effective_steps):
            if self.policy_type == "real":
                frame = self.env.render()
                action_cmd = self.policy.get_action(obs, frame, prompt)
            else:
                action_cmd = self.policy.get_action(obs, prompt)

            obs, reward, done, info = self.env.step(action_cmd)

            # Render every other step: physics needs fine substeps, but a
            # frame per step makes the video long and the run slow.
            if self.policy_type != "scripted_sim" or step % 2 == 0:
                frame = self.env.render()
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
                if i % 4 == 0:
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

        for _ in range(5):
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
            "final_pos": final_object_pos,
            "object_moved": not np.allclose(initial_object_pos, final_object_pos, atol=0.01),
            "duration": elapsed_time,
            "steps": len(self.current_frames),
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
    
    def capture_frame(self):
        frame = self.env.render()
        if frame is not None:
            self.current_frames.append(frame)
        return frame
    
    def _save_video(self, name: str, fps: int = 30) -> Optional[str]:
        from configs.settings import VIDEOS_DIR

        if not self.current_frames:
            return None

        os.makedirs(VIDEOS_DIR, exist_ok=True)
        video_path = os.path.join(VIDEOS_DIR, f"{name}.mp4")

        # OpenCV first: it is a hard dependency here and always present, whereas
        # imageio needs the separate imageio-ffmpeg package. The old order tried
        # imageio first and printed a scary "expected bytes, NoneType found"
        # error on every single subtask before silently succeeding via OpenCV.
        try:
            h, w = self.current_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            for frame in self.current_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[Executor] Saved video ({len(self.current_frames)} frames) to {video_path}")
            return video_path
        except Exception as e:
            print(f"[Executor] OpenCV video write failed ({type(e).__name__}: {e}); trying imageio")

        try:
            imageio.mimsave(video_path, self.current_frames, fps=fps)
            print(f"[Executor] Saved video to {video_path} (imageio)")
            return video_path
        except Exception as e:
            print(f"[Executor] Could not save video: {type(e).__name__}: {e}")
            print("[Executor] Hint: pip install imageio-ffmpeg")
            return None
    
    def close(self):
        self.env.close()


def create_executor(use_mock_policy: bool = True) -> Executor:
    return Executor(use_mock_policy=use_mock_policy)


if __name__ == "__main__":
    print("Testing Executor...")
    
    executor = create_executor(use_mock_policy=True)
    
    test_subtask = {
        "id": 1,
        "action": "pick_and_place",
        "object": "red block",
        "target": "bin"
    }
    
    print(f"\nRunning test subtask: {test_subtask}")
    result = executor.run_subtask(test_subtask, max_steps=50, save_video=True, video_name="test_subtask")
    
    print(f"\nSubtask result:")
    print(f"  Duration: {result['duration']:.2f}s")
    print(f"  Steps: {result['steps']}")
    print(f"  Object moved: {result['object_moved']}")
    print(f"  Initial position: {result['initial_pos']}")
    print(f"  Final position: {result['final_pos']}")
    print(f"  Video saved: {result['video_path']}")
    
    executor.close()
    print("\nExecutor test complete!")
