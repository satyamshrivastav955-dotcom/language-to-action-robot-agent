import numpy as np
import os
import time
from typing import Dict, Optional, Tuple, List
import cv2

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("[Warning] MuJoCo not available, using mock environment")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import (
    HOME_POSITION_DEG,
    OBJECT_GROUNDING,
    TARGET_REGIONS,
    BLOCK_START_POSITIONS,
    BLOCK_SIZE,
)


class MockSO101Environment:

    def __init__(self, xml_path: Optional[str] = None, render_mode: str = "rgb_array"):
        self.render_mode = render_mode
        self.home_position_rad = np.array(HOME_POSITION_DEG) * np.pi / 180.0

        self.joint_indices = list(range(6))

        self.object_names = ["red_block", "blue_block", "green_block", "yellow_block"]
        self.object_ids = {name: i for i, name in enumerate(self.object_names)}

        self._object_positions = {}
        self._reset_object_positions()

        self._frame_count = 0
        self._step_count = 0
        self._joint_positions = self.home_position_rad.copy()
        self.is_physics_backed = False

    def _reset_object_positions(self):
        # Sourced from settings so the env, verifier and scene.xml agree.
        self._object_positions = {
            name: np.array(pos, dtype=float)
            for name, pos in BLOCK_START_POSITIONS.items()
        }
        self._object_positions["bin"] = np.array(TARGET_REGIONS["bin"], dtype=float)

    def reset(self) -> np.ndarray:
        self._reset_object_positions()
        self._joint_positions = self.home_position_rad.copy()
        self._frame_count = 0
        self._step_count = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        return self._joint_positions.copy()

    def get_object_pose(self, obj_name: str) -> np.ndarray:
        if obj_name not in self.object_ids:
            obj_name = OBJECT_GROUNDING.get(obj_name, obj_name)

        if obj_name in self._object_positions:
            return self._object_positions[obj_name].copy()

        return np.array([0.0, 0.0, 0.04])

    def get_target_region(self, target_name: str) -> np.ndarray:
        if target_name in TARGET_REGIONS:
            return np.array(TARGET_REGIONS[target_name])
        elif target_name in self._object_positions:
            return self._object_positions[target_name]
        return np.array([0.45, 0.0, 0.04])

    def set_joint_positions(self, positions: np.ndarray, degrees: bool = False):
        if degrees:
            positions = np.array(positions) * np.pi / 180.0
        self._joint_positions = positions.copy()

    def step(self, action: np.ndarray, n_substeps: int = 10) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.clip(np.asarray(action, dtype=float), -np.pi, np.pi)
        self.set_joint_positions(action)

        # Tiny sensor-style jitter on the blocks only. The bin is furniture and
        # must not drift: it is the verification target, and a random walk over
        # ~50 steps was displacing it a meaningful fraction of the 0.05m
        # tolerance, making success partly a coin flip.
        for name in self._object_positions:
            if name == "bin":
                continue
            self._object_positions[name] = (
                self._object_positions[name] + np.random.randn(3) * 0.0005
            )

        self._step_count = getattr(self, '_step_count', 0) + 1

        observation = self._get_observation()
        reward = 0.0
        done = False
        info = {"joint_positions": observation, "step": self._step_count}

        return observation, reward, done, info

    def simulate_pick_and_place(self, obj: str, target: str, should_succeed: bool):
        if obj not in self._object_positions:
            return

        target_pos = self._get_simulated_target_position(target)

        if should_succeed:
            self._object_positions[obj] = target_pos + np.random.randn(3) * 0.01
        else:
            current_pos = self._object_positions[obj].copy()
            away_direction = current_pos - target_pos
            away_direction = away_direction / (np.linalg.norm(away_direction) + 1e-6)
            random_offset = np.array([
                np.random.uniform(0.05, 0.15) * np.sign(away_direction[0]) if away_direction[0] != 0 else np.random.uniform(-0.15, 0.15),
                np.random.uniform(0.05, 0.15) * np.sign(away_direction[1]) if away_direction[1] != 0 else np.random.uniform(-0.15, 0.15),
                np.random.uniform(0, 0.05)
            ])
            self._object_positions[obj] = current_pos + random_offset

    def _get_simulated_target_position(self, target: str) -> np.ndarray:
        if target in TARGET_REGIONS:
            return np.array(TARGET_REGIONS[target], dtype=float)
        elif target in self._object_positions:
            base_pos = self._object_positions[target].copy()
            base_pos[2] += BLOCK_SIZE
            return base_pos
        return np.array(TARGET_REGIONS["table_center"], dtype=float)

    def render(self, width: int = 640, height: int = 480, **kwargs) -> np.ndarray:
        # **kwargs tolerates stray callers (e.g. render(timeout=...)) rather
        # than raising TypeError mid-rollout.
        self._frame_count += 1

        frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame[:] = (30, 30, 35)

        cv2.rectangle(frame, (0, height - 50), (width, height), (60, 60, 65), -1)

        colors = {
            "red_block": (50, 50, 220),
            "blue_block": (220, 100, 50),
            "green_block": (50, 200, 50),
            "yellow_block": (50, 200, 240),
        }

        for name, pos in self._object_positions.items():
            if name == "bin":
                continue
            x = int((pos[0] - 0.1) / 0.5 * width)
            y = int((pos[1] + 0.3) / 0.6 * height)
            x = max(20, min(width - 20, x))
            y = max(20, min(height - 70, y))

            color = colors.get(name, (200, 200, 200))
            cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), color, -1)
            cv2.putText(frame, name.split("_")[0], (x - 20, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        bin_pos = TARGET_REGIONS["bin"]
        bin_x = int((bin_pos[0] - 0.1) / 0.5 * width)
        bin_y = int((bin_pos[1] + 0.3) / 0.6 * height)
        cv2.rectangle(frame, (bin_x - 30, bin_y - 30), (bin_x + 30, bin_y + 30), (100, 100, 150), 2)
        cv2.putText(frame, "BIN", (bin_x - 15, bin_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 150), 1)

        cv2.putText(frame, f"Frame: {self._frame_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame

    def get_end_effector_pose(self) -> np.ndarray:
        return np.array([0.35, 0.0, 0.1])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class SO101Environment(MockSO101Environment):
    """Real MuJoCo-backed SO-101 workspace.

    env/scene.xml now defines what a rollout actually needs:
      * each block is a <body> with a <freejoint>, so it can be grasped, lifted,
        carried and dropped by contact rather than teleported;
      * <position> actuators drive joint1-6 plus the two gripper fingers.

    Everything the verifier reads (get_object_pose) comes from mjData.xpos, so
    success is measured off the physics state, not off a scripted outcome.

    Falls back to the kinematic mock only if MuJoCo is missing or the model
    fails to load, and says so loudly rather than pretending.
    """

    # Shoulder (joint2) pivot height from scene.xml: base 0.02 + 0.06 + 0.04.
    SHOULDER_Z = 0.12
    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 0.0135

    def __init__(self, xml_path: Optional[str] = None, render_mode: str = "rgb_array"):
        super().__init__(xml_path=xml_path, render_mode=render_mode)
        self.xml_path = xml_path or os.path.join(os.path.dirname(__file__), "scene.xml")
        self.is_physics_backed = False

        self.model = None
        self.data = None
        self._renderer = None
        self._renderer_size = None
        self.load_error: Optional[str] = None
        self._grasped: Optional[str] = None
        self._grasp_offset = np.zeros(3)

        if not MUJOCO_AVAILABLE:
            self.load_error = "mujoco package not installed"
            print("[SO101Env] MuJoCo unavailable - falling back to kinematic mock")
            return

        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)

            self._body_ids = {
                name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in self.object_names
            }
            self._bin_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "bin_body")
            self._ee_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
            self._grip_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")

            # qpos address of each block's freejoint, for reset.
            self._block_qpos_adr = {}
            for name in self.object_names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{name}_free")
                self._block_qpos_adr[name] = self.model.jnt_qposadr[jid]

            self._arm_qpos_adr = []
            for i in range(1, 7):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                self._arm_qpos_adr.append(self.model.jnt_qposadr[jid])
            self._arm_dof_adr = []
            for i in range(1, 7):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                self._arm_dof_adr.append(self.model.jnt_dofadr[jid])

            self.is_physics_backed = True
            print(f"[SO101Env] MuJoCo physics active "
                  f"({self.model.nq} qpos, {self.model.nu} actuators)")
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            self.model = None
            self.data = None
            print(f"[SO101Env] Failed to load {self.xml_path} - {self.load_error}")
            print("[SO101Env] Falling back to kinematic mock")

    # ------------------------------------------------------------------
    # Core gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        if not self.is_physics_backed:
            return super().reset()

        mujoco.mj_resetData(self.model, self.data)
        self._grasped = None

        # Blocks back to the canonical settings positions, upright, at rest.
        for name, pos in BLOCK_START_POSITIONS.items():
            adr = self._block_qpos_adr[name]
            self.data.qpos[adr:adr + 3] = pos
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

        home = self._home_arm_pose()
        for adr, val in zip(self._arm_qpos_adr, home):
            self.data.qpos[adr] = val
        self.data.ctrl[:6] = home
        self.data.ctrl[6:8] = self.GRIPPER_OPEN

        mujoco.mj_forward(self.model, self.data)

        self._frame_count = 0
        self._step_count = 0
        return self._get_observation()

    def _home_arm_pose(self) -> np.ndarray:
        """Home pose in radians, from HOME_POSITION_DEG in settings.

        Reuses the documented degrees->radians convention; the sign convention
        on joints 2-4 differs from the datasheet's, so we solve IK from here
        rather than trusting it as an absolute pose.
        """
        return np.array(HOME_POSITION_DEG, dtype=float) * np.pi / 180.0

    def _get_observation(self) -> np.ndarray:
        if not self.is_physics_backed:
            return super()._get_observation()
        return np.array([self.data.qpos[a] for a in self._arm_qpos_adr])

    def step(self, action: np.ndarray, n_substeps: int = 10) -> Tuple[np.ndarray, float, bool, Dict]:
        """Drive the arm toward `action` (6 joint targets, radians).

        A 7th element, if present, is the gripper command in [0,1] where
        1 = fully closed.
        """
        if not self.is_physics_backed:
            return super().step(action, n_substeps=n_substeps)

        action = np.asarray(action, dtype=float).ravel()

        self.data.ctrl[:6] = np.clip(action[:6], -np.pi, np.pi)
        grip_cmd = 0.0
        if action.size >= 7:
            grip_cmd = float(np.clip(action[6], 0.0, 1.0))
            self.data.ctrl[6:8] = grip_cmd * self.GRIPPER_CLOSED

        self._update_grasp(grip_cmd)

        for _ in range(max(1, n_substeps)):
            if self._grasped is not None:
                self._carry_grasped()
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        observation = self._get_observation()
        info = {
            "joint_positions": observation,
            "step": self._step_count,
            "ee_pos": self.get_end_effector_pose().tolist(),
            "grasped": self._grasped,
        }
        return observation, 0.0, False, info

    # ------------------------------------------------------------------
    # Grasp assist
    # ------------------------------------------------------------------
    # Rigid-body friction grasping of a 40mm cube with parallel pads is very
    # sensitive to pad geometry and solver settings; tuning it burns hours and
    # is not what this project is demonstrating. Instead, when the gripper
    # closes with a block between the pads we kinematically attach that block
    # to the grip site, and on release we hand it straight back to the
    # simulator, which drops it under real gravity and real contact.
    #
    # So: arm motion, approach, collisions, the drop and the final resting
    # pose are all genuine physics; only the carry is assisted. Success is
    # still read from mjData, and a mis-aimed approach still misses the block
    # and still fails. See Executor.describe_policy, which reports this.

    GRASP_XY_TOL = 0.035
    GRASP_Z_TOL = 0.05
    GRASP_TRIGGER = 0.5

    def _update_grasp(self, grip_cmd: float):
        if grip_cmd < self.GRASP_TRIGGER:
            if self._grasped is not None:
                # Release: zero the velocity so it drops, not flings.
                adr = self._block_qpos_adr[self._grasped]
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{self._grasped}_free")
                dof = self.model.jnt_dofadr[jid]
                self.data.qvel[dof:dof + 6] = 0.0
                self._grasped = None
            return

        if self._grasped is not None:
            return

        site = self.data.site_xpos[self._grip_site_id]
        best, best_d = None, 1e9
        for name, bid in self._body_ids.items():
            p = self.data.xpos[bid]
            dxy = float(np.linalg.norm(p[:2] - site[:2]))
            dz = abs(float(p[2] - site[2]))
            if dxy < self.GRASP_XY_TOL and dz < self.GRASP_Z_TOL and dxy < best_d:
                best, best_d = name, dxy

        if best is not None:
            self._grasped = best
            self._grasp_offset = np.array(self.data.xpos[self._body_ids[best]]) - np.array(site)
            # Keep it centred in the pads rather than at whatever offset it
            # happened to be nudged to.
            self._grasp_offset[:2] = 0.0

    def _carry_grasped(self):
        adr = self._block_qpos_adr[self._grasped]
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{self._grasped}_free")
        dof = self.model.jnt_dofadr[jid]
        self.data.qpos[adr:adr + 3] = self.data.site_xpos[self._grip_site_id] + self._grasp_offset
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        self.data.qvel[dof:dof + 6] = 0.0

    def set_gripper(self, closed_fraction: float):
        if not self.is_physics_backed:
            return
        grip = float(np.clip(closed_fraction, 0.0, 1.0)) * self.GRIPPER_CLOSED
        self.data.ctrl[6:8] = grip

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_object_pose(self, obj_name: str) -> np.ndarray:
        """Current xyz of a named body, read from MuJoCo physics state."""
        if not self.is_physics_backed:
            return super().get_object_pose(obj_name)

        if obj_name not in self._body_ids and obj_name != "bin":
            obj_name = OBJECT_GROUNDING.get(obj_name, obj_name)

        if obj_name == "bin":
            # Report the bin's usable interior (where a block comes to rest),
            # matching TARGET_REGIONS["bin"], not the body origin.
            return np.array(TARGET_REGIONS["bin"], dtype=float)

        if obj_name in self._body_ids:
            return np.array(self.data.xpos[self._body_ids[obj_name]], dtype=float)

        return np.array([0.0, 0.0, 0.04])

    def get_end_effector_pose(self) -> np.ndarray:
        if not self.is_physics_backed:
            return super().get_end_effector_pose()
        return np.array(self.data.site_xpos[self._grip_site_id], dtype=float)

    # ------------------------------------------------------------------
    # Inverse kinematics (damped least squares on the grip site)
    # ------------------------------------------------------------------

    def solve_ik(self, target_xyz, iterations: int = 260,
                 tol: float = 2e-3) -> np.ndarray:
        """Joint angles putting grip_site at target_xyz, gripper pointing down.

        Damped least squares on the site Jacobian, solving position AND
        orientation together. Orientation matters: with position alone the
        wrist settles at an arbitrary tilt, the finger pads no longer straddle
        the block along their slide axis, and every grasp shoves the block away
        instead of closing on it.

        Runs on a scratch MjData so it never disturbs live simulation state.
        """
        if not self.is_physics_backed:
            return self._get_observation()

        target = np.asarray(target_xyz, dtype=float)
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, scratch)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        damping = 0.08

        # Gripper straight down, finger slide axis along world y.
        R_goal = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]])

        for _ in range(iterations):
            mujoco.mj_forward(self.model, scratch)

            pos_err = target - scratch.site_xpos[self._grip_site_id]

            R_cur = scratch.site_xmat[self._grip_site_id].reshape(3, 3)
            R_err = R_goal @ R_cur.T
            axis = np.array([R_err[2, 1] - R_err[1, 2],
                             R_err[0, 2] - R_err[2, 0],
                             R_err[1, 0] - R_err[0, 1]]) * 0.5
            sin_a = np.linalg.norm(axis)
            cos_a = (np.trace(R_err) - 1.0) * 0.5
            angle = np.arctan2(sin_a, cos_a)
            rot_err = axis / sin_a * angle if sin_a > 1e-9 else np.zeros(3)

            if np.linalg.norm(pos_err) < tol and np.linalg.norm(rot_err) < 0.05:
                break

            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self._grip_site_id)
            J = np.vstack([jacp[:, self._arm_dof_adr], jacr[:, self._arm_dof_adr]])

            # Position is what must land on the block; orientation is a softer
            # preference so a slightly-off wrist never blocks convergence.
            err6 = np.concatenate([pos_err, rot_err * 0.45])

            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err6)

            step = np.clip(dq, -0.15, 0.15)
            for k, adr in enumerate(self._arm_qpos_adr):
                scratch.qpos[adr] = np.clip(scratch.qpos[adr] + step[k], -np.pi, np.pi)

        return np.array([scratch.qpos[a] for a in self._arm_qpos_adr])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, width: int = 640, height: int = 480,
               camera: str = "camera1", **kwargs) -> np.ndarray:
        if not self.is_physics_backed:
            return super().render(width=width, height=height, **kwargs)

        try:
            if self._renderer is None or self._renderer_size != (height, width):
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = mujoco.Renderer(self.model, height, width)
                self._renderer_size = (height, width)

            self._renderer.update_scene(self.data, camera=camera)
            frame = self._renderer.render()
            self._frame_count += 1
            return frame
        except Exception as e:
            # Headless boxes without a GL backend still get a usable run.
            if self.load_error is None:
                print(f"[SO101Env] Render unavailable ({type(e).__name__}: {e}) "
                      f"- falling back to 2D frames")
            self.load_error = f"render: {e}"
            self.is_physics_backed_render = False
            return super().render(width=width, height=height, **kwargs)

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None


def create_environment(xml_path: Optional[str] = None) -> MockSO101Environment:
    """Build the workspace environment.

    Returns a MuJoCo-backed SO101Environment when the package and scene load;
    that object degrades to the kinematic mock internally otherwise, so callers
    always get a working env and can check `.is_physics_backed`.
    """
    return SO101Environment(xml_path)


if __name__ == "__main__":
    print("Testing Environment...")

    env = create_environment()
    print(f"Environment type: {'MuJoCo' if env.is_physics_backed else 'Mock'}")
    print(f"Objects: {env.object_names}")

    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    print("\nObject positions:")
    for obj_name in env.object_names:
        pos = env.get_object_pose(obj_name)
        print(f"  {obj_name}: {pos}")

    frame = env.render()
    if frame is not None:
        print(f"\nFrame shape: {frame.shape}")
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/test_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Saved test frame to outputs/test_frame.png")

    new_obs, reward, done, info = env.step(obs + 0.01)
    print(f"\nAfter step: observation shape: {new_obs.shape}")

    env.close()
    print("\nEnvironment test complete!")
