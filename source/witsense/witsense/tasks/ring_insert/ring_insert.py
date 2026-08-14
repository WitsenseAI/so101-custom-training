"""Single-arm SO-101 task: pick up the ring and place it around the ghost toy.

The sim counterpart of the real-robot `pick_and_place_ring` recordings. Same bedroom
scene as tasks/bedroom, but one arm instead of two and two rigid objects instead of the
particle garment — no garment loader, no particle config, no cloth solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdShade

import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from witsense.assets.scenes.bedroom import MARBLE_BEDROOM_USD_PATH
from witsense.utils.constant import ASSETS_ROOT
from witsense.devices.action_process import preprocess_device_action
from witsense.tasks.ring_insert.ring_insert_cfg import (
    GHOST_HEIGHT,
    RING_HALF_HEIGHT,
    RING_INNER_RADIUS,
    RingInsertEnvCfg,
)
from witsense.utils.logger import get_logger

logger = get_logger(__name__)


class RingInsertEnv(DirectRLEnv):
    cfg: RingInsertEnvCfg

    def __init__(self, cfg: RingInsertEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.action_scale = self.cfg.action_scale

        # A fresh RandomState per env, seeded only when the config asks for it, so that
        # `use_random_seed=False` gives byte-identical ring placements across runs.
        self.rng = np.random.RandomState(None if cfg.use_random_seed else cfg.random_seed)

        # GUI viewport for teleoperation: behind the arm, looking along its reach.
        # Centred on x=0.23 (the arm) rather than the bimanual task's x=0 midpoint.
        cfg.viewer = cfg.viewer.replace(eye=(0.23, -1.1, 1.15), lookat=(0.23, -0.02, 0.52))
        super().__init__(cfg, render_mode, **kwargs)
        self._apply_joint_limits()

    def _apply_joint_limits(self) -> None:
        """Widen the USD's joint limits to the real arm's measured range.

        Must run after super().__init__(): the articulation does not exist before that,
        and the reset pose is clamped to whatever limits are in force when it is written.
        """
        names = list(self.robot.joint_names)

        spec = self.cfg.joint_limits_deg
        if spec:
            missing = [n for n in spec if n not in names]
            if missing:
                logger.warning(f"joint_limits_deg names not on this robot, ignored: {missing}")
            ids = [names.index(n) for n in spec if n in names]
            deg = torch.tensor([spec[names[i]] for i in ids], dtype=torch.float32, device=self.device)
            limits = torch.deg2rad(deg).unsqueeze(0).expand(self.num_envs, -1, -1)
            self.robot.write_joint_position_limit_to_sim(limits, joint_ids=ids)

        # Written only now, once the limits are wide enough to hold it. init_state's
        # joint_pos had to stay inside the USD's range to survive startup validation, so
        # without this elbow_flex would sit at the USD's 90 rather than the real 93.6.
        home = self.cfg.home_pose_deg
        if home:
            for name, value in home.items():
                if name in names:
                    self.robot.data.default_joint_pos[:, names.index(name)] = np.deg2rad(value)

        actual = torch.rad2deg(self.robot.data.default_joint_pos[0]).tolist()
        logger.info(
            f"[RingInsertEnv] home pose (deg) {[round(v, 1) for v in actual]} "
            f"for joints {names}"
        )

    def _setup_scene(self):
        # `self.robot` is the attribute name devices/action_process.py reads for
        # single-arm keyboard teleop. Renaming it silently breaks teleop.
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.ring = RigidObject(self.cfg.ring)
        self.ghost = RigidObject(self.cfg.ghost)

        scene_cfg = sim_utils.UsdFileCfg(usd_path=f"{MARBLE_BEDROOM_USD_PATH}")
        scene_cfg.func(
            "/World/Scene",
            scene_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Recolour here, while the scene prims have just been authored — not only on
        # reset, so the right colours are in place before the first frame is rendered.
        self._apply_table_texture()
        self._apply_robot_color()

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ring"] = self.ring
        self.scene.rigid_objects["ghost"] = self.ghost
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── stepping ─────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, :6].squeeze(0)

        # EVERY array here must be copied. On a CPU device tensor.cpu() is a no-op and
        # .numpy() returns a VIEW onto the live simulation buffer, which Isaac Lab
        # overwrites in place each step. The recorder keeps these arrays until the episode
        # is saved, so without a copy all N frames alias one buffer and the whole episode
        # is written holding the final step's values — a dataset whose observation.state
        # never moves while the action spans 90 degrees, with no error anywhere.
        # (On CUDA .cpu() copies, which is why this hides when the sim runs on GPU.)
        observations = {
            "action": self.actions.squeeze(0).cpu().detach().numpy().copy(),
            "observation.state": joint_pos.cpu().detach().numpy().copy(),
            "observation.images.top_rgb": self.top_camera.data.output["rgb"]
            .cpu()
            .detach()
            .numpy()
            .squeeze()
            .copy(),
            "observation.images.wrist_rgb": self.wrist_camera.data.output["rgb"]
            .cpu()
            .detach()
            .numpy()
            .squeeze()
            .copy(),
        }

        # Depth is optional: absent when the top camera renders RGB only
        # (cfg.top_camera.data_types == ["rgb"], e.g. --disable_depth).
        if "depth" in self.top_camera.data.output:
            depth_np = self.top_camera.data.output["depth"].squeeze().cpu().detach().numpy().copy()
            # metres -> uint16 millimetres, matching the recorded dataset feature spec
            observations["observation.top_depth"] = np.clip(depth_np * 1000, 0, 65535).astype(
                np.uint16
            )

        return observations

    # ── task ─────────────────────────────────────────────────────────────────────

    def _ring_ghost_offset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Horizontal distance between ring and ghost centres, and the ring's height."""
        ring_pos = self.ring.data.root_pos_w
        ghost_pos = self.ghost.data.root_pos_w
        xy_dist = torch.norm(ring_pos[:, :2] - ghost_pos[:, :2], dim=-1)
        return xy_dist, ring_pos[:, 2]

    def _get_rewards(self) -> torch.Tensor:
        # Shaped only enough to be readable in logs. ACT is trained by behaviour cloning
        # from teleoperation, so nothing consumes this — success is what to look at.
        xy_dist, _ = self._ring_ghost_offset()
        return (1.0 - torch.tanh(5.0 * xy_dist)) + self._get_success().float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> torch.Tensor:
        """Ring encircles the ghost: centred in xy and dropped down over it, not on top."""
        xy_dist, ring_z = self._ring_ghost_offset()
        return (xy_dist < self.cfg.success_xy_tol) & (ring_z < self.cfg.success_z_max)

    def insertion_progress(self) -> torch.Tensor:
        """How far along the insertion is, in [0, 1]. Continuous where success is binary.

        A near miss and a rollout that ignored the ring both score 0 on success, which
        makes them indistinguishable when ranking policies or filtering rollouts to
        retrain on. Two halves, so the whole task is graded rather than just its end:

            approach (0.5) - how far the ring has closed on the ghost horizontally, over
                             the ~0.15 m they start apart
            seating  (0.5) - align x depth, where align is 1 on the ghost's axis and 0
                             once its hole could not contain it, and depth runs from 0
                             with the ring perched on the ghost's head to 1 with it down
                             at its resting height

        Seating alone (the obvious formulation) stays at 0 until the last ~2 cm of
        descent, so a rollout that lifted the ring and carried it to the target scores
        the same as one that never touched it — precisely the distinction this is for.

        Caveat: approach counts lateral closeness however it was achieved, so a ring
        knocked toward the ghost scores ~0.5 without ever being grasped. Use it to rank
        attempts, not as ground truth for success — _get_success remains that.
        """
        xy_dist, ring_z = self._ring_ghost_offset()
        ghost_z = self.ghost.data.root_pos_w[:, 2]

        approach = (1.0 - xy_dist / self.cfg.progress_approach_range).clamp(0.0, 1.0)

        align = (1.0 - xy_dist / RING_INNER_RADIUS).clamp(0.0, 1.0)
        top = ghost_z + GHOST_HEIGHT              # the ghost's head
        seated = ghost_z + RING_HALF_HEIGHT       # ring centre once it is down on the table
        depth = ((top - ring_z) / (top - seated).clamp(min=1e-6)).clamp(0.0, 1.0)

        return 0.5 * approach + 0.5 * align * depth

    # ── reset ────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        jitter = self.cfg.arm_start_jitter_deg
        if jitter:
            noise = self.rng.uniform(-jitter, jitter, size=tuple(joint_pos.shape))
            joint_pos += torch.as_tensor(np.deg2rad(noise), dtype=joint_pos.dtype, device=self.device)
            # Stay inside the limits, or PhysX clamps silently and the pose drifts from
            # what the trace reports.
            limits = self.robot.data.joint_pos_limits[env_ids]
            joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        # Zero the velocities too: writing only positions lets the previous episode's
        # joint velocities survive and bias the first observation of the next one.
        self.robot.write_joint_velocity_to_sim(
            torch.zeros_like(joint_pos), joint_ids=None, env_ids=env_ids
        )

        self._write_object_pose(self.ring, self.cfg.ring_pos, env_ids, jitter=self.cfg.ring_xy_jitter)
        # The ghost has no velocity to clear when it is kinematic — PhysX rejects
        # setLinearVelocity/setAngularVelocity on kinematic bodies and logs an error.
        self._write_object_pose(
            self.ghost, self.cfg.ghost_pos, env_ids, jitter=0.0,
            zero_velocity=not self.cfg.ghost_kinematic,
        )
        self._apply_table_texture()

    def _apply_table_texture(self) -> None:
        """Retexture the table top to match the real mat the policy was trained on."""
        tex_name = self.cfg.table_texture
        if not tex_name:
            return

        # Absolute, so this does not depend on the working directory the way the
        # bimanual task's randomiser does.
        tex_path = Path(ASSETS_ROOT) / "textures" / "surface" / tex_name
        if not tex_path.is_file():
            logger.warning(f"table texture not found: {tex_path}")
            return

        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(self.cfg.table_shader_path)
        if not prim.IsValid():
            logger.warning(f"table shader prim not found: {self.cfg.table_shader_path}")
            return

        shader = UsdShade.Shader(prim)
        tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not tex_input:
            logger.warning("table shader has no 'file' or 'diffuse_texture' input")
            return

        # Author into the root layer. The default edit target during simulation is the
        # session layer, which Isaac Lab drops on stop->play — the table goes dark for a
        # moment during load and then snaps back to the scene's own T_Table038_BC001.png.
        with Usd.EditContext(stage, stage.GetRootLayer()):
            tex_input.Set(Sdf.AssetPath(str(tex_path)))

    def _apply_robot_color(self) -> None:
        """Paint the arm's materials, since the asset is yellow and the real arm is white.

        Walks the materials under the robot prim rather than naming them: the asset has
        several and their paths are not worth hardcoding. Both shader conventions are
        handled — UsdPreviewSurface uses `diffuseColor`, MDL/OmniPBR uses
        `diffuse_color_constant`. Shaders driven by a texture are skipped, so the black
        motor bodies keep their own look.
        """
        color = self.cfg.robot_color
        if color is None:
            return

        stage = stage_utils.get_current_stage()
        root = stage.GetPrimAtPath(self.cfg.robot.prim_path)
        if not root.IsValid():
            logger.warning(f"robot prim not found: {self.cfg.robot.prim_path}")
            return

        painted = 0
        with Usd.EditContext(stage, stage.GetRootLayer()):
            for prim in Usd.PrimRange(root):
                if not prim.IsA(UsdShade.Shader):
                    continue
                shader = UsdShade.Shader(prim)
                for name in ("diffuseColor", "diffuse_color_constant", "base_color_constant"):
                    inp = shader.GetInput(name)
                    # A connected input is driven by a texture; overwriting it does nothing.
                    if inp and not inp.GetConnectedSource():
                        inp.Set(Gf.Vec3f(*color))
                        painted += 1
        logger.info(f"[RingInsertEnv] painted {painted} robot shader inputs {color}")

    def _write_object_pose(
        self,
        obj: RigidObject,
        pos: tuple[float, float, float],
        env_ids: Sequence[int],
        jitter: float,
        zero_velocity: bool = True,
    ) -> None:
        n = len(env_ids)
        xyz = np.tile(np.asarray(pos, dtype=np.float32), (n, 1))
        if jitter:
            xyz[:, :2] += self.rng.uniform(-jitter, jitter, size=(n, 2))

        root_state = obj.data.default_root_state[env_ids].clone()
        # World coordinates, not env-relative: the objects live at absolute prim paths
        # (/World/Object/...) and this task never clones environments, so
        # scene.env_origins is None rather than a (num_envs, 3) offset table.
        root_state[:, :3] = torch.as_tensor(xyz, device=self.device)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        obj.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        if zero_velocity:
            # Drop any linear/angular velocity carried over from the last episode.
            obj.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids=env_ids)

    # ── harness hooks (scripts/utils/dataset_record.py, dataset_replay.py) ────────

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        """No-op. The garment task built its cloth prim here; RigidObject self-initialises."""

    def get_all_pose(self):
        """Object poses to store with an episode, so replay can restore the exact start."""
        return {
            "ring": self.ring.data.root_state_w.cpu().numpy().tolist(),
            "ghost": self.ghost.data.root_state_w.cpu().numpy().tolist(),
        }

    def set_all_pose(self, pose):
        for name, obj in (("ring", self.ring), ("ghost", self.ghost)):
            if name not in pose:
                logger.warning(f"set_all_pose: no '{name}' entry, leaving it where it is")
                continue
            state = torch.as_tensor(pose[name], dtype=torch.float32, device=self.device)
            obj.write_root_state_to_sim(state)
