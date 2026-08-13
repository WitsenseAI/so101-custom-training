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
from pxr import Sdf, Usd, UsdShade

import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from witsense.assets.scenes.bedroom import MARBLE_BEDROOM_USD_PATH
from witsense.utils.constant import ASSETS_ROOT
from witsense.devices.action_process import preprocess_device_action
from witsense.tasks.ring_insert.ring_insert_cfg import RingInsertEnvCfg
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

        # Retexture here, while the scene prims have just been authored — not only on
        # reset, so the dark table is in place before the first frame is rendered.
        self._apply_table_texture()

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

        observations = {
            "action": self.actions.squeeze(0).cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": self.top_camera.data.output["rgb"]
            .cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.wrist_rgb": self.wrist_camera.data.output["rgb"]
            .cpu()
            .detach()
            .numpy()
            .squeeze(),
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

    # ── reset ────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        # Zero the velocities too: writing only positions lets the previous episode's
        # joint velocities survive and bias the first observation of the next one.
        self.robot.write_joint_velocity_to_sim(
            torch.zeros_like(joint_pos), joint_ids=None, env_ids=env_ids
        )

        self._write_object_pose(self.ring, self.cfg.ring_pos, env_ids, jitter=self.cfg.ring_xy_jitter)
        self._write_object_pose(self.ghost, self.cfg.ghost_pos, env_ids, jitter=0.0)
        self._apply_table_texture()

    def _apply_table_texture(self) -> None:
        """Retexture the table top so the white ring and white ghost stand out on it."""
        tex_id = self.cfg.table_texture_id
        if tex_id is None:
            return

        # Absolute, so this does not depend on the working directory the way the
        # bimanual task's randomiser does.
        tex_path = Path(ASSETS_ROOT) / "textures" / "surface" / f"{tex_id}.png"
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

    def _write_object_pose(
        self,
        obj: RigidObject,
        pos: tuple[float, float, float],
        env_ids: Sequence[int],
        jitter: float,
    ) -> None:
        n = len(env_ids)
        xyz = np.tile(np.asarray(pos, dtype=np.float32), (n, 1))
        if jitter:
            xyz[:, :2] += self.rng.uniform(-jitter, jitter, size=(n, 2))

        root_state = obj.data.default_root_state[env_ids].clone()
        root_state[:, :3] = torch.as_tensor(xyz, device=self.device) + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        root_state[:, 7:] = 0.0  # drop any linear/angular velocity from the last episode
        obj.write_root_state_to_sim(root_state, env_ids=env_ids)

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
