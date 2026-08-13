from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from witsense.assets.robots.lerobot import SO101_FOLLOWER_CFG
from witsense.utils.constant import ASSETS_ROOT

OBJECTS_ROOT = Path(ASSETS_ROOT) / "objects"
RING_USD_PATH = str(OBJECTS_ROOT / "ring" / "roundtape.usda")
GHOST_USD_PATH = str(OBJECTS_ROOT / "ghost" / "ghost.usd")

# Measured from the scene USD, not guessed: Table038's world bounding box is
# x [-0.522, 0.468], y [-0.400, 0.400], z [0.000, 0.521]. The top face is that last
# number. (The bimanual task puts its arm bases at 0.5, which is 21 mm *inside* the
# table — the base plate geometry starts 30 mm above the articulation origin, so it
# still lands on the surface.)
TABLE_Z = 0.521

# roundtape.usda: 100 mm across the outside, 80 mm across the hole, 24 mm tall, origin
# at its centre — so it rests half its height above the table.
RING_HALF_HEIGHT = 0.012
RING_INNER_RADIUS = 0.040

# ghost.usd: 62 x 70 mm footprint, 48 mm tall, origin at its BASE (z spans 0 to 0.048),
# so it rests at exactly TABLE_Z with no half-height offset.
GHOST_HEIGHT = 0.048
GHOST_HALF_WIDTH = 0.035

# so101_follower_good.usd's lowest geometry sits 30 mm above its articulation origin,
# so the origin has to go below the table top for the base plate to land on it.
ROBOT_BASE_OFFSET = 0.030


@configclass
class RingInsertEnvCfg(DirectRLEnvCfg):
    # env — single SO-101, so 6 joints in and 6 out (the bimanual garment task is 12)
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0
    action_space = 6
    observation_space = 6
    state_space = 0

    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 90,
        render_interval=decimation,
        render=render_cfg,
        use_fabric=False,
    )

    # random seed
    use_random_seed: bool = True
    random_seed: int = 42

    # ── task geometry ────────────────────────────────────────────────────────────
    # Laid out relative to the arm base at (0.23, -0.25). The base is rotated 180 deg
    # about z, so the arm reaches out along +y. Both objects sit ~0.21-0.23 m from the
    # base, inside the SO-101's working reach, and 0.14 m apart so the gripper can take
    # the ring without knocking the ghost.
    ring_pos: tuple[float, float, float] = (0.16, -0.05, TABLE_Z + RING_HALF_HEIGHT)
    ghost_pos: tuple[float, float, float] = (0.30, -0.03, TABLE_Z)

    # Per-episode uniform jitter (metres, +/-) applied to the ring's x and y on reset.
    # Without this every demonstration starts identically and the policy memorises one
    # trajectory instead of learning to look. At 0.04 the worst-case spawn is still
    # 0.24 m from the base and 0.10 m clear of the ghost.
    ring_xy_jitter: float = 0.04

    # A free-standing ghost gets knocked over on almost every early attempt, which makes
    # the task unlearnable before it is learnable. Kinematic pins it in place.
    ghost_kinematic: bool = True

    # Success: ring low enough to be around the ghost rather than resting on top of it.
    # Ring centre is TABLE_Z+0.012 when it is down on the table and TABLE_Z+0.060 when
    # perched on the ghost's head, so 0.03 separates the two cleanly.
    # Encircling is physically only possible within RING_INNER_RADIUS - GHOST_HALF_WIDTH
    # = 5 mm of centre; the xy tolerance is slack around that, the z check discriminates.
    success_xy_tol: float = 0.015
    success_z_max: float = TABLE_Z + 0.03

    # ── table appearance ─────────────────────────────────────────────────────────
    # The bedroom table ships white, and so are the ghost and the ring — three white
    # things on top of each other are unreadable to a camera and to a person scoring
    # rollouts. Fixed dark table: Assets/textures/surface/<id>.png. 76 is the darkest
    # of the 100 (near-flat neutral grey, mean rgb 63/62/59); 10 and 51 are warm wood
    # tones. Set to None to keep the scene's own white material.
    table_texture_id: int = 76
    # Same shader the bimanual task's texture randomiser targets — see
    # tasks/bedroom/config_file/particle_garment_cfg.yaml.
    table_shader_path: str = (
        "/World/Scene/scene/Table038/looks/M_Table038a/UsdPreviewSurface/________7/________7"
    )

    # ── robot ────────────────────────────────────────────────────────────────────
    # Same x/y the bimanual task gives its right arm. The base is rotated 180 deg about
    # z, so the arm reaches out along +y, over the objects placed above.
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.23, -0.25, TABLE_Z - ROBOT_BASE_OFFSET),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),
    )

    ring: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/Ring",
        spawn=sim_utils.UsdFileCfg(usd_path=RING_USD_PATH),
        init_state=RigidObjectCfg.InitialStateCfg(pos=ring_pos),
    )

    ghost: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/Ghost",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GHOST_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=ghost_kinematic),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=ghost_pos),
    )

    # ── cameras ──────────────────────────────────────────────────────────────────
    # Names match what the recording harness expects for a single arm:
    # observation.images.top_rgb and observation.images.wrist_rgb.
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    # Re-aimed for one arm. The rotation is the bimanual task's, unchanged — the parent
    # frame is the same, so keeping it preserves the exact viewing angle that setup had.
    # Only the position moves, by the amount that recentres the same camera->workspace
    # vector on the new workspace: the bimanual camera sat at world (-0.015, 0.19, 1.06)
    # looking at a garment at (-0.04, -0.05, 0.52); this one sits at (0.255, 0.20, 1.07)
    # looking at (0.23, -0.04, 0.53), the midpoint of the ring and the ghost. The arm
    # base ends up 0.21 m behind that midpoint, the same as before, so the arm frames the
    # way both arms used to.
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.025, -0.45, 0.578),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),  # wxyz
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
