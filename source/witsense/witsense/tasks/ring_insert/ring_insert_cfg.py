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

# roundtape.usda: 100 mm across the outside, 90 mm across the hole, 24 mm tall, origin
# at its centre — so it rests half its height above the table. The 5 mm wall matches a
# used-up roll of tape; it is thin enough that the collider needs sdfResolution 128.
RING_HALF_HEIGHT = 0.012
RING_INNER_RADIUS = 0.045

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
    #
    # decimation 3 over a 1/90 s physics step gives a 30 Hz control rate. That is the rate
    # the real dataset was recorded at and the fps the recording harness declares, so one
    # env step is one dataset frame. At the garment task's decimation=1 the env runs at
    # 90 Hz, which plays a 30 Hz action chunk three times too fast and labels 90 Hz
    # recordings as 30 Hz.
    decimation = 3
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

    # Isaac Lab defaults this to 0, which leaves the RTX camera buffers holding whatever
    # was rendered BEFORE the reset — so the first observation of every episode shows the
    # previous episode's arm pose and object placement. That is the frame a policy is
    # asked to act on and the first frame written into a recorded dataset.
    num_rerenders_on_reset: int = 2

    # random seed
    use_random_seed: bool = True
    random_seed: int = 42

    # ── task geometry ────────────────────────────────────────────────────────────
    # Laid out relative to the arm base at (0.23, -0.25). The base is rotated 180 deg
    # about z, so the arm reaches out along +y. Both objects sit ~0.21-0.23 m from the
    # base, inside the SO-101's working reach, and 0.14 m apart so the gripper can take
    # the ring without knocking the ghost.
    # Ring on the left of frame, ghost on the right, as in the real top-camera view.
    # Larger world x projects to the left of that image, so the ring takes the larger x.
    ring_pos: tuple[float, float, float] = (0.30, -0.03, TABLE_Z + RING_HALF_HEIGHT)
    ghost_pos: tuple[float, float, float] = (0.16, -0.05, TABLE_Z)

    # Per-episode uniform jitter (metres, +/-) applied to the ring's x and y on reset.
    # Without this every demonstration starts identically and the policy memorises one
    # trajectory instead of learning to look. At 0.04 the worst-case spawn is still
    # 0.24 m from the base and 0.10 m clear of the ghost.
    ring_xy_jitter: float = 0.04

    # A free-standing ghost gets knocked over on almost every early attempt, which makes
    # the task unlearnable before it is learnable. Kinematic pins it in place.
    ghost_kinematic: bool = True

    # Per-joint uniform jitter (degrees, +/-) on the arm's home pose at reset.
    #
    # OFF by default, because the current policy cannot take it: all 40 demonstrations
    # begin from the identical folded pose, and +/-3 deg took success from 4/10 to 0/15.
    # That is a real weakness, but the fix is to record demonstrations from varied starts,
    # not to jitter at eval and measure the damage. Turn it on once the training data has
    # the variation to match, or via --arm_jitter_deg to probe how much the policy takes.
    arm_start_jitter_deg: float = 0.0

    # Horizontal distance over which insertion_progress' approach term is scored. The
    # ring and ghost start ~0.14 m apart, so this makes "untouched" score about 0.
    progress_approach_range: float = 0.15

    # Success: ring low enough to be around the ghost rather than resting on top of it.
    # Ring centre is TABLE_Z+0.012 when it is down on the table and TABLE_Z+0.060 when
    # perched on the ghost's head, so 0.03 separates the two cleanly.
    # Encircling is physically only possible within RING_INNER_RADIUS - GHOST_HALF_WIDTH
    # = 10 mm of centre; the xy tolerance is slack around that, the z check discriminates.
    success_xy_tol: float = 0.015
    success_z_max: float = TABLE_Z + 0.03

    # ── table appearance ─────────────────────────────────────────────────────────
    # A file under Assets/textures/surface, or None to keep the scene's white material.
    #
    # real_mat.png is generated to rgb (147, 124, 41), sampled off clean patches of the
    # orange desk mat in the real so101_pick_and_place_ring_33 frames. That mat is the
    # single largest thing in both real camera views, so it is the biggest, cheapest
    # visual gap to close. The numbered textures 1-100 are the LeHome surface set;
    # 19.png is the closest of them but still reads olive rather than orange.
    table_texture: str | None = "real_mat.png"
    # Same shader the bimanual task's texture randomiser targets — see
    # tasks/bedroom/config_file/particle_garment_cfg.yaml.
    table_shader_path: str = (
        "/World/Scene/scene/Table038/looks/M_Table038a/UsdPreviewSurface/________7/________7"
    )

    # so101_follower_good.usd is yellow; the real arm is white with black motors. The arm
    # is a large bright object in the top camera view, so the colour is worth matching.
    # None leaves the asset's own materials alone.
    robot_color: tuple[float, float, float] | None = (0.90, 0.90, 0.88)

    # so101_follower_good.usd is stricter than the real arm: measured over all 18861
    # frames of so101_pick_and_place_ring_33, the real robot goes 7.1 deg past the USD's
    # elbow_flex limit, 5.4 past wrist_flex and 0.5 past shoulder_lift. That clips both
    # the home pose and part of any trajectory the policy replays. These are the real
    # ranges with 5 deg of margin, in degrees, written into PhysX at startup.
    # None keeps the USD's own limits.
    #
    # Applied after the articulation exists, which is also why the home pose is repeated
    # in home_pose_deg below: Isaac Lab validates init_state.joint_pos against the USD's
    # limits during __init__ and raises before any of this can run, so init_state has to
    # stay inside the USD's narrower range and the true pose is written afterwards.
    joint_limits_deg: dict[str, tuple[float, float]] | None = {
        "shoulder_pan": (-110.0, 110.0),
        "shoulder_lift": (-106.0, 106.0),
        "elbow_flex": (-100.0, 103.0),
        "wrist_flex": (-100.0, 106.0),
        "wrist_roll": (-160.0, 160.0),
        "gripper": (-10.0, 100.0),
    }

    # The real robot's home pose in degrees, median over the 33 episodes of
    # so101_pick_and_place_ring_33. This is what actually reaches the sim; init_state's
    # joint_pos below is the same pose clamped to the USD limits, present only so
    # Isaac Lab's startup validation passes.
    home_pose_deg: dict[str, float] | None = {
        "shoulder_pan": 0.4,
        "shoulder_lift": -92.1,
        "elbow_flex": 93.6,
        "wrist_flex": 70.5,
        "wrist_roll": 3.5,
        "gripper": 1.4,
    }

    # ── robot ────────────────────────────────────────────────────────────────────
    # Same x/y the bimanual task gives its right arm. The base is rotated 180 deg about
    # z, so the arm reaches out along +y, over the objects placed above.
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.23, -0.25, TABLE_Z - ROBOT_BASE_OFFSET),
            rot=(0.0, 0.0, 0.0, 1.0),
            # The real robot's home pose, median over the 33 episodes of
            # so101_pick_and_place_ring_33 (shoulder_lift's spread across them is 0.1 deg
            # — every episode starts here). All-zeros leaves the arm stretched out
            # horizontally, which aims the wrist camera at the room instead of the table
            # and hands the policy a first observation nothing like its training data.
            joint_pos={
                "shoulder_pan": 0.0061,  # 0.4 deg
                "shoulder_lift": -1.6080,  # -92.1
                "elbow_flex": 1.5708,  # 90, the USD's cap; home_pose_deg lifts it to 93.6
                "wrist_flex": 1.2298,  # 70.5
                "wrist_roll": 0.0606,  # 3.5
                "gripper": 0.0243,  # 1.4
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
    #
    # Resolutions match the real so101_pick_and_place_ring_33 dataset the ACT policy was
    # trained on — top 640x480, wrist 1280x720 — so a checkpoint trained on real frames
    # takes sim frames without reshaping, and sim recordings share the real dataset's
    # schema. Change these and scripts/run_eval.py refuses to run rather than quietly
    # feeding the policy the wrong shape.
    # Re-aimed for this arm. The bimanual task's offset left the camera pointing almost
    # horizontally (forward z = -0.05), so its centre ray met the table 3.4 m away and it
    # filmed the bedroom furniture instead of the workspace.
    #
    # The mounting authored for the SO-101 gripper in tasks/bedroom, kept as-is. It is a
    # physical camera placement on the real arm, so it is not ours to re-derive: the view
    # it gives depends on the arm's pose, and judging it from the home pose alone is what
    # led to it being wrongly "corrected".
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
        width=1280,
        height=720,
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
