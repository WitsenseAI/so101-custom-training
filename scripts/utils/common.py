import argparse
import contextlib
import errno
import shutil
import time
from typing import TYPE_CHECKING
import numpy as np
import torch

from isaaclab.app import AppLauncher
from isaacsim.simulation_app import SimulationApp

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

@contextlib.contextmanager
def _rmtree_retrying_on_fuse():
    """Make shutil.rmtree survive FUSE's asynchronous unlinks.

    Datasets here live on a fuseblk (NTFS/exFAT) mount. rmtree unlinks every file and
    then immediately rmdir()s the directory, but FUSE completes unlinks lazily — a file
    still held open becomes a transient .fuse_hidden* entry — so the rmdir can see a
    directory that is logically empty and raise

        [Errno 39] Directory not empty

    lerobot calls rmtree deep inside save_episode/encode_episode_videos, after the video
    is encoded but before the metadata is written, so one such failure loses the whole
    episode. Retrying the rmdir for a couple of seconds is enough; the directory really
    is empty moments later. Put the dataset on an ext4 path to avoid this entirely.
    """
    original = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        for attempt in range(20):
            try:
                return original(path, *args, **kwargs)
            except OSError as err:
                if err.errno != errno.ENOTEMPTY:
                    raise
                time.sleep(0.1)
        # Twenty tries is not a race any more. The files are gone either way, so leave
        # the empty directory rather than lose a recorded episode over it.
        original(path, ignore_errors=True)

    shutil.rmtree = rmtree
    try:
        yield
    finally:
        shutil.rmtree = original


def save_episode(dataset, *args, **kwargs) -> None:
    """Save the episode, waiting for the image writer first.

    LeRobotDataset.save_episode() encodes the episode video and then rmtree()s the image
    directory, without ever draining the async image writer. Two things go wrong:

    1. The rmtree races the writer threads and dies with
       ``[Errno 39] Directory not empty``. save_episode() has already popped "size" and
       "task" off the episode buffer by then, so the next access to the buffer raises a
       confusing ``KeyError: 'size'`` that hides the real failure.
    2. Worse when it does not crash: frames still queued when ffmpeg runs are missing
       from the encoded video, silently shortening the episode.

    Draining first fixes both. Call this instead of dataset.save_episode().
    """
    writer = getattr(dataset, "image_writer", None)
    if writer is not None:
        writer.wait_until_done()
    with _rmtree_retrying_on_fuse():
        dataset.save_episode(*args, **kwargs)


def finalize(dataset) -> None:
    """Finalize the dataset if this lerobot has the concept.

    LeRobotDataset.finalize() only exists in lerobot >= 0.4. The 0.3.3 pinned next to
    Isaac Sim writes meta/episodes.jsonl and meta/episodes_stats.jsonl incrementally in
    save_episode(), so the dataset is already complete and calling it is a no-op —
    but the raw AttributeError at the end of a 20-episode session looks exactly like the
    recording failed, when nothing is wrong.
    """
    fn = getattr(dataset, "finalize", None)
    if fn is None:
        return
    with _rmtree_retrying_on_fuse():
        fn()


def clear_episode_buffer(dataset) -> None:
    """Discard the in-progress episode, waiting for the image writer first.

    lerobot's own LeRobotDataset.clear_episode_buffer() shutil.rmtree()s the episode's
    image directory without draining the async image writer. With image_writer_threads=8
    those threads are still flushing frames into that directory, so rmtree enumerates it,
    a new file lands, and the final rmdir raises

        [Errno 39] Directory not empty: .../images/observation.images.top_rgb/episode_000000

    which kills the recording session. It is timing-dependent: discarding a long episode
    after a pause often works, discarding right after pressing record does not.
    """
    writer = getattr(dataset, "image_writer", None)
    if writer is not None:
        writer.wait_until_done()
    with _rmtree_retrying_on_fuse():
        dataset.clear_episode_buffer()


SINGLE_ARM_HOME_POSITION = np.array(
    [
        -1.0363,  # shoulder_pan
        -1.7135,  # shoulder_lift
        1.4979,  # elbow_flex
        1.0534,  # wrist_flex
        -0.085,  # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)

# Left arm uses standard home position
LEFT_ARM_HOME_POSITION = np.array(
    [
        -1.2363,  # shoulder_pan
        -1.7135,  # shoulder_lift
        1.4979,  # elbow_flex
        1.0534,  # wrist_flex
        -0.085,  # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)
# Right arm with symmetric shoulder_pan
RIGHT_ARM_HOME_POSITION = np.array(
    [
        1.2363,  # shoulder_pan
        -1.7135,  # shoulder_lift
        1.4979,  # elbow_flex
        1.0534,  # wrist_flex
        -0.085,  # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)
DUAL_ARM_HOME_POSITION = np.concatenate(
    [LEFT_ARM_HOME_POSITION, RIGHT_ARM_HOME_POSITION]
)


def launch_app(parser: argparse.ArgumentParser) -> SimulationApp:
    """Launch Isaac Sim app from parser (parses args internally).

    Use this when you haven't parsed arguments yet.
    """
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return launch_app_from_args(args)


def launch_app_from_args(args: argparse.Namespace) -> SimulationApp:
    """Launch Isaac Sim app from already parsed arguments.

    Use this when arguments are already parsed (e.g., in subcommand handlers).

    Args:
        args: Already parsed command-line arguments (must include AppLauncher args).

    Returns:
        SimulationApp instance.
    """
    args.kit_args = (
        "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    )
    app_launcher = AppLauncher(vars(args))
    simulation_app = app_launcher.app
    return simulation_app


def close_app(simulation_app: SimulationApp) -> None:
    """Close Isaac Sim app."""
    simulation_app.close()


def stabilize_garment_after_reset(
    env: "DirectRLEnv",
    args: argparse.Namespace,
    num_steps: int = 20,
) -> None:
    """Stabilize garment after environment reset by running physics steps.

    Moves robot to home position and lets garment settle naturally after reset,
    preventing floating or clipping. This is critical for garment physics to
    initialize properly, especially when using CUDA device.

    Args:
        env: Environment instance.
        args: Command-line arguments containing task name.
        num_steps: Number of stabilization steps to run.
    """
    if num_steps <= 0:
        return

    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    try:
        initial_obs = env._get_observations()
        action_dim = (
            len(initial_obs["observation.state"])
            if "observation.state" in initial_obs
            else (12 if is_bimanual else 6)
        )
    except Exception:
        action_dim = 12 if is_bimanual else 6

    # Prefer the environment's own configured home pose. The constants below are the
    # bimanual garment task's and put a single arm 60 deg off to one side, so every
    # recorded episode would start from a posture the task never intended — and one the
    # env's own reset does not reproduce, leaving eval and training data disagreeing
    # about where an episode begins.
    home_joints = None
    robot = getattr(env, "robot", None)
    if robot is not None and not is_bimanual:
        try:
            home_joints = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
        except Exception:
            home_joints = None
    if home_joints is None:
        home_joints = DUAL_ARM_HOME_POSITION if is_bimanual else SINGLE_ARM_HOME_POSITION

    if len(home_joints) != action_dim:
        # Use warning from logger if available, otherwise print
        try:
            from witsense.utils.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"Home position dimension mismatch: got {len(home_joints)}, "
                f"expected {action_dim}. Using zeros."
            )
        except Exception:
            pass
        home_action = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)
    else:
        home_action = torch.from_numpy(home_joints).float().to(env.device).unsqueeze(0)

    for step_idx in range(num_steps):
        env.step(home_action)
        if (step_idx + 1) % 10 == 0 or step_idx == num_steps - 1:
            env.render()
