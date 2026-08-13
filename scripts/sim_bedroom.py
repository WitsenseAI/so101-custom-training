"""Launch the bedroom garment scene and hold position — a smoke test for the install.

    python scripts/sim_bedroom.py --enable_cameras --device cpu
    python scripts/sim_bedroom.py --enable_cameras --device cpu --headless

Run from the repository root: the garment asset and particle-config paths in
GarmentEnvCfg are relative to the current working directory.
"""

import argparse
import os

# witsense.devices installs a pynput keyboard listener at import time, which needs a
# display server. Nothing here teleoperates.
os.environ.setdefault("LEHOME_DISABLE_KEYBOARD", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeHome-BiSO101-Direct-Garment-v2")
parser.add_argument("--garment_name", default="Top_Long_Seen_0")
parser.add_argument("--garment_version", default="Release", choices=["Release", "Holdout"])
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import witsense.tasks  # noqa: F401  (registers the gym ids)


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # Only the garment tasks take these; ring_insert spawns rigid bodies instead.
    if hasattr(env_cfg, "garment_name"):
        env_cfg.garment_name = args.garment_name
        env_cfg.garment_version = args.garment_version

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    print(f"[sim_bedroom] {args.task} up on {args.device}")
    env.reset()

    # Actions are absolute joint position targets, so feeding back the measured
    # positions holds the arms still while the scene settles. Concatenated in scene
    # order, which is the order the env's action vector uses (one arm or two).
    arms = list(env.scene.articulations.values())
    for _ in range(args.steps):
        hold = torch.cat([a.data.joint_pos for a in arms], dim=-1)
        env.step(hold)

    env.close()
    print(f"[sim_bedroom] {args.steps} steps OK")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
