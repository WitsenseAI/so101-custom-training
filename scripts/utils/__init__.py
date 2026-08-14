"""Utility functions for LeHome scripts."""

from . import common
from . import dataset_inspection
from . import parser

# dataset_processing is NOT imported at module level: it needs
# lerobot.datasets.dataset_tools, which only exists in lerobot >= 0.4, and lerobot 0.4
# needs numpy >= 2 while isaacsim-kernel pins numpy == 1.26.0. Recording and replay use
# LeRobotDataset only, which lerobot 0.3.3 has — so importing it eagerly would block
# `record` on a dependency `record` never uses. Import it lazily where merge/augment run.

# Note: evaluation, dataset_record and dataset_replay are not imported at module level
# to avoid importing Isaac Sim modules before SimulationApp is launched.
# They should be imported lazily when needed (after SimulationApp is launched).

# Export commonly used functions for convenience
from .parser import (
    setup_record_parser,
    setup_replay_parser,
    setup_inspect_parser,
    setup_read_parser,
    setup_augment_parser,
    setup_merge_parser,
    setup_eval_parser,
)
from .common import launch_app, launch_app_from_args, close_app
from .dataset_inspection import inspect, read_states

# Note: evaluation functions are not imported at module level to avoid
# importing Isaac Sim modules before SimulationApp is launched.
# Import them lazily when needed: from .utils.evaluation import <function>

__all__ = [
    "setup_record_parser",
    "setup_replay_parser",
    "setup_inspect_parser",
    "setup_read_parser",
    "setup_augment_parser",
    "setup_merge_parser",
    "setup_eval_parser",
    "launch_app",
    "launch_app_from_args",
    "close_app",
    "inspect",
    "read_states",
    # Note: augment_ee_pose / merge_datasets / merge_garment_info are not exported —
    # import them from .utils.dataset_processing, which needs lerobot >= 0.4.
    # Note: evaluation functions, "replay" and "record_dataset" are not exported
    # at module level to avoid importing Isaac Sim modules before SimulationApp
    # is launched. Import them lazily when needed:
    #   from .utils.evaluation import <function>
    #   from .utils import dataset_replay, dataset_record
]
