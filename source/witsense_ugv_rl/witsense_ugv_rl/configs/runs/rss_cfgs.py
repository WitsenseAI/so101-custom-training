from isaaclab.utils import configclass

from witsense_ugv_rl.configs import (
    EnvSetup, RslRlRunConfig, RLTrainConfig, AgentSetup, LogConfig
)

@configclass
class RSS_ELEV_CONFIG(RslRlRunConfig):
    # Sized for a 6 GB / 20 GB laptop. The binding limit is SYSTEM RAM, not VRAM:
    # measured steady-state usage is 14.4 GB at 128 envs and 17.9 GB at 256 (of 19.6 GB,
    # over a 2.9 GB idle baseline), and 1024 envs is killed by the OOM killer.
    # VRAM barely moves — 4.1 GB at 128, 4.2 GB at 256.
    # Upstream runs 1024 envs x 5000 iterations; that budget needs a bigger machine.
    env_setup = EnvSetup(
        num_envs=128,
        task_name="Isaac-MushrElevationRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=3000,   # ~9.6 s/iteration measured at 128 envs -> ~8 h
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )
