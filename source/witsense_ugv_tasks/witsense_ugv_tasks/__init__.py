import gymnasium as gym

from .elevation import MushrElevationRLEnvCfg, MushrElevationPlayEnvCfg
import witsense_ugv_tasks.elevation.config.agents.mushr as mushr_elevation_agents

gym.register(
    id="Isaac-MushrElevationRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MushrElevationRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_elevation_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrElevationPlayEnvCfg,
    },
)
