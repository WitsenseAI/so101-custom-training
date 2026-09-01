import os

from rsl_rl import runners


class OnPolicyRunner(runners.OnPolicyRunner):
    """Maps this repo's LogConfig onto rsl_rl's own logging.

    Upstream WheeledLab overrode learn() to add a tqdm bar and a NaN check. That override
    was written against an older rsl_rl — it used `obs, extras = env.get_observations()`,
    `alg.act(obs, critic_obs)` and `alg.compute_returns(critic_obs)`, none of which match
    the installed version, where get_observations() returns a single TensorDict. Stock
    learn() already saves checkpoints and supports wandb, so it is used unchanged.
    """

    def __init__(self, env, agent_cfg, log_cfg, device="cpu"):
        agent_cfg = dict(agent_cfg)
        agent_cfg["logger"] = "tensorboard" if log_cfg.no_wandb else "wandb"
        super().__init__(env, agent_cfg, log_cfg.run_log_dir, device)
        self.disable_logs = self.disable_logs or log_cfg.no_log

    def save(self, path: str, infos: dict | None = None) -> None:
        """Keep checkpoints under <run>/models/, which train_rl.py's --load-run expects."""
        path = os.path.join(os.path.dirname(path), "models", os.path.basename(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        super().save(path, infos)
