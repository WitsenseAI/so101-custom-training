import gymnasium as gym

# No "Bi" in the id: scripts/utils/dataset_record.py keys off that substring to decide
# between a 12-dim bimanual and a 6-dim single-arm dataset schema.
gym.register(
    id="LeHome-SO101-Direct-RingInsert-v0",
    entry_point=f"{__name__}.ring_insert:RingInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ring_insert_cfg:RingInsertEnvCfg",
    },
)
