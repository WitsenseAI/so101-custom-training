# witsense UGV — MuSHR reinforcement learning

Isaac Lab packages for training a MuSHR RC car to drive over uneven terrain with PPO.
Ported from [WheeledLab](https://github.com/UWRobotLearning/WheeledLab).

For the SO-101 arm and imitation learning, see [witsense/README.md](witsense/README.md).

---

## Packages

```
witsense_assets      robot and terrain configs (MUSHR_SUS_CFG, MAIN_ASSETS_DIR)
witsense_ugv         MDP building blocks — ackermann actions, observations, curriculums
witsense_ugv_tasks   the environments, and the gym.register calls
witsense_ugv_rl      PPO training and playback entry points
```

## Install

Order matters — later packages import earlier ones.

```bash
conda activate env_isaaclab
cd so101-custom-training

pip install -e source/witsense_assets
pip install -e source/witsense_ugv
pip install -e source/witsense_ugv_tasks
pip install -e source/witsense_ugv_rl
```





## Train

All commands run from `source/witsense_ugv_rl`.

```bash
cd source/witsense_ugv_rl

python scripts/train_rl.py --headless -r RSS_ELEV_CONFIG \
    train.log.no_wandb=True train.log.video=False \
    2>&1 | tee ../../outputs/mushr_elev.log
```

`RSS_ELEV_CONFIG` defaults to 128 envs and 3000 iterations — see [Hardware limits](#hardware-limits).
Roughly 9.6 s/iteration, so ~8 hours.

<!-- Override anything from the command line:

```bash
env_setup.num_envs=64          # fewer environments
train.num_iterations=500       # shorter run
train.log.no_wandb=False       # log to wandb (project: witsense-ugv)
train.log.video=True           # needs --enable_cameras as well
``` -->

### Resume

```bash
python scripts/train_rl.py --headless -r RSS_ELEV_CONFIG \
    train.load_run=run-5217772 train.load_run_checkpoint=100 \
    train.num_iterations=2900 \
    train.log.no_wandb=True train.log.video=False
```

`num_iterations` counts iterations **to add**, not a target — resuming at 100 with 2900
ends at 3000.

### Watch it train

Only with few environments. The viewport renders every one, and 1024 will crash the
desktop on a 6 GB GPU.

```bash
python scripts/train_rl.py -r RSS_ELEV_CONFIG \
    env_setup.num_envs=2 train.num_iterations=5 \
    train.log.no_wandb=True train.log.video=False
```

---

## Play a trained policy

### From the Hub

The trained policy lives at
[`witsense-ai/mushr_wheeledlab_elevation`](https://huggingface.co/witsense-ai/mushr_wheeledlab_elevation).

```bash
cd source/witsense_ugv_rl

python scripts/download_hf_checkpoint.py
python scripts/play_policy.py -p logs/mushr_wheeledlab_elevation \
    --steps 1000 env.scene.num_envs=2
```

The download writes `logs/mushr_wheeledlab_elevation/` with the same layout training
produces — `models/model_2999.pt` plus `run_config.pkl` — so `-p` works straight away.

```bash
--repo-id witsense-ai/other_repo   # a different policy
--dest /somewhere/else             # parent directory for the run folder
--name my_run                      # run folder name
--force                            # overwrite an existing folder
```

The repo is private; the script reads `HF_TOKEN` or the cached `hf auth login`.

### From a local run

```bash
python scripts/play_policy.py -p logs/run-5217772 --checkpoint 2999 \
    --steps 1000 env.scene.num_envs=2
```

| flag | |
|---|---|
| `-p` | run folder under `logs/` |
| `--checkpoint` | iteration number; omit for the latest |
| `--steps` | playback length |
| `--video` | record to `<run>/playback/` instead of watching |
| `-sd` | save observations and actions to a `.pt` |

---



## Hardware limits

Measured on a 6 GB RTX 4050 / 20 GB RAM laptop. **System RAM binds first, not VRAM.**

| envs | VRAM | RAM (of 19.6 GB) | s/iter | |
|---|---|---|---|---|
| 128 | 4.1 GB | 14.4 GB | 9.6 | default |
| 256 | 4.2 GB | 17.9 GB | 11.9 | no room for a browser |
| 1024 | — | — | — | OOM-killed |

Idle baseline is ~2.9 GB. VRAM barely moves between 128 and 256 envs; RAM climbs 3.5 GB.

Upstream trains at 1024 envs × 5000 iterations = 655M timesteps. 128 × 3000 is 49M — about
7.5% of that, so expect a weaker policy than the WheeledLab results.

### Results at 128 envs / 3000 iterations

| | iteration 109 | iteration 2999 |
|---|---|---|
| mean reward | 1,872 | 11,348 |
| mean episode length | 18.5 | 97.6 |
| terminated stuck | 0.92 | 0.43 |
| reached goal | 0.008 | 0.020 |

It learns to drive toward the goal and stops getting stuck, but rarely arrives.

---
