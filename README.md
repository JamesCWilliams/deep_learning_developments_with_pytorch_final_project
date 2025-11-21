# RL Baselines with CleanRL + MuJoCo

Authors: James Williams (jwill436@jh.edu), Caleb Dame (cdame1@jhu.edu)

This repo contains baseline implementations of **PPO**, **SAC**, and **TD3** using [CleanRL](https://github.com/vwxyzjn/cleanrl) on MuJoCo continuous-control tasks ( `Hopper-v5`, `Ant-v5`, `HalfCheetah-v5`, `Pusher-v5`).  
Logging is done with **Weights & Biases (W&B)**.

---

## 1. Environment Setup (`scripts/setup_env.sh`)

First, create and populate the Python environment and install all dependencies.

```bash
bash scripts/setup_env.sh
```

This script:

- Creates and activates a virtual environment
- Installs Python packages listed in [requirements.txt](requirements.txt)
- Sets up necessary system-level dependencies

After running this, make sure you activate the environment (e.g., source venv/bin/activate).

---

## 2. Running Baselines

Once the environment is ready, launch the baseline sweeps:

```bash
bash scripts/run_baselines.sh
```

By default this will:

- Run PPO, SAC, and TD3 (via cleanrl.ppo_continuous_action, cleanrl.sac_continuous_action, and cleanrl.td3_continuous_action).

- Train each algorithm on multiple environments (Hopper-v5, Ant-v5, HalfCheetah-v5, Pusher-v5).

- Sweep over 10 random seeds.

- Log results to W&B.

### 2.1. Core Environment Variables

All configuration is controlled via environment variables (with sane defaults). You can override any of these when calling the script, e.g.:

`PROJECT=hopper_sweep TIMESTEPS=250_000 CONCURRENCY=4 bash scripts/run_baselines.sh`

Below is a summary of the key parameters.

**Project / WandB**

- PROJECT (default: baseline)

W&B project name (--wandb-project-name).

- ENTITY (default: empty)

W&B entity (team or username). If not set, your default W&B account is used.

**Training Budget**

- TIMESTEPS (default: 1_000_000)

Number of environment steps per run (--total-timesteps passed to each CleanRL script).

**Sweep Dimensions**

- SEEDS (default: 1 2 3 4 5 6 7 8 9 10)

List of random seeds for each algo × env combination.

- ENVS (default: Hopper-v5 Ant-v5 HalfCheetah-v5 Pusher-v5)

List of environment IDs to run.

**Parallelism**

- CONCURRENCY (default: 1)

Maximum number of runs to execute in parallel (simple job-based semaphore).

**Performance Tuning**

- NUM_ENVS (default: 8)

Number of vectorized environments per run (--num-envs).

- BATCH_SIZE (default: 512)

Training batch size passed to the CleanRL scripts (--batch-size).

**Video Capture**

- CAPTURE_VIDEO (default: 1)

If 1: enable video capture (via --capture-video).

If 0: disable video capture.

Videos are typically only recorded for certain seeds to keep disk usage manageable (see script logic).

**Extra Args**

- EXTRA_ARGS (default: empty)

Additional CLI flags forwarded directly to the CleanRL scripts.

Example:

```EXTRA_ARGS="--learning-rate 3e-4 --gamma 0.98" bash scripts/run_baselines.sh```

---

## 3. Example Commands

### 3.1. Quick sanity run (shorter training)

```TIMESTEPS=50_000 SEEDS="1 2" ENVS="Hopper-v5" CONCURRENCY=1 bash scripts/run_baselines.sh```

### 3.2. Disable video capture

```CAPTURE_VIDEO=0 bash scripts/run_baselines.sh```

---

## 4. Weights & Biases

Each run is tracked with:

- --track

- --wandb-project-name "${PROJECT}"

- --wandb-entity "${ENTITY}" (if provided)

- --exp-name "${ALGO}" (e.g., ppo, sac, td3)

- WANDB_RUN_GROUP set to ppo_baseline, sac_baseline, or td3_baseline.

You can browse runs in the W&B UI, grouped by:

- Project: PROJECT

- Group: ppo_baseline, sac_baseline, td3_baseline

- Config: environment, seed, timesteps, etc.

## 5. Notes on MuJoCo / Rendering

The script sets:

- OMP_NUM_THREADS (default: 1)

- MUJOCO_GL (default: egl) for headless rendering.

If you run into MuJoCo or rendering issues (e.g., on local machines without proper GPU drivers), you may need to adjust MUJOCO_GL or install additional system libraries.
