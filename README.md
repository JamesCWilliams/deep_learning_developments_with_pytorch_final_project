# RL Baselines with CleanRL + MuJoCo

Authors: James Williams (jwill436@jh.edu), Caleb Dame (cdame1@jhu.edu)

This repo contains baseline implementations of **PPO** and **SAC** using [CleanRL](https://github.com/vwxyzjn/cleanrl) on MuJoCo continuous-control tasks ( `Hopper-v5`, `Ant-v5`, `HalfCheetah-v5`, `Pusher-v5`).  
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

## 2. Running Baseline Agents

Baselines are launched using *per-algorithm* sweep scripts:

```bash
scripts/run_ppo_baseline_sweep.sh
scripts/run_sac_baseline_sweep.sh
```

Each sweep script:

- Trains one algorithm (PPO or SAC) 
- on one environment (e.g., Hopper-v5) 
- across a configurable set of random seeds 
- optionally in parallel 
- logs each run to a configurable W&B project and entity

| Variable        | Default                 | Description                        |
| --------------- | ----------------------- | ---------------------------------- |
| `PROJECT`       | `ppo_<ENV_ID>_baseline` | W&B project name                   |
| `SEEDS`         | `1 2 3 4 5 6 7 8 9 10`  | Seeds used in the sweep            |
| `ENTITY`        | `ga-rl-final-project`   | Your W&B entity (team or username) |
| `CUDA`          | `True`                  | Whether to use CUDA                |
| `CAPTURE_VIDEO` | `0`                     | Enable video logging               |
| `CONCURRENCY`   | `1`                     | Max number of parallel workers     |

Example with overrides:

```bash
PROJECT=ppo_hopper_baseline \ 
SEEDS="1 2 3 4 5" \
CONCURRENCY=2 \
bash scripts/run_ppo_baseline_sweep.sh Hopper-v5
```

Each seed is executed by calling:

```bash
scripts/run_ppo_baseline.sh <ENV_ID> <SEED>
scripts/run_sac_baseline.sh <ENV_ID> <SEED>
```

The CleanRL scripts automatically record:

- episodic returns
- losses (actor/critic/value/entropy)
- SPS and timing metrics
- periodic model checkpoints
- final model weights
- optional rollout videos (if enabled)

You must login to W&B once:

```bash
wandb login
```

---

## 3. Generating Generative Algorithm Weights

Train a population with the built-in script:

```bash
python genetic_algo/train_ga.py --env_id HalfCheetah-v5 --population_size 64 --generations 50 \
    --episodes 2 --max_steps 1000 --unique_rollouts --num_workers 4 \
    --novelty --novelty-method nearest_neighbors --novelty-weight 0.5 --novelty-neighbors 5 \
    --best_actor_path halfcheetah_ga.pt
```

Key flags:
- `--env_id` Gymnasium environment id.
- `--population_size` number of individuals per generation.
- `--generations` how many rounds to run.
- `--episodes` and `--max_steps` control rollout length.
- `--num_workers` enables parallel evaluation.
- `--best_actor_path` writes the top-performing policy parameters.
- `--activation` chooses the policy activation (`tanh` or `relu`).

Switch crossover with `--crossover_type` (`uniform`, `index`, or `blend`). Enable novelty search with `--novelty` (alias `--novelty_search`) and tune it via `--novelty-method`, `--novelty-weight`, and `--novelty-neighbors`.

**Export and evaluate**

After training, the top-performing actor parameters are written to ``--best_actor_path``
(``run_{flavor}.pt`` by default). Load this artifact with ``torch.load`` in your
deployment or evaluation pipeline. If you have a standard PyTorch state dict from other
workflows, ``evaluate_state_dict.py`` can roll it out for quick checks.

**Project layout**

- `train_ga.py` – command line entry point for training.
- `ga_framework/` – population management, operators, evaluation, and training loop.
- `models.py` – simple MLP policy with flat-parameter helpers.
- `evaluate_state_dict.py` – small helper for rollouts using saved weights.

---

## 4. Running GA-Weights Agents

These agents are initialized from `.pt` weight files generated under Step 3, where base policy weights were evolved using an external Genetic Algorithm.

GA-initialized runs are launched using a unified sweep script:

```bash
bash scripts/run_from_pretrain.sh <ALGO> <ENV_ID> <WEIGHTS_DIR>
```

Example:

```bash
bash scripts/run_from_pretrain.sh ppo Hopper-v5 ga_weights/ga1/Hopper-v5/PPO/
```

Arguments:

| Positional Arg | Meaning                                             |
| -------------- | --------------------------------------------------- |
| `ALGO`         | `ppo` or `sac`                                      |
| `ENV_ID`       | Gym environment ID (e.g., `Hopper-v5`)              |
| `WEIGHTS_DIR`  | Directory containing one or more `.pt` weight files |

The script finds all `.pt` files inside `WEIGHTS_DIR` and launches one run per file.

Environment variables (optional):

| Variable        | Default                 | Description                                |
| --------------- | ----------------------- | ------------------------------------------ |
| `PROJECT`       | `<algo>_<env>_pretrain` | W&B project name                           |
| `SEEDS`         | `1 2 3 4 5 6 7 8 9 10`  | Seeds assigned round-robin to weight files |
| `ENTITY`        | `ga-rl-final-project`   | W&B entity (username/team)                 |
| `CUDA`          | `False`                 | Whether to use CUDA (`True`/`False`)       |
| `CAPTURE_VIDEO` | `0`                     | Enable video capture                       |
| `CONCURRENCY`   | `1`                     | Max number of parallel jobs                |

Example:

```bash
PROJECT=ga2_hopper \
SEEDS="1 2 3 4" \
CUDA=True \
CONCURRENCY=2 \
bash scripts/run_from_pretrain.sh sac Hopper-v5 ga_weights/sac/hopper/
```

`run_from_pretrain.sh` dynamically selects the correct algorithm runner:

| Algorithm | Runner Script                              |
| --------- | ------------------------------------------ |
| `ppo`     | `scripts/run_ppo_pretrain.sh`              |
| `sac`     | `scripts/run_sac_pretrain.sh`              |

For each weight file:

- A matching seed is chosen

- A unique run name is created:
```
<algo>_preinit_<weightfilename>_<env>_seed<seed>_<timestamp>
```

- Runs are logged to W&B

Algorithm runners:

- Loads the first two hidden layers of the PPO actor from the GA file

- Runs CleanRL PPO for the given `--total_timesteps`

- Logs checkpoints and metrics

- Writes evaluation summaries to `data/<run_id>/summary.npz`

Configurable via:

| Variable        | Default             |
| --------------- | ------------------- |
| `PROJECT`       | `ppo_pretrain_runs` |
| `TIMESTEPS`     | `1_000_000`         |
| `NUM_ENVS`      | `1`                 |
| `CAPTURE_VIDEO` | `0`                 |
| `CUDA`          | `True`              |

---

## 5. Running Evaluation Episodes

After training both **baseline** and **GA-initialized** policies, we evaluate the final models offline and export episode-level metrics to individual CSV files for each algorithm+environment+weights combination. These CSVs are then aggregated into a combined CSV.

There are two main scripts:

```
`scripts/evaluate_wandb_policies.py`: pull final checkpoints from W&B and run eval episodes.
`scripts/aggregate_results.py`: combine all eval CSVs into a single `aggregated_results.csv`.
```

### 1. Export episode-level metrics from W&B

```bash
python scripts/evaluate_wandb_policies.py \
  --entity <WANDB_ENTITY> \
  --project <WANDB_PROJECT> \
  --algo <ppo|sac> \
  --output-csv eval_results/<alg>_<env>_<weights>.csv \
  [--episodes N] \
  [--device cpu|cuda] \
  [--env-id-filter ENV_ID] \
  [--max-runs K]
```

Typical examples:

**PPO baseline on Hopper-v5**:

```
python scripts/evaluate_wandb_policies.py \
  --entity ga-rl-final-project \
  --project ppo_Hopper-v5_baseline \
  --algo ppo \
  --env-id-filter Hopper-v5 \
  --episodes 100 \
  --device cuda \
  --output-csv eval_results/ppo_Hopper-v5_baseline.csv
```

**SAC GA-initialized agents on HalfCheetah-v5**:

```
python scripts/evaluate_wandb_policies.py \
  --entity ga-rl-final-project \
  --project sac_HalfCheetah-v5_pretrain \
  --algo sac \
  --env-id-filter HalfCheetah-v5 \
  --episodes 100 \
  --device cuda \
  --output-csv eval_results/sac_HalfCheetah-v5_ga2.csv
```

The aggregator assumes filenames of the form: `eval_results/<algo>_<env_id>_<weights>.csv` e.g. ppo_Hopper-v5_baseline.csv, sac_Hopper-v5_ga2.csv.

**What `evaluate_wandb_policies.py` does**

For each run in the specified W&B project,

1. Recovers `env_id`, `seed`, `exp_name`, `total_timesteps`, `gamma` from the run config.

2. Finds a final checkpoint:

    - Prefer a W&B model artifact containing `model_final.pt`.

    - If no artifact is found, fall back to the run’s Files tab:

        - Prefer `model_final.pt`

        - Otherwise any `.pt` checkpoint.

3. Loads the checkpoint into the correct CleanRL model:

    - PPO: `cleanrl.ppo_continuous_action.Agent`

    - SAC: `cleanrl.sac_continuous_action.Actor`

4. Runs N evaluation episodes (default `--episodes 100`) with deterministic actions.

5. Records for each episode:

    - `episodic_return (sum of rewards)`

    - `episode_length` (steps until termination/truncation)

    - `max_episode_steps` (environment horizon, if available)

    - `duration_rate = episode_length / max_episode_steps`

Writes all episode records to a CSV in `eval_results/`.

### 2. Aggregating all evaluation results

Once all desired CSVs are in `eval_results/` (e.g., baseline + GA variants across envs), run from the repo root:

```
python scripts/aggregate_results.py
```

This script aggregates each individual CSV file into a combined file, saving as columns the algorithm, weights, and environment, as well as metrics, while dropping some run-specific metadata columns: `project`, `run_id`, `run_name`, `exp_name`, `total_timesteps`.

Results are written to the repo root as `aggregated_results.csv`.

---