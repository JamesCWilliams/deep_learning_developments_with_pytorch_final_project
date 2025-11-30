# Running GA-Pretrained Policies with run_from_pretrain.sh

`run_from_pretrain.sh` launches multiple RL runs starting from GA-pretrained actor weights.

You specify:
- algo: `ppo`, `sac`, or `td3`
- environment: e.g. `Hopper-v5`, `Ant-v5`
- directory of `.pt` weights (one run per file)

Supports:
- CPU or GPU (`CUDA=False` or `CUDA=True`)
- Parallel jobs (`CONCURRENCY`)
- W&B project naming (`PROJECT`)

---

## Basic Usage

`./scripts/run_from_pretrain.sh <algo> <env_id> <weights_dir>`

Examples:

### Run PPO on Hopper-v5 with GA weights (GPU)

```
PROJECT="GA1 - PPO - Hopper" \
CUDA=True \
./scripts/run_from_pretrain.sh ppo Hopper-v5 ./ga_weights/ga1/Hopper-v5/PPO
```

### Run SAC on Ant-v5 (CPU-only) with 3 jobs in parallel

```
PROJECT="GA1 - SAC - Ant" \
CONCURRENCY=3 \
CUDA=False \
./scripts/run_from_pretrain.sh sac Ant-v5 ./ga_weights/ga1/Ant-v5/SAC
```

---

## How It Works

For each weight file, e.g.:

`./genetic_algo/ga1/Ant-v5/SAC/weights1.pt`

It runs:

`./scripts/run_sac_with_weights.sh Ant-v5 <seed> ./genetic_algo/.../weights1.pt`

Experiment names automatically include the weight name:

`sac_preinit_weights1`

Seeds default to:

`1 2 3 4 5 6 7 8 9 10`

Override:

`SEEDS="3 4 5" ./scripts/run_from_pretrain.sh ...`

---

## Quick Examples

### 10-run SAC sweep on Ant-v5 (CPU-only)

```
PROJECT="GA1 - SAC - Ant" \
CUDA=False \
./scripts/run_from_pretrain.sh sac Ant-v5 ./ga_weights/ga1/Ant-v5/SAC
```

### PPO sweep on Hopper-v5 (GPU)

```
PROJECT="GA1 - PPO - Hopper" \
CUDA=True \
./scripts/run_from_pretrain.sh ppo Hopper-v5 ./ga_weights/ga1/Ant-v5/PPO
```

---

## Notes

- `PROJECT` → W&B project name
- `CUDA=True/False` → GPU or CPU
- `CONCURRENCY` → number of parallel jobs
- One run per weight file
- Weight file names automatically appear in run names

