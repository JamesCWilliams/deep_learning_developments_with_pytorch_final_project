# Genetic Algorithm RL

Lightweight tools to evolve continuous control policies, then optionally export them to PyTorch.

## Quick start

Train a population with the built-in script:

```bash
python train_ga.py --env_id HalfCheetah-v5 --population_size 64 --generations 50 \
    --episodes 2 --max_steps 1000 --unique_rollouts --num_workers 4 \
    --novelty --novelty-method nearest_neighbors --novelty-weight 0.5 --novelty-neighbors 5 \
    --save_best halfcheetah_ga.npz --state_dict halfcheetah_ga.pt
```

Key flags:
- `--env_id` Gymnasium environment id.
- `--population_size` number of individuals per generation.
- `--generations` how many rounds to run.
- `--episodes` and `--max_steps` control rollout length.
- `--num_workers` enables parallel evaluation.
- `--save_best` writes the top genome (`.npz`), `--state_dict` saves PyTorch weights.

Switch crossover with `--crossover_type` (`uniform`, `index`, or `blend`). Enable novelty search with `--novelty` (alias `--novelty_search`) and tune it via `--novelty-method`, `--novelty-weight`, and `--novelty-neighbors`.

## Export and evaluate

After training, load the saved state dict for quick checks:

```bash
python evaluate_state_dict.py best_weights.pt --env_id HalfCheetah-v5 --episodes 5
```

## Project layout

- `train_ga.py` – command line entry point for training.
- `ga_framework/` – population management, operators, evaluation, and training loop.
- `models.py` – simple MLP policy with flat-parameter helpers.
- `evaluate_state_dict.py` – small helper for rollouts using saved weights.
