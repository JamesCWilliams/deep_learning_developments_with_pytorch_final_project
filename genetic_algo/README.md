# Genetic Algorithm RL

Lightweight tools to evolve continuous control policies, then optionally export them to PyTorch.

## Quick start

Train a population with the built-in script:

```bash
python train_ga.py --env_id HalfCheetah-v5 --population_size 64 --generations 50 \
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

## Export and evaluate

After training, the top-performing actor parameters are written to ``--best_actor_path``
(``run_{flavor}.pt`` by default). Load this artifact with ``torch.load`` in your
deployment or evaluation pipeline. If you have a standard PyTorch state dict from other
workflows, ``evaluate_state_dict.py`` can roll it out for quick checks.

## Project layout

- `train_ga.py` – command line entry point for training.
- `ga_framework/` – population management, operators, evaluation, and training loop.
- `models.py` – simple MLP policy with flat-parameter helpers.
- `evaluate_state_dict.py` – small helper for rollouts using saved weights.
