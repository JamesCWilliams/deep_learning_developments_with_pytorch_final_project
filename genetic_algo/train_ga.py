import argparse
import json

from ga_framework import GeneticAlgorithmConfig, GeneticAlgorithmTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GA policy population on a Gymnasium environment.")
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4", help="Gymnasium environment id")
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--elite_fraction", type=float, default=0.1)
    parser.add_argument("--tournament_size", type=int, default=3)
    parser.add_argument("--mutation_sigma", type=float, default=0.05)
    parser.add_argument(
        "--mutation_sigma_variance",
        type=float,
        default=None,
        help="Optional multiplier controlling exponential sigma resampling; 0 or None disable.",
    )
    parser.add_argument("--crossover_rate", type=float, default=0.5)
    parser.add_argument(
        "--crossover_type",
        type=str,
        default="uniform",
        choices=["uniform", "index", "blend"],
        help="Built-in crossover operator to use when breeding offspring",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per evaluation")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--unique_rollouts", action="store_true", help="Use distinct seeds per individual evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--terminate_on_truncation", action="store_true")
    parser.add_argument("--save_best", type=str, default="", help="Path to .npz for best genome")
    parser.add_argument("--state_dict", type=str, default="", help="Optional path to save torch state dict (pt)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes to use during evaluation",
    )
    parser.add_argument(
        "--worker_batch_size",
        type=int,
        default=None,
        help="Optional number of genomes to dispatch to each worker at once",
    )
    novelty_group = parser.add_argument_group("novelty search")
    novelty_group.add_argument(
        "--novelty",
        "--novelty_search",
        action="store_true",
        dest="novelty_search",
        help="Combine novelty ranking with fitness when selecting parents.",
    )
    novelty_group.add_argument(
        "--novelty-method",
        "--novelty_method",
        type=str,
        default="nearest_neighbors",
        choices=["nearest_neighbors", "diverse"],
        dest="novelty_method",
        help="How to measure novelty: K-nearest-neighbor distance or greedy diverse subset.",
    )
    novelty_group.add_argument(
        "--novelty-weight",
        "--novelty_weight",
        type=float,
        default=0.5,
        dest="novelty_weight",
        help="0.0 uses fitness only, 1.0 uses novelty only; values in between blend the ranks.",
    )
    novelty_group.add_argument(
        "--novelty-neighbors",
        "--novelty_neighbors",
        "--novelty-k",
        type=int,
        default=5,
        dest="novelty_neighbors",
        help="Number of neighbours to consider when novelty_method=nearest_neighbors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = GeneticAlgorithmConfig(
        env_id=args.env_id,
        population_size=args.population_size,
        generations=args.generations,
        elite_fraction=args.elite_fraction,
        tournament_size=args.tournament_size,
        mutation_sigma=args.mutation_sigma,
        mutation_sigma_variance=args.mutation_sigma_variance,
        crossover_rate=args.crossover_rate,
        crossover_type=args.crossover_type,
        evaluation_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        unique_rollouts=args.unique_rollouts,
        seed=args.seed,
        device=args.device,
        terminate_on_truncation=args.terminate_on_truncation,
        num_workers=args.num_workers,
        worker_batch_size=args.worker_batch_size,
        novelty_search=args.novelty_search,
        novelty_method=args.novelty_method,
        novelty_weight=args.novelty_weight,
        novelty_neighbors=args.novelty_neighbors,
    )

    trainer = GeneticAlgorithmTrainer(config)

    def log_callback(stats, population):
        print(json.dumps(stats))

    final_population = trainer.run(callback=log_callback)
    best = final_population.best()
    if best is None:
        raise RuntimeError("No evaluated individuals found; ensure at least one generation ran.")

    print(json.dumps({"best_fitness": best.fitness}))

    if args.save_best:
        output_path, state_dict = trainer.export_best(args.save_best)
        print(f"Saved best genome to {output_path}")
        if args.state_dict:
            import torch

            torch.save(state_dict, args.state_dict)
            print(f"Saved PyTorch state dict to {args.state_dict}")


if __name__ == "__main__":
    main()
