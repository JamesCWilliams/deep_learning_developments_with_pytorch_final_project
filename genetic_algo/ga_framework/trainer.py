from functools import partial
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch.nn as nn

from models import MLPActor

from .population import Individual, Population
from .selection import elitist_selection, tournament_selection
from .operators import gaussian_mutation, uniform_crossover, index_crossover, blend_crossover
from .evaluator import EvaluationConfig, evaluate_population
from utils import select_novel_rows, select_diverse_rows

from scipy.stats import rankdata


def _flatten_novelty_summary(summary):
    """Flatten observation/action statistics into a feature vector."""
    features = []
    for key in (
        "observation_mean",
        "observation_quantiles",
        "action_mean",
        "action_quantiles",
    ):
        value = summary.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        features.append(arr.reshape(-1))
    total_steps = summary.get("total_steps")
    if total_steps is not None:
        features.append(np.array([float(total_steps)], dtype=np.float32))
    if not features:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(features).astype(np.float32, copy=False)

try:  # tqdm is optional at runtime
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore


class GeneticAlgorithmConfig:
    def __init__(
        self,
        env_id,
        population_size,
        generations,
        elite_fraction=0.1,
        tournament_size=5,
        mutation_sigma=0.02,
        mutation_sigma_variance=None,
        crossover_rate=0.5,
        crossover_type="uniform",
        evaluation_episodes=1,
        max_steps_per_episode=1000,
        unique_rollouts=True,
        seed=0,
        device="cpu",
        terminate_on_truncation=False,
        num_workers=1,
        worker_batch_size=None,
        policy_hidden_sizes=None,
        policy_activation=nn.Tanh,
        policy_final_init_std=None,
        novelty_search=False,
        novelty_method="nearest_neighbors",
        novelty_weight=0.5,
        novelty_neighbors=5,
    ):
        self.env_id = env_id
        self.population_size = population_size
        self.generations = generations
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.mutation_sigma = mutation_sigma
        self.mutation_sigma_variance = mutation_sigma_variance
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.evaluation_episodes = evaluation_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.unique_rollouts = unique_rollouts
        self.seed = seed
        self.device = device
        self.terminate_on_truncation = terminate_on_truncation
        self.num_workers = num_workers
        self.worker_batch_size = worker_batch_size
        self.policy_hidden_sizes = policy_hidden_sizes
        self.policy_activation = policy_activation
        self.policy_final_init_std = policy_final_init_std
        self.novelty_search = novelty_search
        self.novelty_method = novelty_method
        self.novelty_weight = novelty_weight
        self.novelty_neighbors = novelty_neighbors


class GeneticAlgorithmTrainer:
    """
    Coordinate evolutionary optimization of policies for MuJoCo environments.
    """

    def __init__(self, config, initializer=None, mutation_fn=None, crossover_fn=None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        if mutation_fn is None:
            self.mutation_fn = gaussian_mutation
        else:
            self.mutation_fn = mutation_fn
        if crossover_fn is None:
            self.crossover_fn = self._resolve_crossover(config.crossover_type)
        else:
            self.crossover_fn = crossover_fn
        self.initializer = initializer

        env = gym.make(config.env_id)
        hidden_sizes = tuple(config.policy_hidden_sizes) if config.policy_hidden_sizes else (64, 64)

        activation = config.policy_activation
        if config.policy_final_init_std is None:
            final_init_std = 0.01
        else:
            final_init_std = config.policy_final_init_std

        self.template_policy = MLPActor.default_from_env(
            env,
            device=config.device,
            hidden_sizes=hidden_sizes,
            activation=activation,
            final_init_std=final_init_std,
        )
        env.close()

        self.population = Population(
            template_policy=self.template_policy,
            population_size=config.population_size,
            rng=self.rng,
            initializer=self.initializer,
        )
        self.population.initialize()

        if self.config.novelty_search:
            self._validate_novelty_config()

        self.eval_config = EvaluationConfig(
            env_id=config.env_id,
            episodes=config.evaluation_episodes,
            max_steps=config.max_steps_per_episode,
            unique_rollouts=config.unique_rollouts,
            terminate_on_truncation=config.terminate_on_truncation,
            num_workers=config.num_workers,
            worker_batch_size=config.worker_batch_size,
        )

    def policy_factory(self):
        return partial(
            MLPActor,
            obs_dim=self.template_policy.obs_dim,
            act_dim=self.template_policy.act_dim,
            hidden_sizes=tuple(self.template_policy.hidden_sizes),
            activation=self.template_policy.activation,
            final_init_std=self.template_policy.final_init_std,
            action_low=self.template_policy.action_low_t.cpu().numpy(),
            action_high=self.template_policy.action_high_t.cpu().numpy(),
            device=self.config.device,
        )

    def _validate_novelty_config(self):
        method = self.config.novelty_method.lower()
        if method not in {"nearest_neighbors", "diverse"}:
            raise ValueError(
                "novelty_method must be 'nearest_neighbors' or 'diverse'"
            )
        if not (0.0 <= self.config.novelty_weight <= 1.0):
            raise ValueError("novelty_weight must be between 0 and 1 inclusive")
        if method == "nearest_neighbors" and self.config.novelty_neighbors < 1:
            raise ValueError("novelty_neighbors must be >= 1")

    def _resolve_crossover(self, crossover_type):
        crossover_type = crossover_type.lower()
        if crossover_type == "uniform":
            return uniform_crossover
        if crossover_type == "index":
            return index_crossover
        if crossover_type == "blend":
            return partial(blend_crossover, alpha=0.5)
        raise ValueError(
            f"Unsupported crossover_type '{crossover_type}'. Available options are 'uniform', 'index', and 'blend'."
        )

    def _sample_mutation_sigma(self):
        variance = getattr(self.config, "mutation_sigma_variance", None)
        base_sigma = float(self.config.mutation_sigma)
        if not variance:
            return base_sigma

        scaled_sigma = base_sigma * float(variance)
        if scaled_sigma == base_sigma:
            return base_sigma

        lower, upper = sorted((base_sigma, scaled_sigma))
        span = upper - lower
        if span <= 0:
            return base_sigma

        scale = max(abs(base_sigma), np.finfo(np.float32).tiny)
        trunc_cdf = 1.0 - np.exp(-span / scale)
        sample = -scale * np.log(1.0 - self.rng.random() * trunc_cdf)
        return lower + sample

    def step(self, generation, breed_next=True):
        evaluation = evaluate_population(
            env_id=self.config.env_id,
            policy_factory=self.policy_factory(),
            genomes=self.population.genomes(),
            seed=self.config.seed + generation,
            config=self.eval_config,
            return_stats=self.config.novelty_search,
        )

        if self.config.novelty_search:
            fitnesses, metadata = evaluation
        else:
            fitnesses = evaluation
            metadata = None

        self.population.set_fitnesses(fitnesses, metadata)

        stats = {
            "generation": generation,
            "mean_fitness": float(np.mean(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
        }

        if self.config.novelty_search:
            self._apply_novelty_ranking()
            weighted = np.array([ind.fitness for ind in self.population.individuals])
            stats.update(
                {
                    "mean_weighted_fitness": float(np.mean(weighted)),
                    "max_weighted_fitness": float(np.max(weighted)),
                    "min_weighted_fitness": float(np.min(weighted)),
                }
            )

        if breed_next:
            elites = elitist_selection(self.population.individuals, self.config.elite_fraction)

            new_individuals = [elite.clone() for elite in elites]
            while len(new_individuals) < self.config.population_size:
                parent_pool = tournament_selection(
                    population=self.population.individuals,
                    tournament_size=self.config.tournament_size,
                    rng=self.rng,
                    winners=2,
                )
                parent_a, parent_b = parent_pool

                if self.rng.random() < self.config.crossover_rate:
                    child_genome = self.crossover_fn(parent_a.genome, parent_b.genome, self.rng)
                else:
                    child_genome = parent_a.genome.copy()

                mutation_sigma = self._sample_mutation_sigma()
                child_genome = self.mutation_fn(child_genome, mutation_sigma, self.rng)
                new_individuals.append(Individual(genome=child_genome))

            self.population.individuals = new_individuals
        return stats

    def _collect_novelty_features(self):
        features = []
        for ind in self.population.individuals:
            summary = ind.metadata.get("novelty_summary")
            if summary is None:
                return None
            feature_vec = _flatten_novelty_summary(summary)
            features.append(feature_vec)
            ind.metadata["novelty_features"] = feature_vec
        if not features:
            return None
        return np.vstack(features)

    def _apply_novelty_ranking(self):
        feature_matrix = self._collect_novelty_features()
        if feature_matrix is None or feature_matrix.shape[0] <= 1:
            return

        method = self.config.novelty_method.lower()
        if method == "nearest_neighbors":
            neighbors = min(self.config.novelty_neighbors, feature_matrix.shape[0] - 1)
            if neighbors < 1:
                return
            novelty_result = select_novel_rows(feature_matrix, K=neighbors, method="pdist")
            novelty_scores = novelty_result.get("scores")
            if novelty_scores is None:
                novelty_scores = np.zeros(feature_matrix.shape[0], dtype=np.float32)
        else:
            diverse_result = select_diverse_rows(
                feature_matrix,
                N=feature_matrix.shape[0],
                do_local_improve=False,
            )
            D = diverse_result.get("D")
            if D is None:
                novelty_scores = np.zeros(feature_matrix.shape[0], dtype=np.float32)
            else:
                novelty_scores = np.mean(D, axis=1)

        novelty_scores = np.asarray(novelty_scores, dtype=np.float32)
        novelty_scores = np.nan_to_num(novelty_scores, nan=0.0, posinf=0.0, neginf=0.0)
        raw_fitnesses = np.array([
            ind.raw_fitness if ind.raw_fitness is not None else 0.0
            for ind in self.population.individuals
        ], dtype=np.float32)
        raw_fitnesses = np.nan_to_num(raw_fitnesses, nan=0.0, posinf=0.0, neginf=0.0)
        fitness_ranks = rankdata(raw_fitnesses, method="average")
        novelty_ranks = rankdata(novelty_scores, method="average")
        weight = float(self.config.novelty_weight)
        weighted = (1.0 - weight) * fitness_ranks + weight * novelty_ranks

        for idx, ind in enumerate(self.population.individuals):
            ind.metadata["raw_fitness"] = ind.raw_fitness
            ind.metadata["novelty_score"] = float(novelty_scores[idx])
            ind.metadata["novelty_rank"] = float(novelty_ranks[idx])
            ind.metadata["fitness_rank"] = float(fitness_ranks[idx])
            ind.metadata["weighted_fitness"] = float(weighted[idx])
            ind.fitness = float(weighted[idx])
            ind.weighted_fitness = float(weighted[idx])

    def run(self, callback=None):
        """Run the evolutionary loop for the configured number of generations."""
        history = []
        generation_iterable = range(self.config.generations)
        progress_bar = None
        if tqdm is not None:
            progress_bar = tqdm(
                generation_iterable,
                desc="Generations",
                unit="gen",
                leave=True,
            )
            iterator = progress_bar
        else:
            iterator = generation_iterable

        for generation in iterator:
            start_time = perf_counter()
            breed_next = generation < self.config.generations - 1
            stats = self.step(generation, breed_next=breed_next)
            history.append(stats)
            elapsed = perf_counter() - start_time
            if progress_bar is not None:
                progress_bar.set_postfix({"last_gen_time": f"{elapsed:.2f}s"})
            if callback is not None:
                callback(stats, self.population)
        if progress_bar is not None:
            progress_bar.close()
        return self.population

    def export_best(self, output_path: str):
        """
        Save the best genome to ``.npz`` and return a state_dict for PyTorch fine-tuning.

        The returned tuple mirrors the on-disk artifact (``output_path``) and the in-memory
        weights that can be used with :func:`torch.save`.
        """
        best = self.population.best()
        if best is None:
            raise RuntimeError("Population has not been evaluated yet.")
        np.savez(output_path, genome=best.genome, fitness=best.fitness)
        state_dict = self.population.to_policy_state_dict(best.genome)
        return output_path, state_dict
