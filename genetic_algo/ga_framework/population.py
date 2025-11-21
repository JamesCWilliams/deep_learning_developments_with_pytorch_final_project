import numpy as np

from models import MLPActor


class Individual:
    """Container for a genome and its bookkeeping."""

    def __init__(self, genome, fitness=None, episodes=0, metadata=None, raw_fitness=None, weighted_fitness=None):
        self.genome = genome
        self.fitness = fitness
        self.episodes = episodes
        self.metadata = metadata or {}
        self.raw_fitness = raw_fitness
        self.weighted_fitness = weighted_fitness

    def clone(self):
        return Individual(
            self.genome.copy(),
            self.fitness,
            self.episodes,
            dict(self.metadata),
            self.raw_fitness,
            self.weighted_fitness,
        )


class Population:
    """Population of policy genomes backed by a shared policy template."""

    def __init__(self, template_policy, population_size, rng, initializer=None):
        self.template_policy = template_policy
        self.population_size = population_size
        self.rng = rng
        self.initializer = initializer or _default_initializer
        self.individuals = []

    def initialize(self):
        self.individuals = []
        for _ in range(self.population_size):
            genome = self.initializer(self.template_policy, self.rng)
            self.individuals.append(Individual(genome=genome))

    def topk(self, k):
        scored = [ind for ind in self.individuals if ind.fitness is not None]
        scored.sort(key=lambda ind: ind.fitness, reverse=True)
        return scored[:k]

    def best(self):
        scored = self.topk(1)
        return scored[0] if scored else None

    def genomes(self):
        for ind in self.individuals:
            yield ind.genome

    def set_fitnesses(self, values, metadata=None):
        values_list = list(values)
        if metadata is None:
            metadata_list = [None] * len(values_list)
        else:
            metadata_list = list(metadata)
            if len(metadata_list) < len(values_list):
                metadata_list.extend([None] * (len(values_list) - len(metadata_list)))
        for ind, fitness, meta in zip(self.individuals, values_list, metadata_list):
            fit_val = float(fitness)
            ind.fitness = fit_val
            ind.raw_fitness = fit_val
            ind.weighted_fitness = None
            if meta:
                ind.metadata.update(meta)

    def to_policy_state_dict(self, genome):
        policy = MLPActor(
            obs_dim=self.template_policy.obs_dim,
            act_dim=self.template_policy.act_dim,
            hidden_sizes=self.template_policy.hidden_sizes,
            activation=self.template_policy.activation,
            final_init_std=self.template_policy.final_init_std,
            action_low=self.template_policy.action_low_t.cpu().numpy(),
            action_high=self.template_policy.action_high_t.cpu().numpy(),
            device=str(next(self.template_policy.parameters()).device),
        )
        policy.set_parameters_flat(genome)
        return {k: v.cpu() for k, v in policy.state_dict().items()}


def _default_initializer(policy, rng):
    scale = 0.05
    return rng.standard_normal(policy.num_params()).astype(np.float32) * scale
