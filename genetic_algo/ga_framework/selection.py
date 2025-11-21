import numpy as np

from .population import Individual


def elitist_selection(population, fraction):
    """Return the top `fraction` of individuals (at least one)."""
    population = list(population)
    assert 0 < fraction <= 1.0
    k = max(1, int(len(population) * fraction))
    return sorted(population, key=lambda ind: ind.fitness or -np.inf, reverse=True)[:k]


def tournament_selection(population, tournament_size, rng, winners):
    """Sample `winners` individuals via tournament selection."""
    selected = []
    for _ in range(winners):
        competitors = rng.choice(population, size=tournament_size, replace=False)
        winner = max(competitors, key=lambda ind: ind.fitness or -np.inf)
        selected.append(winner)
    return selected


def fitness_proportionate_selection(population, winners, rng):
    """Roulette-wheel selection proportionate to positive-shifted fitness."""
    fitnesses = np.array([ind.fitness for ind in population], dtype=np.float32)
    min_fit = fitnesses.min()
    if not np.isfinite(min_fit):
        raise ValueError("Population fitness contains non-finite values")
    if (fitnesses <= 0).all():
        # shift by constant so that all weights are positive
        fitnesses = fitnesses - min_fit + 1e-6
    probs = fitnesses / fitnesses.sum()
    idx = rng.choice(len(population), size=winners, replace=True, p=probs)
    return [population[i] for i in idx]
