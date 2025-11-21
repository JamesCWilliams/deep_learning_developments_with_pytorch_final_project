import numpy as np


def gaussian_mutation(genome, sigma, rng):
    """Add elementwise Gaussian noise of scale ``sigma`` to ``genome``."""
    return genome + rng.normal(0.0, sigma, size=genome.shape).astype(np.float32)


def adaptive_gaussian_mutation(genome, sigma_schedule, step, rng):
    """
    Piecewise-constant sigma schedule for coarse annealing.
    Example: [0.1, 0.05, 0.02] -> use smaller noise as evolution progresses.
    """
    idx = min(step, len(sigma_schedule) - 1)
    sigma = sigma_schedule[idx]
    return gaussian_mutation(genome, sigma=sigma, rng=rng)


def uniform_crossover(parent_a, parent_b, rng):
    """Sample each gene from parent A or B with p=0.5."""
    mask = rng.random(size=parent_a.shape) < 0.5
    return np.where(mask, parent_a, parent_b).astype(np.float32)


def blend_crossover(parent_a, parent_b, alpha, rng):
    """
    Blended crossover (BLX-alpha).
    Samples uniformly from the extended interval around the parents.
    """
    lower = np.minimum(parent_a, parent_b)
    upper = np.maximum(parent_a, parent_b)
    diff = upper - lower
    min_bound = lower - alpha * diff
    max_bound = upper + alpha * diff
    return rng.uniform(min_bound, max_bound).astype(np.float32)


def index_crossover(parent_a, parent_b, rng):
    """
    Split the genome into hyper-rectangles anchored at the origin.

    For each axis, draw an index ``n`` uniformly from ``[0, dim]`` and copy
    values from ``parent_a`` when the coordinate along that axis is strictly
    less than ``n``. Genes outside that region are copied from ``parent_b``.
    """

    if parent_a.shape != parent_b.shape:
        raise ValueError("Parent genomes must share the same shape for crossover")

    if parent_a.ndim == 0:
        return parent_a.astype(np.float32) if rng.random() < 0.5 else parent_b.astype(np.float32)

    thresholds = [rng.integers(0, dim + 1) for dim in parent_a.shape]
    if not thresholds:
        return parent_a.copy().astype(np.float32)

    indices = np.indices(parent_a.shape)
    mask = np.ones(parent_a.shape, dtype=bool)
    for axis, threshold in enumerate(thresholds):
        mask &= indices[axis] < threshold

    return np.where(mask, parent_a, parent_b).astype(np.float32)


def elite_mutation(elites, sigma, per_elite, rng):
    """Yield mutated clones of elites for exploitation/exploration balance."""
    for elite in elites:
        for _ in range(per_elite):
            yield gaussian_mutation(elite, sigma=sigma, rng=rng)
