import atexit
import math
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

import gymnasium as gym
import numpy as np

default_env_maker = lambda env_id: gym.make(env_id)

class EvaluationConfig:
    def __init__(
        self,
        env_id,
        episodes,
        max_steps,
        unique_rollouts=True,
        terminate_on_truncation=False,
        num_workers=1,
        worker_batch_size=None,
    ):
        self.env_id = env_id
        self.episodes = episodes
        self.max_steps = max_steps
        self.unique_rollouts = unique_rollouts
        self.terminate_on_truncation = terminate_on_truncation
        self.num_workers = num_workers
        self.worker_batch_size = worker_batch_size


NOVELTY_QUANTILES = (0.25, 0.5, 0.75)


def _summarise_rollout_statistics(observations, actions, policy):
    """Compute summary statistics used for novelty search."""
    obs_dim = getattr(policy, "obs_dim", observations.shape[1] if observations.size else 0)
    act_dim = getattr(policy, "act_dim", actions.shape[1] if actions.size else 0)

    def _ensure_array(arr, dim):
        if arr.size == 0:
            return np.zeros(dim, dtype=np.float32)
        return arr.astype(np.float32, copy=False)

    obs = _ensure_array(observations, obs_dim)
    acts = _ensure_array(actions, act_dim)

    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if acts.ndim == 1:
        acts = acts.reshape(1, -1)

    if obs.size:
        obs_mean = np.mean(obs, axis=0)
        obs_quantiles = np.quantile(obs, NOVELTY_QUANTILES, axis=0)
    else:
        obs_mean = np.zeros(obs_dim, dtype=np.float32)
        obs_quantiles = np.zeros((len(NOVELTY_QUANTILES), obs_dim), dtype=np.float32)

    if acts.size:
        act_mean = np.mean(acts, axis=0)
        act_quantiles = np.quantile(acts, NOVELTY_QUANTILES, axis=0)
    else:
        act_mean = np.zeros(act_dim, dtype=np.float32)
        act_quantiles = np.zeros((len(NOVELTY_QUANTILES), act_dim), dtype=np.float32)

    return {
        "observation_mean": obs_mean.astype(np.float32, copy=False),
        "observation_quantiles": obs_quantiles.astype(np.float32, copy=False),
        "action_mean": act_mean.astype(np.float32, copy=False),
        "action_quantiles": act_quantiles.astype(np.float32, copy=False),
        "quantiles": NOVELTY_QUANTILES,
        "total_steps": int(max(len(obs), len(acts))),
    }


def _evaluate_in_env(env, policy, genome, seed, config, collect_stats=False):
    """Evaluate the genome using an existing environment instance."""

    policy.set_parameters_flat(genome)
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()

    episodes = config.episodes
    if episodes <= 0:
        return 0.0

    unique_rollouts = config.unique_rollouts
    max_steps = config.max_steps
    terminate_on_truncation = config.terminate_on_truncation

    policy_act = policy.act
    env_reset = env.reset
    env_step = env.step

    cumulative_return = 0.0

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []

    for ep in range(episodes):
        rollout_seed = seed + ep if unique_rollouts else seed
        obs, _ = env_reset(seed=rollout_seed)
        terminated = False
        truncated = False
        steps = 0
        episode_return = 0.0
        while not (terminated or truncated):
            if collect_stats:
                observations.append(np.asarray(obs, dtype=np.float32))
            action = policy_act(obs)
            if collect_stats:
                actions.append(np.asarray(action, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env_step(action)
            episode_return += float(reward)
            steps += 1
            if max_steps and steps >= max_steps:
                if terminate_on_truncation:
                    terminated = True
                truncated = True
                break
        cumulative_return += episode_return

    mean_return = float(cumulative_return / episodes)

    if not collect_stats:
        return mean_return

    if observations:
        obs_arr = np.stack(observations, axis=0)
    else:
        obs_arr = np.empty((0, getattr(policy, "obs_dim", 0)), dtype=np.float32)
    if actions:
        act_arr = np.stack(actions, axis=0)
    else:
        act_arr = np.empty((0, getattr(policy, "act_dim", 0)), dtype=np.float32)

    summary = _summarise_rollout_statistics(obs_arr, act_arr, policy)
    return mean_return, summary


def evaluate_individual(env_fn, policy, genome, seed, config, close_env=True):
    """Evaluate a genome by averaging episodic returns."""
    env = env_fn()
    try:
        return _evaluate_in_env(env, policy, genome, seed, config)
    finally:
        if close_env:
            env.close()


_WORKER_ENV = None
_WORKER_POLICY_FACTORY = None
_WORKER_POLICY = None
_WORKER_CONFIG = None
_WORKER_COLLECT_STATS = False


def _shutdown_worker_env():
    global _WORKER_ENV
    env = _WORKER_ENV
    if env is not None:
        try:
            env.close()
        finally:
            _WORKER_ENV = None


def _worker_initializer(env_id, policy_factory, config, collect_stats):
    """Initialize per-process state for parallel evaluation."""
    global _WORKER_ENV, _WORKER_POLICY_FACTORY, _WORKER_POLICY, _WORKER_CONFIG
    if _WORKER_ENV is None:
        _WORKER_ENV = gym.make(env_id)
        atexit.register(_shutdown_worker_env)
    _WORKER_POLICY_FACTORY = policy_factory
    if _WORKER_POLICY is None:
        _WORKER_POLICY = policy_factory()
    _WORKER_CONFIG = config
    global _WORKER_COLLECT_STATS
    _WORKER_COLLECT_STATS = collect_stats


def _evaluate_worker_batch(tasks):
    if (
        _WORKER_ENV is None
        or _WORKER_POLICY_FACTORY is None
        or _WORKER_CONFIG is None
        or _WORKER_POLICY is None
    ):
        raise RuntimeError("Worker not properly initialised")
    policy = _WORKER_POLICY
    env = _WORKER_ENV
    collect_stats = _WORKER_COLLECT_STATS
    results: List[Tuple[float, Optional[dict]]]
    if collect_stats:
        results = []
    else:
        results_float: List[float] = []
    evaluate = _evaluate_in_env
    for genome, env_seed in tasks:
        if collect_stats:
            fit, summary = evaluate(
                env,
                policy,
                genome,
                env_seed,
                _WORKER_CONFIG,
                collect_stats=True,
            )
            results.append((fit, {"novelty_summary": summary}))
        else:
            fit = evaluate(env, policy, genome, env_seed, _WORKER_CONFIG)
            results_float.append(float(fit))
    return results if collect_stats else results_float


def evaluate_population(env_id, policy_factory, genomes, seed, config, return_stats=False):
    """
    Sequentially evaluate each genome.
    For reproducibility we create fresh envs per evaluation and optionally unique seeds.
    """
    genome_list: List[np.ndarray] = list(genomes)
    if not genome_list:
        return []

    max_workers = (
        min(config.num_workers, len(genome_list)) if config.num_workers else 0
    )
    if max_workers <= 1:
        env = gym.make(env_id)
        policy = policy_factory()
        try:
            returns = np.empty(len(genome_list), dtype=np.float32)
            metadata: List[Optional[dict]] = [None] * len(genome_list)
            evaluate = _evaluate_in_env
            for idx, genome in enumerate(genome_list):
                env_seed = seed + idx if config.unique_rollouts else seed
                if return_stats:
                    fit, summary = evaluate(
                        env,
                        policy,
                        genome,
                        env_seed,
                        config,
                        collect_stats=True,
                    )
                    returns[idx] = fit
                    metadata[idx] = {"novelty_summary": summary}
                else:
                    returns[idx] = evaluate(env, policy, genome, env_seed, config)
        finally:
            env.close()
        if return_stats:
            return returns.tolist(), metadata
        return returns.tolist()

    if config.worker_batch_size:
        chunk_size = max(1, min(config.worker_batch_size, len(genome_list)))
    else:
        target_chunks = max_workers * 4
        chunk_size = max(1, int(math.ceil(len(genome_list) / target_chunks)))
    chunk_size = min(chunk_size, len(genome_list))

    if config.unique_rollouts:
        seeds = [seed + idx for idx in range(len(genome_list))]
    else:
        seeds = [seed for _ in genome_list]

    tasks: List[Tuple[np.ndarray, int]] = list(zip(genome_list, seeds))

    executor_kwargs = {
        "max_workers": max_workers,
        "initializer": _worker_initializer,
        "initargs": (env_id, policy_factory, config, return_stats),
    }

    try:
        executor_kwargs["mp_context"] = get_context("spawn")
    except ValueError:
        # Some platforms may not support changing the context; fall back to default.
        pass

    try:
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            batched_tasks = [
                tasks[idx : idx + chunk_size]
                for idx in range(0, len(tasks), chunk_size)
            ]
            results = executor.map(_evaluate_worker_batch, batched_tasks)
            flattened_returns: List[float] = []
            metadata: List[Optional[dict]] = []
            for batch in results:
                if return_stats:
                    for fit, meta in batch:  # type: ignore[misc]
                        flattened_returns.append(float(fit))
                        metadata.append(meta)
                else:
                    flattened_returns.extend(batch)  # type: ignore[arg-type]
            if return_stats:
                return flattened_returns, metadata
            return flattened_returns
    except (BrokenProcessPool, RuntimeError):
        # Fallback to sequential execution if the worker pool fails to start
        env = gym.make(env_id)
        policy = policy_factory()
        try:
            evaluate = _evaluate_in_env
            returns = np.empty(len(genome_list), dtype=np.float32)
            metadata: List[Optional[dict]] = [None] * len(genome_list)
            for idx, (genome, env_seed) in enumerate(tasks):
                if return_stats:
                    fit, summary = evaluate(
                        env,
                        policy,
                        genome,
                        env_seed,
                        config,
                        collect_stats=True,
                    )
                    returns[idx] = fit
                    metadata[idx] = {"novelty_summary": summary}
                else:
                    returns[idx] = evaluate(env, policy, genome, env_seed, config)
            if return_stats:
                return returns.tolist(), metadata
            return returns.tolist()
        finally:
            env.close()
