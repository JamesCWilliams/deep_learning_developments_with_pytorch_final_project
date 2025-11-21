"""Utility for evaluating GA-trained policies saved as PyTorch state_dict files."""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from models import MLPActor


def evaluate_policy(env_id, policy, episodes=5, max_steps=1000, seed=0, render=False, terminate_on_trunc=False):
    """
    Roll out ``policy`` for ``episodes`` episodes and return the mean episodic return.

    The helper accepts standard Gymnasium arguments so it can be reused in notebooks or
    scripts where additional logging or rendering control is helpful.
    """

    env = gym.make(env_id, render_mode="human" if render else None)
    env.reset(seed=seed)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            steps += 1
            if max_steps and steps >= max_steps:
                if terminate_on_trunc:
                    terminated = True
                truncated = True
                break
        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns))


def load_policy_from_state_dict(env_id, state_dict_path, device="cpu"):
    """
    Instantiate :class:`MLPActor` for ``env_id`` and load weights from ``state_dict_path``.

    ``state_dict_path`` should point to a file saved with ``torch.save``. The helper mirrors
    the logic used during training so that exported genomes can be evaluated without
    re-specifying architecture details.
    """

    env = gym.make(env_id)
    policy = MLPActor.default_from_env(env, device=device)
    env.close()

    state_dict = torch.load(state_dict_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy



def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_dict", type=Path, help="Path to torch-saved state_dict file")
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--terminate_on_trunc", action="store_true")
    args = parser.parse_args(argv)

    policy = load_policy_from_state_dict(args.env_id, args.state_dict, device=args.device)

    avg_return = evaluate_policy(
        env_id=args.env_id,
        policy=policy,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render,
        terminate_on_trunc=args.terminate_on_trunc,
    )
    print(f"Average return over {args.episodes} episodes: {avg_return:.2f}")


if __name__ == "__main__":
    main()
