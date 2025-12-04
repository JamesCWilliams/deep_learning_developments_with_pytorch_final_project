#!/usr/bin/env python
import argparse
import csv
import os
import tempfile

import gymnasium as gym
import numpy as np
import torch
import wandb

import cleanrl.ppo_continuous_action as ppo_mod
import cleanrl.sac_continuous_action as sac_mod


# ---------- Helper: max episode steps ----------

def get_max_episode_steps(env_id: str):
    env = gym.make(env_id)
    max_steps = getattr(env, "_max_episode_steps", None)
    if max_steps is None and env.spec is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    env.close()
    return max_steps


# ---------- Artifact helpers ----------

def find_final_model_artifact(run, artifact_file="model_final.pt"):
    """
    Given a wandb.Run, find the 'final' model artifact containing artifact_file.
    Assumes training logged a final artifact with step=None and that it contains model_final.pt.
    """
    final_candidate = None

    # First pass: look for artifacts with metadata.step is None = "final"
    for art in run.logged_artifacts():
        if art.type != "model":
            continue
        try:
            step = art.metadata.get("step", None)
        except Exception:
            step = None
        if step is None:
            final_candidate = art
            break

    # Fallback: any model artifact containing artifact_file
    if final_candidate is None:
        for art in run.logged_artifacts():
            if art.type != "model":
                continue
            try:
                files = list(art.files())
            except Exception:
                continue
            for f in files:
                if f.name.endswith(artifact_file):
                    final_candidate = art
                    break
            if final_candidate is not None:
                break

    if final_candidate is None:
        return None

    has_file = any(f.name.endswith(artifact_file) for f in final_candidate.files())
    return final_candidate if has_file else None


def download_ckpt_from_artifact(artifact, artifact_file="model_final.pt"):
    tmpdir = tempfile.mkdtemp(prefix="wandb_eval_")
    local_dir = artifact.download(root=tmpdir)
    ckpt_path = os.path.join(local_dir, artifact_file)
    if not os.path.exists(ckpt_path):
        for root, _, files in os.walk(local_dir):
            for fn in files:
                if fn.endswith(".pt"):
                    return os.path.join(root, fn)
        raise FileNotFoundError(f"Could not find {artifact_file} in {local_dir}")
    return ckpt_path


# ---------- PPO loading + eval ----------

def load_ppo_agent_from_ckpt(ckpt_path, env_id, gamma, device):
    def make_single_env():
        return ppo_mod.make_env(env_id, 0, False, "eval", gamma)

    envs = gym.vector.SyncVectorEnv([make_single_env()])
    agent = ppo_mod.Agent(envs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("agent", ckpt)
    agent.load_state_dict(state)
    agent.eval()
    envs.close()
    return agent


def evaluate_ppo_agent(agent, env_id, gamma, device, episodes=10, seed=0):
    max_steps = get_max_episode_steps(env_id)

    def make_single_env():
        return ppo_mod.make_env(env_id, 0, False, "eval", gamma)

    env = gym.vector.SyncVectorEnv([make_single_env()])
    results = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
                action, _, _, _ = agent.get_action_and_value(obs_t)
                action_np = action.cpu().numpy()
            obs, reward, terminations, truncations, infos = env.step(action_np)
            done = bool(terminations[0] or truncations[0])
            ep_return += float(reward[0])
            ep_len += 1

        if max_steps is not None and max_steps > 0:
            duration_rate = ep_len / max_steps
        else:
            duration_rate = np.nan

        results.append(
            {
                "episodic_return": ep_return,
                "episode_length": ep_len,
                "max_episode_steps": max_steps,
                "duration_rate": duration_rate,
            }
        )

    env.close()
    return results


# ---------- SAC loading + eval ----------

def load_sac_actor_from_ckpt(ckpt_path, env_id, device):
    def make_single_env():
        return sac_mod.make_env(env_id, 0, 0, False, "eval")

    envs = gym.vector.SyncVectorEnv([make_single_env()])
    actor = sac_mod.Actor(envs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("actor", ckpt)
    actor.load_state_dict(state)
    actor.eval()
    envs.close()
    return actor


def evaluate_sac_actor(actor, env_id, device, episodes=10, seed=0):
    max_steps = get_max_episode_steps(env_id)

    def make_single_env():
        return sac_mod.make_env(env_id, seed, 0, False, "eval")

    env = gym.vector.SyncVectorEnv([make_single_env()])
    results = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
                action, _, _ = actor.get_action(obs_t)
                action_np = action.cpu().numpy()
            obs, reward, terminations, truncations, infos = env.step(action_np)
            done = bool(terminations[0] or truncations[0])
            ep_return += float(reward[0])
            ep_len += 1

        if max_steps is not None and max_steps > 0:
            duration_rate = ep_len / max_steps
        else:
            duration_rate = np.nan

        results.append(
            {
                "episodic_return": ep_return,
                "episode_length": ep_len,
                "max_episode_steps": max_steps,
                "duration_rate": duration_rate,
            }
        )

    env.close()
    return results


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="wandb entity/user")
    parser.add_argument("--project", required=True, help="wandb project (ONE env+config)")
    parser.add_argument("--algo", required=True, choices=["ppo", "sac"])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--env-id-filter", type=str, default=None,
                        help="Optional env_id filter (usually unnecessary if project is env-specific)")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Optional cap on number of runs to evaluate")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="")
    fieldnames = [
        "project",
        "algo",
        "run_id",
        "run_name",
        "env_id",
        "seed",
        "exp_name",
        "total_timesteps",
        "episode_index",
        "episodic_return",
        "episode_length",
        "max_episode_steps",
        "duration_rate",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    evaluated = 0

    for run in runs:
        cfg = dict(run.config)
        env_id = cfg.get("env_id") or cfg.get("env")
        seed = cfg.get("seed")
        exp_name = cfg.get("exp_name", "")
        total_timesteps = cfg.get("total_timesteps", None)
        gamma = cfg.get("gamma", 0.99)

        if env_id is None or seed is None:
            print(f"[SKIP] run {run.id} missing env_id or seed")
            continue

        if args.env_id_filter is not None and env_id != args.env_id_filter:
            continue

        print(f"[INFO] Evaluating run {run.id} ({run.name}), env={env_id}, seed={seed}")

        artifact = find_final_model_artifact(run)
        if artifact is None:
            print(f"[WARN] No final model artifact for run {run.id}, skipping.")
            continue

        try:
            ckpt_path = download_ckpt_from_artifact(artifact, artifact_file="model_final.pt")
        except Exception as e:
            print(f"[WARN] Failed to download checkpoint for run {run.id}: {e}")
            continue

        if args.algo == "ppo":
            agent = load_ppo_agent_from_ckpt(ckpt_path, env_id, gamma, device)
            ep_results = evaluate_ppo_agent(
                agent, env_id, gamma, device, episodes=args.episodes, seed=seed
            )
        else:  # sac
            actor = load_sac_actor_from_ckpt(ckpt_path, env_id, device)
            ep_results = evaluate_sac_actor(
                actor, env_id, device, episodes=args.episodes, seed=seed
            )

        for ep_idx, ep in enumerate(ep_results):
            writer.writerow(
                {
                    "project": args.project,
                    "algo": args.algo,
                    "run_id": run.id,
                    "run_name": run.name,
                    "env_id": env_id,
                    "seed": seed,
                    "exp_name": exp_name,
                    "total_timesteps": total_timesteps,
                    "episode_index": ep_idx,
                    "episodic_return": ep["episodic_return"],
                    "episode_length": ep["episode_length"],
                    "max_episode_steps": ep["max_episode_steps"],
                    "duration_rate": ep["duration_rate"],
                }
            )

        evaluated += 1
        if args.max_runs is not None and evaluated >= args.max_runs:
            break

    csv_file.close()
    print(f"[DONE] Evaluated {evaluated} runs from project '{args.project}'.")
    print(f"[DONE] Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
