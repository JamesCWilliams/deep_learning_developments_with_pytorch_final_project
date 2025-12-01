#!/usr/bin/env python
import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

import cleanrl.ppo_continuous_action as ppo_mod
import wandb  # only used if tracking is enabled


# ---- Weight loader: generic PPO format ----


def load_ppo_hidden_layers_from_file(
    actor_mean: nn.Module, path: str, strict_shape: bool = True
):
    """
    Load first two hidden Linear layers for PPO from a generic file format:

    {
      "algo": "ppo",
      "hidden_layers": [
        {"weight": tensor, "bias": tensor},   # first Linear
        {"weight": tensor, "bias": tensor}    # second Linear
      ],
      "meta": {...optional...}
    }

    Only overwrites the first two nn.Linear modules in actor_mean.
    """
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PPO init file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if payload.get("algo") != "ppo":
        raise ValueError(f"Expected algo='ppo' in payload, got {payload.get('algo')!r}")

    layers = payload.get("hidden_layers", [])
    if len(layers) < 2:
        raise ValueError(f"PPO payload must have at least 2 hidden_layers, got {len(layers)}")

    linears = [m for m in actor_mean.modules() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        raise RuntimeError(f"actor_mean has only {len(linears)} Linear layers, need at least 2.")

    changes = []
    for idx in range(2):
        src = layers[idx]
        tgt = linears[idx]
        w = src["weight"]
        b = src["bias"]

        if strict_shape:
            if tgt.weight.shape != w.shape or tgt.bias.shape != b.shape:
                raise ValueError(
                    f"PPO layer {idx}: shape mismatch. "
                    f"target weight {tuple(tgt.weight.shape)}, file weight {tuple(w.shape)}; "
                    f"target bias {tuple(tgt.bias.shape)}, file bias {tuple(b.shape)}"
                )

        if tgt.weight.shape == w.shape:
            tgt.weight.data.copy_(w)
        if tgt.bias.shape == b.shape:
            tgt.bias.data.copy_(b)

        changes.append(
            {
                "layer_index": idx,
                "weight_shape": tuple(w.shape),
                "bias_shape": tuple(b.shape),
            }
        )

    print(f"[PPO] Loaded hidden layers from {path}: {changes}")
    return {"changed_layers": changes}


def save_checkpoint(path: str, agent: nn.Module, global_step: int):
    """Save a PPO checkpoint with agent parameters and step."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "global_step": global_step,
        },
        path,
    )
    print(f"[PPO] Saved checkpoint to {path}")


def log_model_artifact(
    path: str,
    run,          # wandb.Run or None
    run_name: str,
    env_id: str,
    seed: int,
    step,         # int or None
):
    """
    Log a model file as a W&B artifact if tracking is enabled.

    step: training step, or None for "final"
    """
    if run is None:
        return

    artifact_name = f"{run_name}-final" if step is None else f"{run_name}-step{step}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={
            "env_id": env_id,
            "seed": seed,
            "step": step,
        },
    )
    artifact.add_file(path)
    run.log_artifact(artifact)
    print(f"[W&B] Logged artifact {artifact_name} from {path}")


def evaluate_ppo_policy(
    agent: nn.Module,
    env_id: str,
    device: torch.device,
    gamma: float,
    episodes: int = 5,
    seed: int = 0,
):
    """
    Simple evaluation loop for PPO:
    - deterministic actions from current `agent`
    - no learning, just rollout
    """
    def make_single_env():
        return ppo_mod.make_env(env_id, 0, False, "eval", gamma)

    env = gym.vector.SyncVectorEnv([make_single_env()])
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    torch.as_tensor(obs, device=device)
                )
                action = action.cpu().numpy()
            obs, reward, terminations, truncations, infos = env.step(action)
            done = bool(terminations[0] or truncations[0])
            ep_return += float(reward[0])
        returns.append(ep_return)

    env.close()
    return returns


def main():
    # --- Pre-parse our extra flags ---
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--actor-init-file",
        type=str,
        default=None,
        help="Optional PPO pretrain file for first two hidden layers.",
    )
    pre_parser.add_argument(
        "--cuda",
        type=str,
        default=None,
        help="Set to False to force CPU even if CUDA is available.",
    )
    pre_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory to save logs/checkpoints. Defaults to runs/{run_name}.",
    )
    pre_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save data outputs. Defaults to data/{run_name}.",
    )

    extra_args, remaining_argv = pre_parser.parse_known_args()
    actor_init_file = extra_args.actor_init_file
    cuda_override = extra_args.cuda
    run_dir_override = extra_args.run_dir
    data_dir_override = extra_args.data_dir

    # --- CleanRL args via tyro ---
    Args = ppo_mod.Args
    args = tyro.cli(Args, args=remaining_argv)

    # CUDA override
    if cuda_override is not None:
        if isinstance(cuda_override, str):
            cuda_override = cuda_override.lower() in ("1", "true", "yes", "on")
        args.cuda = bool(cuda_override)

    # Derived quantities
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Run naming & dirs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if run_dir_override is not None:
        run_dir = run_dir_override
    else:
        run_dir = os.path.join("runs", run_name)

    if data_dir_override is not None:
        data_dir = data_dir_override
    else:
        data_dir = os.path.join("data", run_name)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # W&B
    run = None
    if args.track:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # TensorBoard
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [
            ppo_mod.make_env(
                args.env_id, i, args.capture_video, run_name, args.gamma
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = ppo_mod.Agent(envs).to(device)

    # Pretrained weights
    if actor_init_file is not None:
        load_ppo_hidden_layers_from_file(agent.actor_mean, actor_init_file)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    eval_interval = 10_000
    last_eval_step = 0

    ckpt_interval = 50_000
    last_ckpt_step = 0

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * (
                        (newvalue - b_returns[mb_inds]) ** 2
                    ).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        # Logging
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), global_step
        )
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

        # Periodic checkpoint
        if global_step - last_ckpt_step >= ckpt_interval:
            last_ckpt_step = global_step
            ckpt_name = f"model_step{global_step}.pt"

            ckpt_path_run = os.path.join(run_dir, ckpt_name)
            save_checkpoint(ckpt_path_run, agent, global_step)

            ckpt_path_data = os.path.join(data_dir, ckpt_name)
            save_checkpoint(ckpt_path_data, agent, global_step)

            log_model_artifact(
                path=ckpt_path_run,
                run=run,
                run_name=run_name,
                env_id=args.env_id,
                seed=args.seed,
                step=global_step,
            )
            if run is not None:
                wandb.save(ckpt_path_run)

        # Periodic eval
        if global_step - last_eval_step >= eval_interval:
            last_eval_step = global_step
            eval_returns = evaluate_ppo_policy(
                agent, args.env_id, device, args.gamma, episodes=3, seed=args.seed
            )
            mean_eval = float(np.mean(eval_returns))
            writer.add_scalar("eval/mean_return", mean_eval, global_step)
            if run is not None:
                wandb.log({"eval/mean_return": mean_eval})

    # Final checkpoint
    final_ckpt_name = "model_final.pt"

    final_ckpt_run = os.path.join(run_dir, final_ckpt_name)
    save_checkpoint(final_ckpt_run, agent, global_step=args.total_timesteps)

    final_ckpt_data = os.path.join(data_dir, final_ckpt_name)
    save_checkpoint(final_ckpt_data, agent, global_step=args.total_timesteps)

    log_model_artifact(
        path=final_ckpt_run,
        run=run,
        run_name=run_name,
        env_id=args.env_id,
        seed=args.seed,
        step=None,
    )
    if run is not None:
        wandb.save(final_ckpt_run)

    # Final eval
    eval_returns = evaluate_ppo_policy(
        agent, args.env_id, device, args.gamma, episodes=10, seed=args.seed
    )
    for idx, r in enumerate(eval_returns):
        writer.add_scalar("eval/episodic_return", r, idx)

    mean_eval_return = float(np.mean(eval_returns))
    writer.add_scalar("eval/mean_return_final", mean_eval_return, args.total_timesteps)
    if run is not None:
        wandb.log({"eval/mean_return_final": mean_eval_return})

    # Summary into data_dir
    summary_path = os.path.join(data_dir, "summary.npz")
    np.savez(
        summary_path,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        final_mean_eval_return=mean_eval_return,
    )

    envs.close()

    if run is not None and args.capture_video:
        import glob

        video_dir = f"videos/{run_name}"
        mp4s = glob.glob(os.path.join(video_dir, "*.mp4"))
        if mp4s:
            latest_mp4 = max(mp4s, key=os.path.getmtime)
            wandb.log(
                {
                    "rollout_video": wandb.Video(
                        latest_mp4, fps=30, format="mp4"
                    )
                }
            )

    writer.close()


if __name__ == "__main__":
    main()
