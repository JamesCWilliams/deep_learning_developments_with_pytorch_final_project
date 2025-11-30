#!/usr/bin/env python
import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
import cleanrl.td3_continuous_action as td3_mod


# ---- TD3 actor loader: GA format (fc1/fc2/fc_mu) ----

def load_td3_actor_from_file(actor: nn.Module, path: str):
    """
    Expect payload like:

      {
        "fc1.weight": ...,
        "fc1.bias": ...,
        "fc2.weight": ...,
        "fc2.bias": ...,
        "fc_mu.weight": ...,
        "fc_mu.bias": ...,
      }

    Partially loads these into CleanRL's TD3 Actor.
    Buffers (action_scale, action_bias) and everything else remain untouched.
    """
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TD3 init file not found: {path}")

    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(f"TD3 init file must contain a dict, got {type(sd)}")

    model_sd = actor.state_dict()
    matched = {}

    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            matched[k] = v
        else:
            print(
                f"[TD3] Skipping key {k}: not found or shape mismatch "
                f"(model: {model_sd.get(k, None) and tuple(model_sd.get(k, torch.empty(0)).shape)}, "
                f"file: {tuple(v.shape)})"
            )

    model_sd.update(matched)
    actor.load_state_dict(model_sd, strict=False)
    print(f"[TD3] Loaded actor weights from {path}: {sorted(matched.keys())}")
    return {"matched_keys": sorted(matched.keys())}


def main():
    # --- Pre-parse our custom flags: actor-init-file + cuda override ---
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--actor-init-file",
        type=str,
        default=None,
        help="Optional TD3 actor init file (GA format: fc1/fc2/fc_mu).",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default=None,
        help="Set to False to force CPU even if CUDA is available.",
    )
    extra, remaining_argv = parser.parse_known_args()
    actor_init_file = extra.actor_init_file
    cuda_override = extra.cuda

    # --- Use CleanRL's Args class with tyro for the rest ---
    Args = td3_mod.Args
    args = tyro.cli(Args, args=remaining_argv)

    # Apply CUDA override if user passed --cuda
    if cuda_override is not None:
        if isinstance(cuda_override, str):
            cuda_override = cuda_override.lower() in ("1", "true", "yes", "on")
        args.cuda = bool(cuda_override)

    # ---- From here down, this is your CleanRL TD3 script with a tiny hook ----
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            td3_mod.make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor = td3_mod.Actor(envs).to(device)
    qf1 = td3_mod.QNetwork(envs).to(device)
    qf2 = td3_mod.QNetwork(envs).to(device)
    qf1_target = td3_mod.QNetwork(envs).to(device)
    qf2_target = td3_mod.QNetwork(envs).to(device)
    target_actor = td3_mod.Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

    # --- OUR HOOK: load GA actor weights if provided, then sync target ---
    if actor_init_file is not None:
        load_td3_actor_from_file(actor, actor_init_file)
        target_actor.load_state_dict(actor.state_dict())

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # action logic
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(
                    0, actor.action_scale * args.exploration_noise
                )
                actions = actions.cpu().numpy().clip(
                    envs.single_action_space.low, envs.single_action_space.high
                )

        # step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # episodic logging
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )
                    break

        # replay buffer; handle truncations
        real_next_obs = next_obs.copy()

        final_obs = infos.get("final_observation", None)
        if final_obs is not None:
            for idx, trunc in enumerate(truncations):
                if trunc and final_obs[idx] is not None:
                    real_next_obs[idx] = final_obs[idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device)
                    * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(
                    envs.single_action_space.low[0],
                    envs.single_action_space.high[0],
                )
                qf1_next_target = qf1_target(
                    data.next_observations, next_state_actions
                )
                qf2_next_target = qf2_target(
                    data.next_observations, next_state_actions
                )
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = (
                    data.rewards.flatten()
                    + (1 - data.dones.flatten())
                    * args.gamma
                    * (min_qf_next_target).view(-1)
                )

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize critic
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(
                    data.observations, actor(data.observations)
                ).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update targets
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data
                        + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data
                        + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data
                        + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf1_loss", qf1_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_loss", qf2_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/qf_loss", qf_loss.item() / 2.0, global_step
                )
                writer.add_scalar(
                    "losses/actor_loss", actor_loss.item(), global_step
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    # model saving / eval (unchanged)
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            (actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path
        )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            td3_mod.make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(td3_mod.Actor, td3_mod.QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "TD3",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()

    if args.track and args.capture_video:
        import glob
        import wandb

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
