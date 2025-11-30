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
import cleanrl.sac_continuous_action as sac_mod


# ---- Loader: SAC actor weights from GA (fc1/fc2/fc_mean), ReLU activations ----

def load_sac_actor_from_file(actor: nn.Module, path: str):
    """
    GA saves SAC actor as a state-dict-like mapping:

      {
        "fc1.weight": ...,
        "fc1.bias": ...,
        "fc2.weight": ...,
        "fc2.bias": ...,
        "fc_mean.weight": ...,
        "fc_mean.bias": ...,
      }

    We partially load these into CleanRL's Actor.
    fc_logstd & buffers remain untouched.
    """
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SAC init file not found: {path}")

    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(f"SAC init file must contain a dict, got {type(sd)}")

    model_sd = actor.state_dict()
    matched = {}

    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            matched[k] = v
        else:
            print(
                f"[SAC] Skipping key {k}: not found or shape mismatch "
                f"(model: {model_sd.get(k, None) and tuple(model_sd[k].shape)}, file: {tuple(v.shape)})"
            )

    model_sd.update(matched)
    actor.load_state_dict(model_sd, strict=False)

    print(f"[SAC] Loaded actor weights from {path}: {sorted(matched.keys())}")
    return {"matched_keys": sorted(matched.keys())}


def main():
    # ---- Pre-parse our extra flags ----
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--actor-init-file",
        type=str,
        default=None,
        help="Optional SAC actor state-dict file (fc1/fc2/fc_mean).",
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

    Args = sac_mod.Args
    args = tyro.cli(Args, args=remaining_argv)

    # apply CUDA override if passed
    if cuda_override is not None:
        if isinstance(cuda_override, str):
            cuda_override = cuda_override.lower() in ("1", "true", "yes", "on")
        args.cuda = bool(cuda_override)

    # ---- From here down, mostly verbatim from cleanrl.sac_continuous_action ----
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
            sac_mod.make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = sac_mod.Actor(envs).to(device)
    qf1 = sac_mod.SoftQNetwork(envs).to(device)
    qf2 = sac_mod.SoftQNetwork(envs).to(device)
    qf1_target = sac_mod.SoftQNetwork(envs).to(device)
    qf2_target = sac_mod.SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # ---- OUR HOOK: load GA actor weights if provided ----
    if actor_init_file is not None:
        load_sac_actor_from_file(actor, actor_init_file)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

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
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # episodic logs
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

        # replay buffer
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
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
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

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar(
                    "losses/qf_loss", qf_loss.item() / 2.0, global_step
                )
                writer.add_scalar(
                    "losses/actor_loss", actor_loss.item(), global_step
                )
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
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
