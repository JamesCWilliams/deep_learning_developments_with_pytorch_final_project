# models.py
import math
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    """Simple orthogonal init; use std=0.01 for final layers."""
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _resolve_activation(activation):
    """Turn a string or module class into a torch activation class."""

    if isinstance(activation, str):
        activation_key = activation.lower()
        activation_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "softplus": nn.Softplus,
            "identity": nn.Identity,
        }
        if activation_key not in activation_map:
            raise ValueError(f"Unsupported activation '{activation}'.")
        return activation_map[activation_key]

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation

    raise TypeError(
        "Activation must be a string identifier or an nn.Module subclass."
    )


class MLPActor(nn.Module):
    """Compact policy network with handy genome helpers."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
        final_init_std=0.01,
        action_low=None,
        action_high=None,
        device="cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = list(hidden_sizes)
        self.activation_cls = _resolve_activation(activation)
        self.activation = self.activation_cls  # Backwards compatibility for factories
        self.activation_name = (
            activation
            if isinstance(activation, str)
            else self.activation_cls.__name__
        )
        self.final_init_std = final_init_std
        if action_low is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
        if action_high is None:
            action_high = np.ones(act_dim, dtype=np.float32)
        self.register_buffer("action_low_t", torch.tensor(action_low))
        self.register_buffer("action_high_t", torch.tensor(action_high))

        layers: List[nn.Module] = []
        prev = obs_dim
        for h in self.hidden_sizes:
            layers += [layer_init(nn.Linear(prev, h)), self.activation_cls()]
            prev = h
        # Output layer = mean; your specified std=0.01 init
        layers += [layer_init(nn.Linear(prev, act_dim), std=final_init_std)]
        self.actor_mean = nn.Sequential(*layers)
        self.to(device)

    @torch.no_grad()
    def forward(self, obs):
        """Return the unbounded mean action for ``obs``; squashing happens in :meth:`act`."""
        return self.actor_mean(obs)

    @torch.no_grad()
    def act(self, obs):
        """Map observations to clipped actions (single or batched)."""
        single = False
        if obs.ndim == 1:
            obs = obs[None, :]
            single = True
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.actor_mean[0].weight.device)
        mean = self.forward(obs_t)
        # tanh to [-1,1], then scale to [low, high]
        squashed = torch.tanh(mean)
        low, high = self.action_low_t, self.action_high_t
        scaled = low + 0.5 * (squashed + 1.0) * (high - low)
        scaled = torch.clamp(scaled, min=low, max=high)
        out = scaled.cpu().numpy()
        return out[0] if single else out

    # ======= Flat parameter helpers (great for GA genomes) =======

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_parameters_flat(self):
        with torch.no_grad():
            return torch.cat([p.view(-1).cpu() for p in self.parameters()]).numpy()

    def set_parameters_flat(self, flat):
        assert flat.ndim == 1, "Flat parameter vector must be 1-D"
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                new_vals = torch.from_numpy(flat[idx:idx+n]).view_as(p).to(p.device, dtype=p.dtype)
                p.copy_(new_vals)
                idx += n
        assert idx == flat.size, "Flat vector size did not match network parameters"

    @staticmethod
    def default_from_env(env, device="cpu", hidden_sizes=(64, 64), activation=nn.Tanh, final_init_std=0.01):
        obs_dim = int(np.array(env.observation_space.shape).prod())
        act_dim = int(np.array(env.action_space.shape).prod())
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        return MLPActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            final_init_std=final_init_std,
            action_low=low,
            action_high=high,
            device=device,
        )
