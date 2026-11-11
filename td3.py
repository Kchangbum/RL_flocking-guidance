"""Independent sigmoid TD3 adapted from the local PyTorch benchmark."""

import torch
from torch import nn
import torch.nn.functional as F


LOWER_BOUNDS = (0.1, 0.0, 0.2)


class Actor(nn.Module):
    """FC(64,128,64,3) actor with independent sigmoid outputs.

    Matches the inspected MATLAB R2024a actor ending in sigmoidLayer.
    """

    def __init__(self, obs_dim: int, action_dim: int = 3):
        super().__init__()
        if action_dim != 3:
            raise ValueError("Expected three ACS/PF/VF weights.")
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs))


class Critic(nn.Module):
    """obs:64, action:32, concat:256->256->1, matching described layers."""

    def __init__(self, obs_dim: int, action_dim: int = 3):
        super().__init__()
        self.obs_path = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU())
        self.act_path = nn.Sequential(nn.Linear(action_dim, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(96, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.obs_path(obs), self.act_path(action)), dim=-1)
        return self.head(combined)


class IndependentTD3:
    def __init__(self, obs_dim: int, device: torch.device, policy_delay: int):
        self.device = device
        self.action_lower = torch.tensor(LOWER_BOUNDS, device=device, dtype=torch.float32)
        self.actor = Actor(obs_dim).to(device)
        self.critic1 = Critic(obs_dim).to(device)
        self.critic2 = Critic(obs_dim).to(device)
        self.actor_target = Actor(obs_dim).to(device)
        self.critic1_target = Critic(obs_dim).to(device)
        self.critic2_target = Critic(obs_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=2e-4)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=2e-4)
        self.policy_delay = policy_delay
        self.grad_updates = 0

    @staticmethod
    @torch.no_grad()
    def polyak(source: nn.Module, target: nn.Module, tau: float = 0.005) -> None:
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.lerp_(p, tau)

    def update(self, batch: tuple[torch.Tensor, ...]) -> None:
        obs, action, reward, next_obs, done = batch
        with torch.no_grad():
            # Independent sigmoid actions: no sum-to-one normalization.
            # Target policy smoothing must respect MATLAB action-spec bounds.
            target_action = self.actor_target(next_obs)
            noise = torch.randn_like(target_action).mul_(0.2).clamp_(-0.5, 0.5)
            target_action = torch.maximum(target_action + noise, self.action_lower).clamp_max_(1.0)
            next_q = torch.minimum(
                self.critic1_target(next_obs, target_action),
                self.critic2_target(next_obs, target_action),
            )
            target_q = reward + 0.99 * (1.0 - done) * next_q

        for critic, optimizer in (
            (self.critic1, self.critic1_optim),
            (self.critic2, self.critic2_optim),
        ):
            loss = F.mse_loss(critic(obs, action), target_q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            optimizer.step()

        self.grad_updates += 1
        if self.grad_updates % self.policy_delay == 0:
            # Actor gradient should pass through critic Q, but critic parameters
            # themselves should not be updated by the actor optimizer.
            for p in self.critic1.parameters():
                p.requires_grad_(False)
            actor_loss = -self.critic1(obs, torch.maximum(self.actor(obs), self.action_lower)).mean()
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optim.step()
            for p in self.critic1.parameters():
                p.requires_grad_(True)
            self.polyak(self.actor, self.actor_target)
            self.polyak(self.critic1, self.critic1_target)
            self.polyak(self.critic2, self.critic2_target)
