"""Train five independent sigmoid TD3 agents in the Python swarm environment."""

import argparse
from pathlib import Path

import numpy as np
import torch

from swarm_env import ACTION_LOWER, OBSERVATION_SIZE, UAV_COUNT, SwarmEnv
from td3 import IndependentTD3


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.total = 0
        self.observations = np.empty((capacity, UAV_COUNT, OBSERVATION_SIZE), np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, UAV_COUNT, 3), np.float32)
        self.rewards = np.empty((capacity, UAV_COUNT), np.float32)

    def add(self, obs, actions, rewards, next_obs):
        index = self.total % self.capacity
        self.observations[index] = obs
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.next_observations[index] = next_obs
        self.total += 1

    def sample(self, agent, batch_size, rng, device):
        indices = rng.choice(min(self.total, self.capacity), batch_size, replace=False)
        data = (
            self.observations[indices, agent],
            self.actions[indices, agent],
            self.rewards[indices, agent, None],
            self.next_observations[indices, agent],
            # The episode ends at a time limit, so keep the bootstrap term.
            np.zeros((batch_size, 1), dtype=np.float32),
        )
        return tuple(torch.as_tensor(values, device=device) for values in data)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--steps", type=positive_int, default=2000)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--warmup", type=positive_int, default=256)
    parser.add_argument("--buffer-size", type=positive_int, default=200_000)
    parser.add_argument("--update-every", type=positive_int, default=1)
    parser.add_argument("--policy-delay", type=positive_int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/td3_swarm.pt"))
    args = parser.parse_args()
    if args.batch_size > args.buffer_size:
        parser.error("--batch-size must fit in the replay buffer")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = SwarmEnv(args.steps)
    agents = [IndependentTD3(OBSERVATION_SIZE, device, args.policy_delay) for _ in range(UAV_COUNT)]
    replay = ReplayBuffer(args.buffer_size)
    exploration = 0.15

    for episode in range(1, args.episodes + 1):
        observations = env.reset()
        returns = np.zeros(UAV_COUNT)
        for _ in range(args.steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observations, device=device)
                actions = torch.stack([agent.actor(obs_tensor[i]) for i, agent in enumerate(agents)])
                actions = actions.cpu().numpy()
            actions = np.clip(actions + rng.normal(0, exploration, actions.shape), ACTION_LOWER, 1)
            exploration = max(0.01, exploration * (1 - 1e-6))
            next_obs, rewards, _ = env.step(actions)
            replay.add(observations, actions, rewards, next_obs)
            observations = next_obs
            returns += rewards
            if (replay.total >= max(args.warmup, args.batch_size)
                    and replay.total % args.update_every == 0):
                for index, agent in enumerate(agents):
                    agent.update(replay.sample(index, args.batch_size, rng, device))

        print(f"episode={episode} steps={replay.total} "
              f"mean_return={returns.mean():.4f} updates_per_agent={agents[0].grad_updates}",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actors": [agent.actor.state_dict() for agent in agents],
        "critics": [(agent.critic1.state_dict(), agent.critic2.state_dict()) for agent in agents],
        "action_order": ("ACS", "PF", "VF"),
        "observation_size": OBSERVATION_SIZE,
        "episodes": args.episodes,
        "steps": replay.total,
        "updates_per_agent": agents[0].grad_updates,
    }, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
