"""Red neuronal DQN, memoria de repetición y actualización de Bellman."""

from __future__ import annotations

from collections import deque
import math
import random
from typing import Deque, Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import TrainConfig


class ReplayBuffer:
    """Guarda transiciones y devuelve minilotes aleatorios."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.data: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.data.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int, device: torch.device):
        batch = self.rng.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )


class QNetwork(nn.Module):
    """Red pequeña: nueve entradas, dos capas ocultas y nueve valores Q."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class DQNAgent:
    """Agente con política epsilon-greedy, red objetivo y pérdida Huber."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.action_size = action_size
        self.device = device
        self.policy_net = QNetwork(
            observation_size, action_size, config.hidden_size
        ).to(device)
        self.target_net = QNetwork(
            observation_size, action_size, config.hidden_size
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=config.learning_rate, amsgrad=True
        )
        self.replay = ReplayBuffer(config.replay_capacity, config.seed)
        self.total_steps = 0

    @property
    def epsilon(self) -> float:
        fraction = math.exp(-self.total_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * fraction

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(self.policy_net(state).argmax(dim=1).item())
        if training:
            self.total_steps += 1
        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self) -> Dict[str, float]:
        if len(self.replay) < max(
            self.config.batch_size, self.config.learning_starts
        ):
            return {}

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size, self.device
        )
        predicted_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.soft_update()

        return {
            "loss": float(loss.item()),
            "q_mean": float(predicted_q.mean().item()),
            "target_mean": float(target_q.mean().item()),
        }

    def soft_update(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_parameter, policy_parameter in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_parameter.mul_(1.0 - tau)
                target_parameter.add_(tau * policy_parameter)

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = int(checkpoint.get("total_steps", 0))

