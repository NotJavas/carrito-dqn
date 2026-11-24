"""Carga un modelo entrenado y observa su conducción sin exploración aleatoria."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from car_env import CarTrackEnv
from config import TrainConfig
from dqn_agent import DQNAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación visual del carrito DQN")
    parser.add_argument("--model", type=Path, default=Path("results/best_model.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"No se encontró {args.model}. Ejecuta primero: python train.py"
        )

    device = torch.device("cpu")  # CPU fue más rápida en la prueba de este proyecto.
    render_mode = None if args.no_render else "human"
    env = CarTrackEnv(render_mode=render_mode)
    observation, _ = env.reset(seed=1000, options={"random_start": False})
    agent = DQNAgent(
        observation_size=len(observation),
        action_size=env.action_space.n,
        config=TrainConfig(),
        device=device,
    )
    agent.load(str(args.model))

    results = []
    for episode in range(args.episodes):
        observation, _ = env.reset(
            seed=1000 + episode, options={"random_start": False}
        )
        total_reward = 0.0
        done = False
        info = {}
        while not done and not env.closed:
            action = agent.select_action(observation, training=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append(
            {
                "reward": total_reward,
                "success": info.get("success", 0.0),
                "collision": info.get("collision", 0.0),
                "steps": info.get("steps", 0.0),
            }
        )
        if env.closed:
            break
        print(f"Episodio {episode + 1}: {results[-1]}")

    env.close()
    if results:
        print(
            "Resumen:",
            {
                "reward_mean": float(np.mean([row["reward"] for row in results])),
                "success_rate": float(np.mean([row["success"] for row in results])),
                "collision_rate": float(np.mean([row["collision"] for row in results])),
            },
        )


if __name__ == "__main__":
    main()

