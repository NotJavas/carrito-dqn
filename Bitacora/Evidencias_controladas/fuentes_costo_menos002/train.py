"""Entrena el carrito con Deep Q Learning y registra evidencia reproducible."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from car_env import CarTrackEnv
from config import ACTION_NAMES, TrackConfig, TrainConfig
from dqn_agent import DQNAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento DQN del carrito")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-laps", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_curves(logs: pd.DataFrame, output_path: Path, train_config: TrainConfig) -> None:
    rolling_reward = logs["reward"].rolling(train_config.success_window, min_periods=1).mean()
    rolling_success = logs["success"].rolling(train_config.success_window, min_periods=1).mean()
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(logs["episode"], logs["reward"], alpha=0.25, color="#2F80ED")
    axes[0].plot(logs["episode"], rolling_reward, color="#0B2545", linewidth=2)
    axes[0].set_ylabel("Retorno por episodio")
    axes[0].grid(alpha=0.2)
    axes[1].plot(logs["episode"], rolling_success, color="#00A6A6", linewidth=2)
    axes[1].axhline(
        train_config.early_stop_rate,
        color="#E76F51",
        linestyle="--",
        label=f"Meta {train_config.early_stop_rate * 100:g} %",
    )
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel(f"Éxito móvil ({train_config.success_window} episodios)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.suptitle("Evolución del aprendizaje del carrito")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_without_render(
    agent: DQNAgent,
    track_config: TrackConfig,
    episodes: int = 10,
    seed: int = 10_000,
) -> dict:
    env = CarTrackEnv(render_mode=None, track_config=track_config)
    returns, successes, collisions = [], [], []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode, options={"random_start": False})
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            action = agent.select_action(observation, training=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
        successes.append(info.get("success", 0.0))
        collisions.append(info.get("collision", 0.0))
    env.close()
    return {
        "evaluation_reward_mean": float(np.mean(returns)),
        "evaluation_success_rate": float(np.mean(successes)),
        "evaluation_collision_rate": float(np.mean(collisions)),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_config = replace(TrainConfig(), episodes=args.episodes, seed=args.seed)
    track_config = replace(
        TrackConfig(), target_laps=args.target_laps, max_steps=args.max_steps
    )
    set_seeds(train_config.seed)

    device = torch.device("cpu")  # CPU fue más rápida en la prueba de este proyecto.
    env = CarTrackEnv(render_mode=None, track_config=track_config)
    observation, _ = env.reset(seed=train_config.seed)
    agent = DQNAgent(
        observation_size=len(observation),
        action_size=env.action_space.n,
        config=train_config,
        device=device,
    )

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(args.output / "tensorboard"))
    except Exception:
        writer = None

    recent_success = deque(maxlen=train_config.success_window)
    recent_rewards = deque(maxlen=train_config.success_window)
    best_moving_reward = -float("inf")
    rows = []

    print(f"Dispositivo: {device} | Observaciones: {len(observation)} | Acciones: {env.action_space.n}")
    for episode in range(1, train_config.episodes + 1):
        observation, _ = env.reset(seed=train_config.seed + episode)
        total_reward = 0.0
        losses = []
        action_counts = {action: 0 for action in ACTION_NAMES}
        done = False
        info = {}

        while not done:
            action = agent.select_action(observation, training=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, reward, next_observation, done)
            diagnostics = agent.train_step()
            if diagnostics:
                losses.append(diagnostics["loss"])
            observation = next_observation
            total_reward += reward
            action_counts[action] += 1

        success = int(info.get("success", 0.0) > 0.5)
        collision = int(info.get("collision", 0.0) > 0.5)
        recent_success.append(success)
        recent_rewards.append(total_reward)
        row = {
            "episode": episode,
            "reward": total_reward,
            "steps": int(info.get("steps", 0)),
            "success": success,
            "collision": collision,
            "laps": info.get("laps", 0.0),
            "epsilon": agent.epsilon,
            "loss_mean": float(mean(losses)) if losses else np.nan,
            "speed_final": info.get("speed", 0.0),
        }
        row.update({f"action_{action}": count for action, count in action_counts.items()})
        rows.append(row)

        moving_reward = mean(recent_rewards)
        moving_success = mean(recent_success)
        if moving_reward > best_moving_reward and len(recent_rewards) == recent_rewards.maxlen:
            best_moving_reward = moving_reward
            agent.save(str(args.output / "best_model.pt"))

        if writer is not None:
            writer.add_scalar("train/reward", total_reward, episode)
            writer.add_scalar(f"train/success_rate_{train_config.success_window}", moving_success, episode)
            writer.add_scalar("train/epsilon", agent.epsilon, episode)
            if losses:
                writer.add_scalar("train/loss", mean(losses), episode)

        if episode == 1 or episode % 20 == 0:
            print(
                f"Episodio {episode:4d} | retorno {total_reward:8.2f} | "
                f"éxito móvil {moving_success:5.1%} | epsilon {agent.epsilon:.3f}"
            )

        if (
            len(recent_success) == recent_success.maxlen
            and moving_success >= train_config.early_stop_rate
        ):
            print(
                f"Criterio de parada alcanzado en el episodio {episode}: "
                f"{sum(recent_success)}/{len(recent_success)} éxitos "
                f"(meta {train_config.early_stop_rate * 100:g} %)."
            )
            break

    env.close()
    if writer is not None:
        writer.close()

    agent.save(str(args.output / "final_model.pt"))
    logs = pd.DataFrame(rows)
    logs.to_csv(args.output / "training_log.csv", index=False)
    save_curves(logs, args.output / "training_curves.png", train_config)

    evaluation = evaluate_without_render(agent, track_config)
    pd.DataFrame([evaluation]).to_csv(args.output / "evaluation_summary.csv", index=False)
    print("Evaluación final:", evaluation)
    print(f"Resultados guardados en: {args.output.resolve()}")


if __name__ == "__main__":
    main()

