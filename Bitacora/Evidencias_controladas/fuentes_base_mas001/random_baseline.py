"""Línea base: un agente que elige acciones al azar."""

import numpy as np

from car_env import CarTrackEnv


def main(episodes: int = 30) -> None:
    env = CarTrackEnv(render_mode=None)
    rewards, successes, collisions = [], [], []
    for episode in range(episodes):
        _, _ = env.reset(seed=900 + episode)
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        successes.append(info.get("success", 0.0))
        collisions.append(info.get("collision", 0.0))
    env.close()
    print(
        {
            "episodes": episodes,
            "reward_mean": float(np.mean(rewards)),
            "success_rate": float(np.mean(successes)),
            "collision_rate": float(np.mean(collisions)),
        }
    )


if __name__ == "__main__":
    main()

