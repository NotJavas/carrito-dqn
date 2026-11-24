"""Controlador por reglas para comprobar que la pista es resoluble.

No aprende. Su función pedagógica es separar la calidad del entorno de la
calidad del algoritmo y ofrecer una segunda línea base además del azar.
"""

import numpy as np

from car_env import CarTrackEnv


ACTION_FROM_CONTROL = {
    (0, 0): 0,
    (1, 0): 1,
    (-1, 0): 2,
    (0, -1): 3,
    (0, 1): 4,
    (1, -1): 5,
    (1, 1): 6,
    (-1, -1): 7,
    (-1, 1): 8,
}


def heuristic_action(observation) -> int:
    far_left, left, front, right, far_right, speed, _, _, lateral = observation
    steering = 0
    if front < 0.38:
        steering = -1 if max(left, far_left) > max(right, far_right) else 1
    else:
        opening_difference = (right + far_right) - (left + far_left)
        if opening_difference > 0.16:
            steering = 1
        elif opening_difference < -0.16:
            steering = -1
        elif lateral > 0.35:
            steering = -1
        elif lateral < -0.35:
            steering = 1

    if speed > 0.72 and (front < 0.65 or steering != 0):
        throttle = -1
    elif speed < 0.60:
        throttle = 1
    else:
        throttle = 0
    return ACTION_FROM_CONTROL[(throttle, steering)]


def main(episodes: int = 10) -> None:
    env = CarTrackEnv(render_mode=None)
    rewards, successes = [], []
    for episode in range(episodes):
        observation, _ = env.reset(
            seed=500 + episode, options={"random_start": False}
        )
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            action = heuristic_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        successes.append(info.get("success", 0.0))
    env.close()
    print(
        {
            "episodes": episodes,
            "reward_mean": float(np.mean(rewards)),
            "success_rate": float(np.mean(successes)),
        }
    )


if __name__ == "__main__":
    main()

