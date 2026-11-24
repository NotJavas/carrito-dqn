"""Diagnóstico rápido de instalación y contrato Gymnasium."""

import platform

import gymnasium
from gymnasium.utils.env_checker import check_env
import numpy
import pygame
import torch

from car_env import CarTrackEnv


def main() -> None:
    print(
        {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "pygame": pygame.version.ver,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
        }
    )
    env = CarTrackEnv(render_mode=None)
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=42)
    assert observation.shape == (9,)
    for _ in range(100):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()

    render_env = CarTrackEnv(render_mode="rgb_array")
    render_env.reset(seed=42)
    frame = render_env.render()
    assert frame.shape == (700, 1000, 3)
    render_env.close()
    print("Entorno correcto: espacios, transiciones y render fueron validados.")


if __name__ == "__main__":
    main()

