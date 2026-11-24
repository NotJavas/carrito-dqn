"""Entorno Gymnasium y visualización Pygame del carrito."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from config import ACTION_NAMES, CarConfig, TrackConfig
from simulator_core import CarSimulator


class CarTrackEnv(gym.Env):
    """Pista 2D con nueve acciones discretas y observación vectorial."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        track_config: Optional[TrackConfig] = None,
        car_config: Optional[CarConfig] = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode debe ser None, 'human' o 'rgb_array'")

        self.render_mode = render_mode
        self.track_config = track_config or TrackConfig()
        self.car_config = car_config or CarConfig()
        self.simulator = CarSimulator(self.track_config, self.car_config)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=np.asarray([0, 0, 0, 0, 0, 0, -1, -1, -1], dtype=np.float32),
            high=np.ones(9, dtype=np.float32),
            dtype=np.float32,
        )
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self.closed = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.simulator.seed(seed)
        options = options or {}
        random_start = bool(options.get("random_start", True))
        observation = self.simulator.reset(random_start=random_start)
        info = {"action_name": ACTION_NAMES[0], "success": 0.0, "collision": 0.0}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.simulator.step(int(action))
        info["action_name"] = ACTION_NAMES[int(action)]
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _draw_scene(self, canvas: pygame.Surface) -> None:
        cfg = self.track_config
        canvas.fill((76, 144, 88))
        outer_rect = pygame.Rect(
            cfg.center_x - cfg.outer_a,
            cfg.center_y - cfg.outer_b,
            2 * cfg.outer_a,
            2 * cfg.outer_b,
        )
        inner_rect = pygame.Rect(
            cfg.center_x - cfg.inner_a,
            cfg.center_y - cfg.inner_b,
            2 * cfg.inner_a,
            2 * cfg.inner_b,
        )
        pygame.draw.ellipse(canvas, (78, 84, 91), outer_rect)
        pygame.draw.ellipse(canvas, (76, 144, 88), inner_rect)
        pygame.draw.ellipse(canvas, (236, 236, 236), outer_rect, width=4)
        pygame.draw.ellipse(canvas, (236, 236, 236), inner_rect, width=4)

        mid_rect = pygame.Rect(
            cfg.center_x - self.simulator.mid_a,
            cfg.center_y - self.simulator.mid_b,
            2 * self.simulator.mid_a,
            2 * self.simulator.mid_b,
        )
        pygame.draw.ellipse(canvas, (221, 207, 92), mid_rect, width=2)

        state = self.simulator.state
        start = (int(state.x), int(state.y))
        for end_x, end_y, distance in self.simulator.sensor_endpoints():
            fraction = distance / cfg.sensor_range
            color = (246, int(90 + 120 * fraction), 70)
            pygame.draw.line(canvas, color, start, (int(end_x), int(end_y)), width=2)
            pygame.draw.circle(canvas, color, (int(end_x), int(end_y)), 3)

        front = (
            state.x + math.cos(state.heading) * 18,
            state.y + math.sin(state.heading) * 18,
        )
        left = (
            state.x + math.cos(state.heading + 2.45) * 13,
            state.y + math.sin(state.heading + 2.45) * 13,
        )
        right = (
            state.x + math.cos(state.heading - 2.45) * 13,
            state.y + math.sin(state.heading - 2.45) * 13,
        )
        pygame.draw.polygon(canvas, (34, 116, 210), [front, left, right])
        pygame.draw.polygon(canvas, (10, 40, 80), [front, left, right], width=2)

        font = pygame.font.Font(None, 25)
        lines = [
            f"Acción: {ACTION_NAMES[self.simulator.last_action]}",
            f"Velocidad: {state.speed:5.1f}",
            f"Vuelta: {max(self.simulator.total_progress / (2 * math.pi), 0):.2f}",
            f"Recompensa: {self.simulator.last_reward:+.3f}",
        ]
        for index, text in enumerate(lines):
            surface = font.render(text, True, (255, 255, 255))
            canvas.blit(surface, (18, 18 + index * 25))

    def render(self):
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode(
                    (self.track_config.width, self.track_config.height)
                )
                pygame.display.set_caption("Carrito DQN en pista elíptica")
            else:
                self._screen = pygame.Surface(
                    (self.track_config.width, self.track_config.height)
                )
        if self._clock is None:
            self._clock = pygame.time.Clock()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.closed = True

        canvas = pygame.Surface((self.track_config.width, self.track_config.height))
        self._draw_scene(canvas)

        if self.render_mode == "human":
            self._screen.blit(canvas, canvas.get_rect())
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        frame = np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))
        return frame

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None
        self.closed = True

