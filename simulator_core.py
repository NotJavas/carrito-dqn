"""Núcleo matemático del carrito sin dependencias gráficas.

La pista es una corona elíptica. El agente observa cinco sensores de distancia,
su velocidad, la orientación respecto de la tangente y la posición lateral.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Tuple

import numpy as np

from config import ACTION_TO_CONTROL, CarConfig, TrackConfig


SENSOR_ANGLES_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)


def wrap_angle(angle: float) -> float:
    """Lleva un ángulo al intervalo [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class CarState:
    x: float
    y: float
    heading: float
    speed: float


class CarSimulator:
    """Simulador determinista que contiene la dinámica y la recompensa."""

    def __init__(
        self,
        track: TrackConfig | None = None,
        car: CarConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.track = track or TrackConfig()
        self.car = car or CarConfig()
        self.rng = np.random.default_rng(seed)
        self.state = CarState(0.0, 0.0, 0.0, 0.0)
        self.previous_theta = 0.0
        self.total_progress = 0.0
        self.steps = 0
        self.last_reward = 0.0
        self.last_action = 0
        self.reset()

    @property
    def mid_a(self) -> float:
        return (self.track.outer_a + self.track.inner_a) / 2.0

    @property
    def mid_b(self) -> float:
        return (self.track.outer_b + self.track.inner_b) / 2.0

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, random_start: bool = True) -> np.ndarray:
        """Coloca el auto en la parte inferior de la pista y devuelve el estado."""
        theta = math.pi / 2.0
        if random_start:
            theta += float(self.rng.uniform(-0.08, 0.08))

        x = self.track.center_x + self.mid_a * math.cos(theta)
        y = self.track.center_y + self.mid_b * math.sin(theta)
        tangent = self.tangent_heading(theta)
        heading_noise = float(self.rng.uniform(-0.04, 0.04)) if random_start else 0.0
        self.state = CarState(x=x, y=y, heading=tangent + heading_noise, speed=0.0)
        self.previous_theta = theta
        self.total_progress = 0.0
        self.steps = 0
        self.last_reward = 0.0
        self.last_action = 0
        return self.observation()

    def is_on_track(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Comprueba que un punto esté dentro de la corona elíptica."""
        dx = x - self.track.center_x
        dy = y - self.track.center_y
        outer_a = max(self.track.outer_a - margin, 1.0)
        outer_b = max(self.track.outer_b - margin, 1.0)
        inner_a = self.track.inner_a + margin
        inner_b = self.track.inner_b + margin
        inside_outer = (dx / outer_a) ** 2 + (dy / outer_b) ** 2 <= 1.0
        outside_inner = (dx / inner_a) ** 2 + (dy / inner_b) ** 2 >= 1.0
        return bool(inside_outer and outside_inner)

    def track_theta(self, x: float, y: float) -> float:
        """Estima la posición angular sobre la línea media de la pista."""
        dx = (x - self.track.center_x) / self.mid_a
        dy = (y - self.track.center_y) / self.mid_b
        return math.atan2(dy, dx)

    def tangent_heading(self, theta: float) -> float:
        """Calcula la dirección tangente para el sentido de avance elegido."""
        tx = -self.mid_a * math.sin(theta)
        ty = self.mid_b * math.cos(theta)
        return math.atan2(ty, tx)

    def lateral_position(self) -> float:
        """Devuelve -1 junto al borde interior y +1 junto al exterior."""
        dx = self.state.x - self.track.center_x
        dy = self.state.y - self.track.center_y
        phi = math.atan2(dy, dx)
        c2 = math.cos(phi) ** 2
        s2 = math.sin(phi) ** 2
        outer_radius = 1.0 / math.sqrt(c2 / self.track.outer_a**2 + s2 / self.track.outer_b**2)
        inner_radius = 1.0 / math.sqrt(c2 / self.track.inner_a**2 + s2 / self.track.inner_b**2)
        actual_radius = math.hypot(dx, dy)
        ratio = (actual_radius - inner_radius) / max(outer_radius - inner_radius, 1e-6)
        return float(np.clip(2.0 * ratio - 1.0, -1.0, 1.0))

    def ray_distance(self, relative_angle_deg: float) -> float:
        """Avanza un rayo hasta tocar un borde y devuelve la distancia."""
        angle = self.state.heading + math.radians(relative_angle_deg)
        distance = 0.0
        while distance < self.track.sensor_range:
            x = self.state.x + math.cos(angle) * distance
            y = self.state.y + math.sin(angle) * distance
            if not self.is_on_track(x, y):
                break
            distance += self.track.sensor_step
        return min(distance, self.track.sensor_range)

    def sensor_distances(self) -> np.ndarray:
        values = [self.ray_distance(angle) for angle in SENSOR_ANGLES_DEG]
        return np.asarray(values, dtype=np.float32)

    def sensor_endpoints(self) -> Iterable[Tuple[float, float, float]]:
        distances = self.sensor_distances()
        for relative_angle, distance in zip(SENSOR_ANGLES_DEG, distances):
            angle = self.state.heading + math.radians(relative_angle)
            yield (
                self.state.x + math.cos(angle) * float(distance),
                self.state.y + math.sin(angle) * float(distance),
                float(distance),
            )

    def observation(self) -> np.ndarray:
        """Construye el vector que recibe la red neuronal."""
        sensors = self.sensor_distances() / self.track.sensor_range
        speed = np.asarray([self.state.speed / self.car.max_speed], dtype=np.float32)
        theta = self.track_theta(self.state.x, self.state.y)
        heading_error = wrap_angle(self.state.heading - self.tangent_heading(theta))
        orientation = np.asarray(
            [math.sin(heading_error), math.cos(heading_error)], dtype=np.float32
        )
        lateral = np.asarray([self.lateral_position()], dtype=np.float32)
        return np.concatenate([sensors, speed, orientation, lateral]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        """Aplica una acción y devuelve la transición estilo Gymnasium."""
        if action not in ACTION_TO_CONTROL:
            raise ValueError(f"Acción inválida: {action}")

        throttle, steering = ACTION_TO_CONTROL[action]
        dt = self.car.dt

        if throttle > 0:
            self.state.speed += self.car.acceleration * dt
        elif throttle < 0:
            self.state.speed -= self.car.braking * dt
        else:
            self.state.speed -= self.car.rolling_friction * dt
        self.state.speed = float(np.clip(self.state.speed, 0.0, self.car.max_speed))

        speed_factor = 0.30 + 0.70 * (self.state.speed / self.car.max_speed)
        self.state.heading = wrap_angle(
            self.state.heading + steering * self.car.turn_rate * speed_factor * dt
        )
        self.state.x += math.cos(self.state.heading) * self.state.speed * dt
        self.state.y += math.sin(self.state.heading) * self.state.speed * dt

        theta = self.track_theta(self.state.x, self.state.y)
        delta_theta = wrap_angle(theta - self.previous_theta)
        self.previous_theta = theta
        self.total_progress += delta_theta
        self.steps += 1
        self.last_action = action

        tangent = self.tangent_heading(theta)
        alignment = math.cos(wrap_angle(self.state.heading - tangent))
        progress_reward = 18.0 * delta_theta
        speed_reward = 0.035 * (self.state.speed / self.car.max_speed) * max(alignment, 0.0)
        control_cost = 0.004 * abs(steering)
        reward = progress_reward + speed_reward - 0.02 - control_cost

        collision = not self.is_on_track(
            self.state.x, self.state.y, margin=self.car.radius
        )
        lap = self.total_progress >= 2.0 * math.pi * self.track.target_laps
        terminated = bool(collision or lap)
        truncated = bool(self.steps >= self.track.max_steps and not terminated)

        if collision:
            reward -= 25.0
        if lap:
            reward += 35.0

        self.last_reward = float(reward)
        info = {
            "collision": float(collision),
            "success": float(lap),
            "laps": max(self.total_progress / (2.0 * math.pi), 0.0),
            "progress": self.total_progress,
            "alignment": alignment,
            "speed": self.state.speed,
            "steps": float(self.steps),
        }
        return self.observation(), float(reward), terminated, truncated, info