"""Parámetros centrales del simulador y del entrenamiento.

Los estudiantes pueden modificar este archivo para realizar experimentos
controlados sin alterar la lógica del entorno ni del agente.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackConfig:
    width: int = 1000
    height: int = 700
    center_x: float = 500.0
    center_y: float = 350.0
    outer_a: float = 430.0
    outer_b: float = 285.0
    inner_a: float = 255.0
    inner_b: float = 125.0
    sensor_range: float = 175.0
    sensor_step: float = 4.0
    target_laps: float = 1.0
    max_steps: int = 1800


@dataclass(frozen=True)
class CarConfig:
    radius: float = 12.0
    max_speed: float = 155.0
    acceleration: float = 105.0
    braking: float = 150.0
    rolling_friction: float = 28.0
    turn_rate: float = 2.35
    dt: float = 1.0 / 30.0


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 5000
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 50_000
    learning_starts: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 45_000
    tau: float = 0.005
    gradient_clip: float = 10.0
    seed: int = 42
    success_window: int = 100
    early_stop_rate: float = 0.99


ACTION_NAMES = {
    0: "inercia",
    1: "acelerar",
    2: "frenar",
    3: "izquierda",
    4: "derecha",
    5: "acelerar + izquierda",
    6: "acelerar + derecha",
    7: "frenar + izquierda",
    8: "frenar + derecha",
}


# Cada acción discreta combina los cuatro controles primitivos del problema.
# El primer valor es aceleración/freno (-1, 0, +1) y el segundo es dirección.
ACTION_TO_CONTROL = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (-1.0, 0.0),
    3: (0.0, -1.0),
    4: (0.0, 1.0),
    5: (1.0, -1.0),
    6: (1.0, 1.0),
    7: (-1.0, -1.0),
    8: (-1.0, 1.0),
}

