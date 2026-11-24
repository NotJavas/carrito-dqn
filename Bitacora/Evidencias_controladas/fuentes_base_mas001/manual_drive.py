"""Permite conducir con W, S, A y D para entender el espacio de acciones."""

import pygame

from car_env import CarTrackEnv


def keys_to_action() -> int:
    keys = pygame.key.get_pressed()
    accelerate = keys[pygame.K_w] or keys[pygame.K_UP]
    brake = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

    if accelerate and left:
        return 5
    if accelerate and right:
        return 6
    if brake and left:
        return 7
    if brake and right:
        return 8
    if accelerate:
        return 1
    if brake:
        return 2
    if left:
        return 3
    if right:
        return 4
    return 0


def main() -> None:
    env = CarTrackEnv(render_mode="human")
    env.reset(seed=42, options={"random_start": False})
    print("Controles: W acelerar, S frenar, A izquierda, D derecha, ESC salir.")
    done = False
    while not env.closed:
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            break
        action = keys_to_action()
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Episodio terminado:", info)
            pygame.time.wait(450)
            env.reset(options={"random_start": False})
    env.close()


if __name__ == "__main__":
    main()

