import pygame
import pymunk
import pymunk.pygame_util

from robotPygame import Robot
from envRLppo import EnvRL

import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from functools import partial

from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining * initial_value < 0.0003:
            return 0.0003
        return progress_remaining * initial_value

    return func


def train_model(
    dim, n_envs, n_timesteps, model_name, learning_rate=0.0003, gamma=0.99, n_steps=2500
):
    print(
        "Tensorboard command: python3 -m tensorboard.main --logdir=2d/pygame/PPOrandom/tensorboard/{}/".format(
            model_name
        )
    )
    env = make_vec_env(partial(EnvRL, dim), n_envs=n_envs)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        n_steps=n_steps,
        learning_rate=learning_rate,
        batch_size=n_steps * n_envs,
        tensorboard_log="2d/pygame/PPOrandom/tensorboard/{}".format(model_name),
        gamma=gamma,
    )

    model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
    model.save("2d/pygame/PPOrandom/trained/{}".format(model_name))


def train_existing_model(
    dim, n_envs, n_timesteps, model_name, n_iters, learning_rate=0.0003, verbose=1
):
    env = make_vec_env(partial(EnvRL, dim), n_envs=n_envs)
    model = PPO.load(
        "2d/pygame/PPOrandom/trained/{}".format(model_name),
        env=env,
        device="cpu",
        verbose=verbose,
        learning_rate=learning_rate,
    )

    for i in range(n_iters):
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
        model.save(
            "2d/pygame/PPOrandom/trained/{}{}".format(model_name, (i + 1) * n_timesteps)
        )


def test_model(model_name, render=False, verbose=True):
    env = EnvRL(2)
    model = PPO.load("2d/pygame/PPOrandom/trained/{}".format(model_name))
    obs = env.reset()

    if render:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        running = True
        font = pygame.font.SysFont("Arial", 16)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if render:
            screen.fill(pygame.Color("black"))

            for ray in env.rays:
                env.space.add(ray)

            env.space.debug_draw(draw_options)
            pygame.display.flip()
            clock.tick(fps)

            for ray in env.rays:
                env.space.remove(ray)

            env.rays = []

        if verbose:
            print("----------------------------------")
            print("Timestep : " + str(env.nbStep))
            print("Action choisie : " + str(action))
            print("Speed : " + str(env.robot.speed))
            print("Angle : " + str(env.robot.angle))
            print("Position : " + str(env.robot.get_position()))
            print("Reward : " + str(reward))
            print("Distance : " + str(info["distance"]))
            print("Distance obs : " + str(obs[6:]))

        if done:
            obs = env.reset()


def play_robot(dim):
    r = Robot(dim)
    r.simulation()


if __name__ == "__main__":
    width, height = 700, 600
    fps = 500

    test_model("PPOrandom", render=True, verbose=True)
    train_existing_model(
        2, 1, 1000000, "PPOrandom", 1, learning_rate=linear_schedule(0.01)
    )
