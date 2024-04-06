import os

import pygame
import pymunk
import pymunk.pygame_util

from robotPygame import Robot
from envRLppo import EnvRL

import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from functools import partial

import optuna

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
        if progress_remaining * initial_value < 0.0001:
            return 0.0001
        return progress_remaining * initial_value

    return func


def optimize_PPO(trial):
    return {
        "gamma": trial.suggest_float("gamma", 0.99, 0.9997),
    }


LOG_DIR = "2d/pygame/PPOtechlabv4/logs_opt/"
OPT_DIR = "2d/pygame/PPOtechlabv4/opt/"


def optimize_agent(trial):
    try:
        model_params = optimize_PPO(trial)

        n_envs = 32

        env = make_vec_env(partial(EnvRL, 2, False), n_envs=n_envs)
        # env = VecFrameStack(env, 4, channels_order="last")

        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=LOG_DIR,
            verbose=1,
            batch_size=512,
            **model_params,
        )
        model.learn(total_timesteps=250000, log_interval=1)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=25)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, "trial_{}_best_model".format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        print(e)
        return -1000


def optimize_model():
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=25, n_jobs=1)
    print(study.best_params)
    print(study.best_trial)


def train_model(
    dim,
    n_envs,
    n_timesteps,
    model_name,
    learning_rate=0.0003,
    gamma=0.99,
    n_steps=2048,
    gae_lambda=0.95,
):
    print(
        "Tensorboard command: python3 -m tensorboard.main --logdir=2d/pygame/PPOtechlabv4/tensorboard/{}/".format(
            model_name
        )
    )
    env = make_vec_env(partial(EnvRL, dim, False), n_envs=n_envs)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        n_steps=n_steps,
        learning_rate=learning_rate,
        batch_size=512,
        tensorboard_log="2d/pygame/PPOtechlabv4/tensorboard/{}".format(model_name),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
    model.save("2d/pygame/PPOtechlabv4/trained/{}".format(model_name))


def train_existing_model(
    dim, n_envs, n_timesteps, model_name, n_iters, learning_rate=0.0003, verbose=1
):
    env = make_vec_env(partial(EnvRL, dim, False), n_envs=n_envs)
    model = PPO.load(
        "2d/pygame/PPOtechlabv4/trained/{}".format(model_name),
        env=env,
        device="cpu",
        verbose=verbose,
        learning_rate=learning_rate,
        tensorboard_log="2d/pygame/PPOtechlabv4/tensorboard/{}".format(model_name),
    )

    for i in range(n_iters):
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
        model.save(
            "2d/pygame/PPOtechlabv4/trained/{}{}".format(
                model_name, (i + 1) * n_timesteps
            )
        )


def test_model(model_name, render=False, verbose=True):
    env = EnvRL(2, render)

    model = PPO.load("2d/pygame/PPOtechlabv4/trained/{}".format(model_name))

    obs = env.reset()

    if render:
        pygame.init()

        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
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

    test_model("PPOobstacles", render=True, verbose=True)


#   Optimisations :
# batch_size : 512
# gamma : 0.991
