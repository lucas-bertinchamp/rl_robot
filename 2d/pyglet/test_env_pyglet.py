import threading

import pyglet
import pymunk

from robotPyglet import Robot
from envRLpyglet import EnvRL
import gym
import numpy as np
import stable_baselines3
import stable_baselines3.common.env_checker
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from functools import partial


def train_model(dim, n_envs, n_timesteps, model_name):
    env = make_vec_env(partial(EnvRL, dim), n_envs=n_envs)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
    model.save("pyglet/PPO_android/{}".format(model_name))


def train_existing_model(dim, n_envs, n_timesteps, model_name, n_iters):
    env = make_vec_env(partial(EnvRL, dim), n_envs=n_envs)
    model = PPO.load(
        "pyglet/PPO_android/{}".format(model_name), env=env, device="cpu", verbose=1
    )

    for i in range(n_iters):
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
        model.save("pyglet/PPO_android/{}{}".format(model_name, i * n_timesteps))


def test_model(model_name, render=False, verbose=True):
    def test_model2():
        model = PPO.load("pyglet/PPO_android/{}".format(model_name))
        obs = env.reset()

        while True:
            print("----------------------------------")
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if verbose:
                print("Timestep : " + str(env.nbStep))
                print("Action choisie : " + str(action))
                print("Speed : " + str(env.robot.speed))
                print("Angle : " + str(env.robot.angle))
                print("Position : " + str(env.robot.get_position()))
                print("Reward : " + str(reward))
                print("Distance : " + str(info["distance"]))

            if done:
                obs = env.reset()

    if render:
        env = EnvRL(2)

        space = pymunk.Space()
        space.gravity = (0.0, 0.0)

        t = threading.Thread(target=test_model2)
        t.setName("Environnement")
        t.start()

        window = pyglet.window.Window(640, 380)

        def update(dt):
            window.clear()
            env.robot.robot_sprite.draw()
            env.robot.target.draw()

        pyglet.clock.schedule_interval(update, 1 / 180)
        pyglet.app.run()


def play_robot(dim):
    r = Robot(dim)
    r.simulation()


if __name__ == "__main__":
    target = np.array([320, 190])

    test_model("targetRobotRandom", render=True, verbose=True)
    train_existing_model(2, 1, 250000, "robotFixe", 10)
