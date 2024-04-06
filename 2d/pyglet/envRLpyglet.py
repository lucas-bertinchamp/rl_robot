import gym
from gym import spaces
import numpy as np
from robotPyglet import Robot

class EnvRL(gym.Env):

    def __init__(self, dim):
        self.robot = Robot(dim)
        self.nbStep = 0

        self.height = 380
        self.width = 640

        self.vmax = 10

        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(6,))
        self.action_space = spaces.MultiDiscrete([2, 2])

    def _get_observation(self):
        pos = self.robot.get_position()
        params = self.robot.get_params()
        target = self.robot.get_target()

        #Normalisation
        pos = (pos - np.array([self.width/2, self.height/2])) / np.array([self.width/2, self.height/2])
        target = (target - np.array([self.width/2, self.height/2])) / np.array([self.width/2, self.height/2])

        params[0] = params[0] / self.vmax
        params[1] = params[1] / 360

        return np.concatenate((pos, params, target)).reshape(6,).astype(np.float32)

    def step(self, action):

        v, a = self.robot.get_params()
        if action[0] == 0 and self.robot.speed < 10:
            v += 0.5
        elif action[0] == 1 and self.robot.speed > -10 :
            v -= 0.5
        if action[1] == 0 and self.robot.angle < 180:
            a += 5
        elif action[1] == 1 and self.robot.angle > -180:
            a -= 5

        self.robot.set_params(v, a)
        self.robot.next_position()

        self.nbStep += 1

        obs = self._get_observation()
        done = self.terminated()
        reward = self.reward()
        info = {"distance": np.linalg.norm(self.robot.get_position() - self.robot.get_target())}

        return obs, reward, done, info

    def reset(self):
        self.nbStep = 0
        self.robot.reset()
        return self._get_observation()

    def render(self, interface, mode="human"):
        interface.actualiser()

    def terminated(self):
        done = False
        if self.robot.dim == 2:
            pos = self.robot.get_position()
            target = self.robot.get_target()
            posx, posy = self.robot.get_position()
            dist = np.linalg.norm(pos - target)
            if self.nbStep >= 1000 or dist <= 15 or posx < -500 or posx > self.width + 500 or posy < -500 or posy > self.height+500:
                done = True
                if dist <= 15:
                    print("Bravo !")
        return done

    def reward(self):
        dist = np.linalg.norm(self.robot.get_position() - self.robot.get_target())

        if dist <= 15:
            reward = 1
        else:
            reward = 0
        
        return reward

