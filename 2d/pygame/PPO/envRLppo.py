import gym
from gym import spaces
import numpy as np
from robotPygame import Robot
import pymunk
import random


class EnvRL(gym.Env):
    def __init__(self, dim, nbObstacles=2):
        self.robot = Robot(dim)
        self.nbStep = 0

        self.height = 380
        self.width = 640

        self.vmax = 10

        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(14,))
        self.action_space = spaces.MultiDiscrete([2, 2])

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.add(self.robot.robot_body, self.robot.robot_shape)
        self.space.add(self.robot.target_body, self.robot.target_shape)

        self.goal_collision_handler = self.space.add_collision_handler(2, 5)
        self.goal_collision_handler.begin = self.goal_collision

        self.nbObstacles = nbObstacles
        self.obstacles = []
        self.handler = []
        self.rays = []

    def _get_observation(self):
        pos = self.robot.get_position()
        params = self.robot.get_params()
        target = self.robot.get_target()
        obstaclesDist = self.raycast()

        # Normalisation
        pos = (pos - np.array([self.width / 2, self.height / 2])) / np.array(
            [self.width / 2, self.height / 2]
        )
        target = (target - np.array([self.width / 2, self.height / 2])) / np.array(
            [self.width / 2, self.height / 2]
        )

        params[0] = params[0] / self.vmax
        params[1] = params[1] / 360

        return (
            np.concatenate((pos, params, target, obstaclesDist))
            .reshape(
                14,
            )
            .astype(np.float32)
        )

    def step(self, action):
        v, a = self.robot.get_params()
        if action[0] == 0 and self.robot.speed < 10:
            v += 0.5
        elif action[0] == 1 and self.robot.speed > -10:
            v -= 0.5
        if action[1] == 0 and self.robot.angle < 180:
            a += 5
        elif action[1] == 1 and self.robot.angle > -180:
            a -= 5

        self.robot.set_params(v, a)
        self.robot.next_position()

        self.space.step(1 / 2)

        self.nbStep += 1

        obs = self._get_observation()
        done = self.terminated(obs)
        reward = self.reward(obs, done)
        info = {
            "distance": np.linalg.norm(
                self.robot.get_position() - self.robot.get_target()
            )
        }

        return obs, reward, done, info

    def reset(self):
        for obstacle in self.obstacles:
            self.space.remove(obstacle[0], obstacle[1])

        self.nbObstacles = random.randint(2, 5)

        self.obstacles = []

        self.create_obstacles()

        self.nbStep = 0
        self.robot.reset()

        distObjets = self.raycast()

        while np.min(distObjets) < 10 / 500:
            self.robot.reset()
            distObjets = self.raycast()

        self.rays = []

        return self._get_observation()

    def render(self, mode="human"):
        pass

    def terminated(self, obs):
        done = False
        if self.robot.dim == 2:
            pos = self.robot.get_position()
            target = self.robot.get_target()
            posx, posy = self.robot.get_position()
            dist = np.linalg.norm(pos - target)
            distObstacles = obs[6:]
            if (
                self.nbStep >= 400
                or dist <= 15
                or posx < -500
                or posx > self.width + 500
                or posy < -500
                or posy > self.height + 500
            ):
                done = True
                if dist <= 15:
                    print("Bravo !")
            for distObstacle in distObstacles:
                if distObstacle <= 5 / 500:
                    return True
        return done

    def reward(self, obs, done):
        dist = np.linalg.norm(self.robot.get_position() - self.robot.get_target())
        """distObs = obs[6:]

        if done:
            if dist <= 15:
                return 100
            else:
                return -100

        else:
            return min(distObs) - 1"""
        if done:
            if dist <= 15:
                return 1000
        return -dist / 500

    def create_obstacles(self):
        for i in range(self.nbObstacles):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (random.randint(0, 640), random.randint(0, 380))

            height = random.randint(25, 75)
            width = random.randint(25, 75)

            vs = [
                (-width, -height),
                (-width, height),
                (width, height),
                (width, -height),
            ]
            shape = pymunk.Poly(body, vs)
            shape.collision_type = 2
            shape.density = 1
            self.space.add(body, shape)

            self.obstacles.append((body, shape))

    def raycast(self):
        distances = []

        for i in range(8):
            angle = -self.robot.angle * np.pi / 180 + i * np.pi / 4
            beginPoint = self.robot.get_position() + np.array(
                [
                    int((self.robot.radius + 5) * np.cos(angle)),
                    int((self.robot.radius + 5) * np.sin(angle)),
                ]
            )
            endPoint = beginPoint + np.array(
                [int(500 * np.cos(angle)), int(500 * np.sin(angle))]
            )
            ray = self.space.segment_query_first(
                (beginPoint[0], beginPoint[1]),
                (endPoint[0], endPoint[1]),
                1,
                pymunk.ShapeFilter(),
            )

            if ray:
                contactPoint = ray.point
                line = pymunk.Segment(
                    self.space.static_body,
                    (beginPoint[0], beginPoint[1]),
                    (contactPoint[0], contactPoint[1]),
                    1,
                )
                line.sensor = True
                line.color = (255, 0, 0, 255)
                line.collision_type = 3
                self.rays.append(line)
                distances.append(np.linalg.norm(beginPoint - contactPoint) / 500)

            else:
                distances.append(1)

        return distances

    def goal_collision(self, arbiter, space, data):
        return False

    def reset_target(self):
        self.robot.reset_target()
