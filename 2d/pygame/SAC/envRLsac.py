import gym
from gym import spaces
import numpy as np
from robotPygame import Robot
import pymunk
import random


class EnvRL(gym.Env):
    def __init__(self, dim, rendering, nbObstacles=2):
        self.robot = Robot(dim)
        self.rendering = rendering

        self.nbStep = 0
        self.nbRays = 8

        self.height = 600
        self.width = 700

        self.maxTimestep = 500

        self.vmax = 10
        self.numberRaycast = 3

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6 + self.nbRays * self.numberRaycast,)
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.add(self.robot.robot_body, self.robot.robot_shape)
        self.space.add(self.robot.target_body, self.robot.target_shape)

        self.goal_collision_handler = self.space.add_collision_handler(2, 5)
        self.goal_collision_handler.begin = self.goal_collision
        self.goal_collision_handler.pre_solve = self.goal_collision

        self.robot_collision_handler = self.space.add_collision_handler(1, 2)
        self.robot_collision_handler.begin = self.robot_collision
        self.robot_collision_handler.pre_solve = self.robot_collision

        self.nbObstacles = nbObstacles
        self.obstacles = []
        self.handler = []
        self.rays = []

        self.previousDist = np.linalg.norm(
            self.robot.get_position() - self.robot.get_target()
        )
        self.previousRaycast = [self.raycast() for i in range(self.numberRaycast - 1)]

    def _get_observation(self):
        pos = self.robot.get_position()
        params = self.robot.get_params()
        target = self.robot.get_target()
        obstaclesDist = self.raycast()

        # Normalisation
        pos_norm = [
            np.interp(pos[0], [-500, self.width + 500], [-1, 1]),
            np.interp(pos[1], [-500, self.height + 500], [-1, 1]),
        ]
        target_norm = [
            np.interp(target[0], [-500, self.width + 500], [-1, 1]),
            np.interp(target[1], [-500, self.height + 500], [-1, 1]),
        ]

        params[0] = params[0] / self.vmax
        params[1] = params[1] / 180

        obs = (
            np.concatenate(
                (
                    pos_norm,
                    params,
                    target_norm,
                    obstaclesDist,
                    self.previousRaycast[-1],
                    self.previousRaycast[-2],
                )
            )
            .reshape(
                6 + self.nbRays * self.numberRaycast,
            )
            .astype(np.float32)
        )
        self.previousRaycast.append(obstaclesDist)
        self.previousRaycast.pop(0)
        return obs

    def step(self, action):
        v = action[0] * self.vmax
        a = action[1] * 180

        self.robot.set_params(v, a)
        self.robot.next_position()

        self.space.step(1 / 4)

        self.nbStep += 1

        obs = self._get_observation()
        dist = np.linalg.norm(self.robot.get_position() - self.robot.get_target())
        done = self.terminated(obs, dist)
        reward = self.reward(obs, done, dist)
        info = {
            "distance": dist,
        }

        return obs, reward, done, info

    def reset(self):
        for obstacle in self.obstacles:
            self.space.remove(obstacle[0], obstacle[1])

        self.currentReward = 0

        self.nbObstacles = 5
        self.obstacles = []
        self.create_obstacles()

        self.nbStep = 0
        self.robot.reset_robot()
        self.robot.reset_target()

        self.rays = []
        self.previousRaycast = [self.raycast() for i in range(self.numberRaycast - 1)]

        return self._get_observation()

    def render(self, mode="human"):
        pass

    def terminated(self, obs, dist):
        done = False
        if self.robot.dim == 2:
            posx, posy = self.robot.get_position()
            distObstacles = obs[6 : 6 + self.nbRays]
            if (
                self.nbStep >= self.maxTimestep
                or dist <= 15
                or posx < -500
                or posx > self.width + 500
                or posy < -500
                or posy > self.height + 500
            ):
                done = True
            for distObstacle in distObstacles:
                if distObstacle <= -495 / 500:
                    return True
        return done

    def reward(self, obs, done, dist):
        Rtarget = 0
        Robstacles = 0
        if done:
            if dist <= 15:
                Rtarget = 10
            elif min(obs[6:]) <= -490 / 500:
                Robstacles = -10

            return Rtarget + Robstacles

        else:
            Rtarget = self.previousDist - dist
            self.previousDist = dist

            return Rtarget + Robstacles

    def create_obstacles(self):
        for i in range(self.nbObstacles):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (
                random.randint(0, self.width),
                random.randint(0, self.height),
            )

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

        for i in range(self.nbRays):
            angle = (
                (-self.robot.angle * np.pi) / 180
                - (3 * np.pi / 4)
                + 3 * np.pi * i / (2 * self.nbRays)
            )
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
                if self.rendering:
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

                distances.append(
                    (np.linalg.norm(beginPoint - contactPoint) - 500) / 500
                )

            else:
                distances.append(1)

        return distances

    def goal_collision(self, arbiter, space, data):
        self.robot.reset_target()
        return False

    def robot_collision(self, arbiter, space, data):
        if self.nbStep < 10:
            self.robot.reset_robot()
        else:
            self.reset()
        return False
