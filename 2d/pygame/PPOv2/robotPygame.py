import pyglet
from pyglet import sprite
from pyglet import shapes
from pyglet.window import key
import numpy as np
import gym
import random
import pymunk


class Robot:
    def __init__(self, dim):
        self.dim = dim

        if self.dim == 2:
            self.radius = 20

            self.robot_body = pymunk.Body()
            self.robot_body.position = (320, 240)
            self.robot_body.velocity = (0, 0)

            self.robot_shape = pymunk.Circle(self.robot_body, 20)
            self.robot_shape.density = 1
            self.robot_shape.collision_type = 1

            self.speed = 0
            self.angle = 0

            """self.robot_sprite = sprite.Sprite(robot_image, 256, 256)
            self.robot_sprite.scale = 0.1
            self.robot_sprite.rotation = 0"""

            self.target_body = pymunk.Body()
            self.target_body.position = (320, 190)
            self.target_body.velocity = (0, 0)

            self.target_shape = pymunk.Circle(self.target_body, 5)
            self.target_shape.density = 1
            self.target_shape.collision_type = 5
            self.target_shape.color = (255, 0, 0, 255)
            self.target_shape.sensor = True

    def next_position(self):
        if self.dim == 2:
            self.robot_body.angle = -np.pi * self.angle / 180

            self.robot_shape.body.velocity = (
                self.speed * np.cos(-np.pi * self.angle / 180),
                self.speed * np.sin(-np.pi * self.angle / 180),
            )

    def get_target(self):
        if self.dim == 2:
            return np.array(
                [self.target_shape.body.position.x, self.target_shape.body.position.y]
            )

    def set_target(self, x, y):
        if self.dim == 2:
            self.target.x = x
            self.target.y = y

    def get_position(self):
        if self.dim == 2:
            return np.array(
                [self.robot_shape.body.position.x, self.robot_shape.body.position.y]
            )

    def set_position(self, x, y):
        if self.dim == 2:
            self.robot_shape.body.position.x = x
            self.robot_shape.body.position.y = y

    def get_params(self):
        if self.dim == 2:
            return np.array([self.speed, self.angle])

    def set_params(self, s, a):
        if self.dim == 2:
            self.speed = s
            self.angle = a

    def reset_robot(self):
        self.speed = 0
        self.angle = 0

        # self.robot_shape.body.position = (700 * random.random(), 600 * random.random())
        self.robot_shape.body.position = (350, 300)

    def reset_target(self):
        self.target_shape.body.position = (700 * random.random(), 600 * random.random())
        # self.target_shape.body.position = (350, 300)
