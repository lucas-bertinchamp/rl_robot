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

            robot_image = pyglet.image.load('pyglet/android.png')
            robot_image.anchor_x = robot_image.width // 2
            robot_image.anchor_y = robot_image.height // 2

            self.robot_sprite = sprite.Sprite(robot_image, 256, 256)
            self.robot_sprite.scale = 0.1
            self.robot_sprite.rotation = 0

            self.target_x = 320
            self.target_y = 190

            self.target = shapes.Circle(x=self.target_x, y=self.target_y, radius=5, color=(255, 0, 0))

            self.speed = 0
            self.angle = 0

    def next_position(self):
        if self.dim == 2:
            self.robot_sprite.rotation = self.angle

            self.robot_sprite.x += self.speed * np.cos(-np.pi * self.angle / 180)
            self.robot_sprite.y += self.speed * np.sin(-np.pi * self.angle / 180)

    def get_target(self):
        if self.dim == 2:
            return np.array([self.target_x, self.target_y])
        
    def set_target(self, x, y):
        if self.dim == 2:
            self.target_x = x
            self.target_y = y

    def get_position(self):
        if self.dim == 2:
            return np.array([self.robot_sprite.x, self.robot_sprite.y])

    def set_position(self, x, y):
        if self.dim == 2:
            self.robot_sprite.x = x
            self.robot_sprite.y = y

    def get_params(self):
        if self.dim == 2:
            return np.array([self.speed, self.angle])

    def set_params(self, s, a):
        if self.dim == 2:
            self.speed = s
            self.angle = a
    
    def reset(self):
        self.robot_sprite.x = 640*random.random()
        self.robot_sprite.y = 380*random.random()

        self.target_x = 640*random.random()
        self.target_y = 380*random.random()

        self.target.x = self.target_x
        self.target.y = self.target_y

        self.speed = 0
        self.angle = 0

    