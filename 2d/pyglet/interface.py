"""import pyglet
import pyglet.window.key as key
import threading

from pyglet.libs.x11.xlib import XInitThreads


class Interface(pyglet.window.Window):

    def __init__(self, robot_object):
        super(Interface, self).__init__(640, 480)
        self.robot = robot_object


    def on_key_press(self, symbol, modifiers):
        if symbol == key.RIGHT:
            self.robot.angle += 5
        elif symbol == key.LEFT:
            self.robot.angle -= 5
        elif symbol == key.UP:
            self.robot.speed += 0.1
        elif symbol == key.DOWN:
            self.robot.speed -= 0.1

    def on_draw(self):
        self.clear()
        self.robot.robot_sprite.draw()
        self.robot.target.draw()

    def actualiser(self):
        self.clear()
        self.robot.robot_sprite.draw()
        self.robot.target.draw()

    def launch(self):
        pyglet.app.run()"""