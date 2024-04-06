# Reinforcement learning and robot walking control

The aim of this project is to enable a robot (Unitree Go1) to move independently in its environment using a reinforcement learning algorithm.  

The robot does not have to move along the vertical axis. Therefore, working in a two-dimensional plane is sufficient. Most of the work was done in 2D using [pygame](https://www.pygame.org/) and [pymunk](https://www.pymunk.org/) and can be found in the "2d" folder. Reinforcement learning was carried out using the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/#) library.  

This project was progressing at a rate of 3h/week. As such, it is far from complete and is intended to be continued in the future. A detailed PDF report of the work accomplished is provided above in the GitHub repo.  

## Example of operation

The robot (blue circle) must reach a target (red dot). Obstacles are in its path (gray rectangles). The robot is equipped with a LIDAR, in this case raycasts, to detect the presence of obstacles.

[![Watch the video](https://img.youtube.com/vi/bJotKTeecww/maxresdefault.jpg)](https://youtu.be/bJotKTeecww)
