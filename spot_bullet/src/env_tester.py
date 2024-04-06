#!/usr/bin/env python

import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt
import copy

from collision_utils import *
from map_utils import *

import sys

import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../../')

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.util.gui import GUI
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.Kinematics.LieAlgebra import RPY
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.spot_env_randomizer import SpotEnvRandomizer

# TESTING
from spotmicro.OpenLoopSM.SpotOL import BezierStepper

import time


import argparse

# ARGUMENTS
descr = "Spot Mini Mini Environment Tester (No Joystick)."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-hf",
                    "--HeightField",
                    help="Use HeightField",
                    action='store_true')
parser.add_argument("-r",
                    "--DebugRack",
                    help="Put Spot on an Elevated Rack",
                    action='store_true')
parser.add_argument("-p",
                    "--DebugPath",
                    help="Draw Spot's Foot Path",
                    action='store_true')
parser.add_argument("-ay",
                    "--AutoYaw",
                    help="Automatically Adjust Spot's Yaw",
                    action='store_true')
parser.add_argument("-ar",
                    "--AutoReset",
                    help="Automatically Reset Environment When Spot Falls",
                    action='store_true')
parser.add_argument("-dr",
                    "--DontRandomize",
                    help="Do NOT Randomize State and Environment.",
                    action='store_true')
parser.add_argument("-rd",
                    "--RenderActivated",
                    help="Activate render of the Environment.",
                    action='store_true')
ARGS = parser.parse_args()


def main():
    """ The main() function. """

    print("STARTING SPOT TEST ENV")
    seed = 0
    max_timesteps = 4e6

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")
    models_path = os.path.join(my_path, "../models")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if ARGS.DebugRack:
        on_rack = True
    else:
        on_rack = False

    if ARGS.DebugPath:
        draw_foot_path = True
    else:
        draw_foot_path = False

    if ARGS.HeightField:
        height_field = True
    else:
        height_field = False

    if ARGS.DontRandomize:
        env_randomizer = None
    else:
        env_randomizer = SpotEnvRandomizer()

    if ARGS.RenderActivated:
        render = True
    else :
        render = False
    
    env = spotBezierEnv(render=render,
                        on_rack=on_rack,
                        height_field=height_field,
                        draw_foot_path=draw_foot_path,
                        env_randomizer=env_randomizer)


    # Set seeds
    env.seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    print("STATE DIM: {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM: {}".format(action_dim))
    max_action = float(env.action_space.high[0])

    state = env.reset()

    if render : g_u_i = GUI(env.spot.quadruped)

    spot = SpotModel()
    T_bf0 = spot.WorldToFoot
    T_bf = copy.deepcopy(T_bf0)

    bzg = BezierGait(dt=env._time_step)

    bz_step = BezierStepper(dt=env._time_step, mode=0)

    action = env.action_space.sample()

    FL_phases = []
    FR_phases = []
    BL_phases = []
    BR_phases = []

    FL_Elbow = []

    yaw = 0.0

    StepLength = 0
    YawRate = 0

    print("SPAWNING OBJECT")

    # Spawn all objects

    objects = createWall("map.txt")
    #spot id is in env.spot.quadruped

    # Create collision detector
    objectsCollsion = [NamedCollisionObject("wall{}".format(i)) for i in range(len(objects))]
    print(objects)
    objects["robot"] = env.spot.quadruped
    robot_collision = NamedCollisionObject("robot")

    pairs_collision = [(objectsCollsion[i], robot_collision) for i in range(len(objects) - 1)]
    col_detector = CollisionDetector(
        env._pybullet_client._client,  # client ID for collision physics server
        objects,  # bodies in the simulation
        pairs_collision, # these are the pairs of objects to compute distances between
    )

    print("OBJECTS SPAWNED")


    print("CAMERA")


    projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.1)


    print("STARTED SPOT TEST ENV")
    t = 0
    compteurTemps = 1
    while t < (int(max_timesteps)):
        position = pb.getBasePositionAndOrientation(env.spot.quadruped)[0]
        orientation = pb.getEulerFromQuaternion(env.spot.GetBaseOrientation())
        print(position)
        print(orientation)

        if (t // 100 == compteurTemps):
            viewMatrix = pb.computeViewMatrix(
                cameraEyePosition= [position[0], position[1], position[2]+0.2],
                cameraTargetPosition=[position[0] + 2*np.cos(orientation[2]), position[1] + 2*np.sin(orientation[2]), position[2] + 0.2],
                cameraUpVector=[0, 0, 0.5])

            width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            width=224, 
            height=224,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)
            compteurTemps += 1

        bz_step.ramp_up()

        pos, orn, _, LateralFraction, _, StepVelocity, ClearanceHeight, PenetrationDepth = bz_step.StateMachine()

        if render :
            pos, orn, _, LateralFraction, _, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod = g_u_i.UserInput()
        else :
            SwingPeriod=0.0

        keys = pb.getKeyboardEvents()
        if keys.get(65297):
            if (StepLength <= 0.15):
                StepLength += (0.15-StepLength)/10
        if keys.get(65298):
            if (StepLength >= -0.15):
                StepLength += (-0.15 - StepLength)/10
        if keys.get(65296):
            YawRate = -1.5
        if keys.get(65295):
            YawRate = 1.5

        if StepLength < 0.01 and StepLength > -0.01:
            StepLength = 0
        if YawRate < 0.01 and YawRate > -0.01:
            YawRate = 0

        StepLength -= StepLength/10
        YawRate -= YawRate/10

        #Calculate the distace to the wall
        d = col_detector.compute_distances()
        in_col = col_detector.in_collision(margin=0.1)

        print(f"Distance to obstacles = {d}")
        print(f"In collision = {in_col}")

        # Update Swing Period
        if render :
            bzg.Tswing = SwingPeriod

        yaw = env.return_yaw()

        P_yaw = 5.0

        if ARGS.AutoYaw:
            YawRate += -yaw * P_yaw

        # print("YAW RATE: {}".format(YawRate))

        # TEMP
        bz_step.StepLength = StepLength
        bz_step.LateralFraction = LateralFraction
        bz_step.YawRate = YawRate
        bz_step.StepVelocity = StepVelocity

        contacts = state[-4:]

        FL_phases.append(env.spot.LegPhases[0])
        FR_phases.append(env.spot.LegPhases[1])
        BL_phases.append(env.spot.LegPhases[2])
        BR_phases.append(env.spot.LegPhases[3])

        # Get Desired Foot Poses
        T_bf = bzg.GenerateTrajectory(StepLength, LateralFraction, YawRate,
                                      StepVelocity, T_bf0, T_bf,
                                      ClearanceHeight, PenetrationDepth,
                                      contacts)
        joint_angles = spot.IK(orn, pos, T_bf)

        FL_Elbow.append(np.degrees(joint_angles[0][-1]))

        # for i, (key, Tbf_in) in enumerate(T_bf.items()):
        #     print("{}: \t Angle: {}".format(key, np.degrees(joint_angles[i])))
        # print("-------------------------")

        env.pass_joint_angles(joint_angles.reshape(-1))
        # Get External Observations
        env.spot.GetExternalObservations(bzg, bz_step)
        # Step
        state, reward, done, _ = env.step(action)
        print(StepLength, YawRate)
        # print("IMU Roll: {}".format(state[0]))
        # print("IMU Pitch: {}".format(state[1]))
        # print("IMU GX: {}".format(state[2]))
        # print("IMU GY: {}".format(state[3]))
        # print("IMU GZ: {}".format(state[4]))
        # print("IMU AX: {}".format(state[5]))
        # print("IMU AY: {}".format(state[6]))
        # print("IMU AZ: {}".format(state[7]))
        # print("-------------------------")
        if done:
            if ARGS.AutoReset:
                env.reset()
                # plt.plot()
                # # plt.plot(FL_phases, label="FL")
                # # plt.plot(FR_phases, label="FR")
                # # plt.plot(BL_phases, label="BL")
                # # plt.plot(BR_phases, label="BR")
                # plt.plot(FL_Elbow, label="FL ELbow (Deg)")
                # plt.xlabel("dt")
                # plt.ylabel("value")
                # plt.title("Leg Phases")
                # plt.legend()
                # plt.show()

        # time.sleep(1.0)

        t += 1
    env.close()




if __name__ == '__main__':
    main()

