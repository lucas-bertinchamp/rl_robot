import sys
import os
import copy
import math

from gym import spaces
import numpy as np
import pybullet as pb

import stable_baselines3.common.env_checker
from stable_baselines3 import DQN

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../../')

from collision_utils import *
from map_utils import *
from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.util.gui import GUI
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.Kinematics.LieAlgebra import RPY
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.spot_env_randomizer import SpotEnvRandomizer

# TESTING
from spotmicro.OpenLoopSM.SpotOL import BezierStepper



from gym.envs.registration import register

register(
    id='RLSpot-v0',
    entry_point='spotbullet.src:envRLSpot',
    max_episode_steps=300,
)

class envRLSpot (spotBezierEnv) :

    def __init__(self,
                target = np.array([3, 0, 0]),
                render = False,
                height_field = False,
                on_rack = False,
                draw_foot_path = True,
                env_randomizer = SpotEnvRandomizer()) -> None:
        
        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)
        
        self.target = target
        self.ja = np.array([0])
        self.obs = np.zeros((20, 20))

        super().__init__(render=render,
                        on_rack=on_rack,
                        height_field=height_field,
                        draw_foot_path=draw_foot_path,
                        env_randomizer=env_randomizer)

        # Find abs path to this file
        self.my_path = os.path.abspath(os.path.dirname(__file__))
        self.results_path = os.path.join(self.my_path, "../results")
        self.models_path = os.path.join(self.my_path, "../models")

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        # Set seeds
        seed = 0
        self.seed(seed)
        np.random.seed(seed)

        # Spawn all objects
        self.objects = createWall("map2.txt")
        #spot id is in env.spot.quadruped

        # Create collision detector
        self.objectsCollsion = [NamedCollisionObject("wall{}".format(i)) for i in range(len(self.objects))]

        self.objects["robot"] = self.spot.quadruped
        self.robot_collision = NamedCollisionObject("robot")

        self.pairs_collision = [(self.objectsCollsion[i], self.robot_collision) for i in range(len(self.objects) - 1)]
        self.col_detector = CollisionDetector(
            self._pybullet_client._client,  # client ID for collision physics server
            self.objects,  # bodies in the simulation
            self.pairs_collision, # these are the pairs of objects to compute distances between
        )

        self.action_space = spaces.Discrete(6)
        
        #self.observation_space = spaces.Box(low=0, high=1, shape=(36, 36))
        self.observation_space = spaces.Box(low=0, high=255, shape=(36, 36, 1), dtype=np.uint8) # Pour CNN
        #print(self.observation_space)
        #print("---------------------------------")

        self.g_u_i = GUI(self.spot.quadruped)

        self.spotModel = SpotModel()
        self.T_bf0 = self.spotModel.WorldToFoot
        self.T_bf = copy.deepcopy(self.T_bf0)

        self.bzg = BezierGait(dt=self._time_step)

        self.bz_step = BezierStepper(dt=self._time_step, mode=0)
        
        self.FL_phases = []
        self.FR_phases = []
        self.BL_phases = []
        self.BR_phases = []

        self.FL_Elbow = []

        self.yaw = 0.0

        self.StepLength = 0
        self.YawRate = 0
        self.state = super().reset


    def _get_observation(self):
        self.position = pb.getBasePositionAndOrientation(self.spot.quadruped)[0]
        self.orientation = pb.getEulerFromQuaternion(self.spot.GetBaseOrientation())

        viewMatrix = pb.computeViewMatrix(
                cameraEyePosition= [self.position[0], self.position[1], self.position[2]+0.2],
                cameraTargetPosition=[self.position[0] + 2*np.cos(self.orientation[2]), self.position[1] + 2*np.sin(self.orientation[2]), self.position[2] + 0.2],
                cameraUpVector=[0, 0, 0.5])

        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            width=36, 
            height=36,
            viewMatrix=viewMatrix,
            projectionMatrix=self.projectionMatrix)
    
        return (255 * depthImg).astype(int).reshape((36, 36, 1)) #Pour CNN
        #return depthImg
    
    def _get_info(self):
        return {"distance": math.dist(self.position, self.target)}

    def reset(self, initial_motor_angles=None, reset_duration=1, desired_velocity=None, desired_rate=None):
        super().reset(initial_motor_angles, reset_duration, desired_velocity, desired_rate)
        #self.target = list(np.random.randint(-10, 10, size=2))+[0]
        self.target = [3,0,0]
        return self._get_observation()
    

    def close(self) -> None:
        return super().close()


    def _reward(self):
        #Calculate the distance to the wall
        d = self.col_detector.compute_distances()
        in_col = self.col_detector.in_collision(margin=0.1)

        score_walls = 0
        for i in range(len(d)):
            score_walls += 1/d[i]**2
    
        return -math.dist(self.position, self.target) - score_walls + len(d)

    def step(self, action):

        _, _, done, _ = super().step(0)

        self.bz_step.ramp_up()

        pos, orn, _, LateralFraction, _, StepVelocity, ClearanceHeight, PenetrationDepth = self.bz_step.StateMachine()

        
        if action == 0 :
            if (self.StepLength <= 0.15):
                self.StepLength += (0.15-self.StepLength)/10
        elif action == 1 :
            if (self.StepLength >= -0.15):
                self.StepLength += (-0.15 - self.StepLength)/10
        elif action == 2 :
            self.YawRate = -1.5
            self.StepLength += (0.15-self.StepLength)/10
        elif action == 3 :
            self.YawRate = 1.5
            self.StepLength += (0.15-self.StepLength)/10

        if self.StepLength < 0.01 and self.StepLength > -0.01:
            self.StepLength = 0
        if self.YawRate < 0.01 and self.YawRate > -0.01:
            self.YawRate = 0

        self.StepLength -= self.StepLength/10
        self.YawRate -= self.YawRate/10

        self.yaw = self.return_yaw()

        # TEMP
        self.bz_step.StepLength = self.StepLength
        self.bz_step.LateralFraction = LateralFraction
        self.bz_step.YawRate = self.YawRate
        self.bz_step.StepVelocity = StepVelocity

        self.FL_phases.append(self.spot.LegPhases[0])
        self.FR_phases.append(self.spot.LegPhases[1])
        self.BL_phases.append(self.spot.LegPhases[2])
        self.BR_phases.append(self.spot.LegPhases[3])

        # Get Desired Foot Poses
        self.T_bf = self.bzg.GenerateTrajectory(self.StepLength, LateralFraction, self.YawRate,
                                      StepVelocity, self.T_bf0, self.T_bf,
                                      ClearanceHeight, PenetrationDepth)
        joint_angles = self.spotModel.IK(orn, pos, self.T_bf)

        self.FL_Elbow.append(np.degrees(joint_angles[0][-1]))

        self.pass_joint_angles(joint_angles.reshape(-1))
        # Get External Observations
        self.spot.GetExternalObservations(self.bzg, self.bz_step)
        # Step
        
        obs = self._get_observation()

        info = self._get_info()
        reward = self._reward()

        return obs, reward, done, info


env = envRLSpot()

#Check environment
#stable_baselines3.common.env_checker.check_env(env)
#print("Fin du check de l'environnement")

model = DQN("CnnPolicy", env, verbose=1)
print("Fin de création du modèle")

model.learn(total_timesteps=25000, log_interval=100) #valeur par défaut de log_interval
print("Fin de l'apprentissage")

model.save("dqn_cartpole")
print("Fin de l'enregistrement du modèle")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")
print("Fin du chargement du modèle")
obs = env.reset()
print("Fin du reset de l'environnement")

print(model)
while True:
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    print("Action choisie : " + str(action))
    obs, reward, done, info = env.step(action)
    print("Position : " + str(env.spot.GetBasePosition()) )
    if done:
        obs = env.reset()

