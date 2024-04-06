import sys
import os
import copy
import math
from collections import OrderedDict

from gym import spaces
import numpy as np
import pybullet as pb



import stable_baselines3.common.env_checker
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
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
    max_episode_steps=10000,
)

class envRL2Spot (spotBezierEnv) :

    def __init__(self,
                target = np.array([3, 0, 0]),
                render = False,
                height_field = False,
                on_rack = False,
                draw_foot_path = True,
                env_randomizer = SpotEnvRandomizer()) -> None:

        self.nbStep = 0

        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)
        
        self.target = target
        self.distanceInitiale = np.linalg.norm(self.target)
        self.previousDistance = self.distanceInitiale

        self.ja = np.array([0])
        self.obs = np.zeros((36, 36))

        self.score_target = 15

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
        self.objects = createWall("empty_map.txt")
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

        self.action_space = spaces.MultiDiscrete([2,2])
        
        #self.observation_space = spaces.Box(low=0, high=1, shape=(36, 36))

        #self.observation_space = spaces.Box(low=0, high=255, shape=(36, 36, 1), dtype=np.uint8) # Pour CNN

        self.observation_space = spaces.Dict({"position": spaces.Box(low= -math.inf, high=math.inf, shape=(3,1,1)), "image": spaces.Box(low=0, high=255, shape=(36, 36, 1), dtype=np.uint8)})

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

        self.StepLength = 0.1
        self.YawRate = 0
        self.state = super().reset


    def _get_observation(self):
        self.position = pb.getBasePositionAndOrientation(self.spot.quadruped)[0]
        self.orientation = pb.getEulerFromQuaternion(self.spot.GetBaseOrientation())

        if self.nbStep % 100 == 0:
            viewMatrix = pb.computeViewMatrix(
                    cameraEyePosition= [self.position[0], self.position[1], self.position[2]+0.2],
                    cameraTargetPosition=[self.position[0] + 2*np.cos(self.orientation[2]), self.position[1] + 2*np.sin(self.orientation[2]), self.position[2] + 0.2],
                    cameraUpVector=[0, 0, 0.5])

            width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                width=36, 
                height=36,
                viewMatrix=viewMatrix,
                projectionMatrix=self.projectionMatrix)
            
            return OrderedDict({"image" : (255 * depthImg).astype(int).reshape((36, 36, 1)).astype(np.uint8), 
            "position" : np.array(self.spot.GetBasePosition()).reshape((3,1,1)).astype(np.float32)}) #Pour CNN

    
    def _get_info(self):
        return {"distance": math.dist(self.position, self.target)}

    def reset(self, initial_motor_angles=None, reset_duration=1, desired_velocity=None, desired_rate=None):
        super().reset(initial_motor_angles, reset_duration, desired_velocity, desired_rate)
        #self.target = list(np.random.randint(-10, 10, size=2))+[0]
        self.target = [3,0,0]
        self.StepLength = 0
        self.YawRate = 0
        self.previousDistance = self.distanceInitiale
        self.score_target = 0
        self.nbStep = 0
        return self._get_observation()
    

    def close(self) -> None:
        return super().close()


    def _reward(self):
        """
        #Calculate the distance to the wall
        d = self.col_detector.compute_distances()
        in_col = self.col_detector.in_collision(margin=0.1)

        #Score des murs : x point qui correspond a la distance minimale à un mur
        score_walls = min(d)

        #Score de collision : 2 points si non collision
        score_col = 0
        if in_col:
            score_col = -100
        else:
            score_col = 0

        #Score de distance target
        distanceActuelle = math.dist(self.position, self.target)
        diff = self.previousDistance - distanceActuelle
        if diff < -0.00001:
            self.score_target -= 0.1
        elif diff > 0.00001:
            self.score_target += 0.1
        self.previousDistance = distanceActuelle

        #Score si atteint la cible
        score_reach = 0
        if distanceActuelle < 0.1:
            score_reach += 1000
        """
        distance = np.linalg.norm(np.array(self.target) - np.array(self.spot.GetBasePosition()))
        reward = 5 - int(distance)

        if distance < 0.1:
            return 1
        else:
            return 0

        return self.score_target + score_col

    def step(self, action):
        #Verifie si le robot est en collision ou est tombé
        _, _, done, _ = super().step(0)
        d = self.col_detector.compute_distances()
        in_col = self.col_detector.in_collision(margin=0.1)
        done = bool(done or in_col)

        self.bz_step.ramp_up()

        pos, orn, _, LateralFraction, _, StepVelocity, ClearanceHeight, PenetrationDepth = self.bz_step.StateMachine()

        actionStep = action[0]
        actionYaw = action[1]

        if actionStep == 0:
            if self.StepLength < 0.10:
                self.StepLength += 0.01
        elif actionStep == 1:
            if self.StepLength > -0.10:
                self.StepLength -= 0.01
        else:
            self.StepLength = self.StepLength

        if actionYaw == 0:
            if self.YawRate < 1.5:
                self.YawRate += 0.01
        elif actionYaw == 1:
            if self.YawRate > -1.5:
                self.YawRate -= 0.01
        else:
            self.YawRate = self.YawRate

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

        if self.nbStep % 100 == 0:
            obs = self._get_observation()
            self.obs = obs
        else:
            obs = self.obs
            obs["position"] = np.array(self.spot.GetBasePosition()).reshape((3,1,1)).astype(np.float32)

        self.nbStep += 1

        info = self._get_info()
        reward = self._reward()

        if self.nbStep == 5000 or reward == 1:
            done = True

        return obs, reward, done, info
"""
env = make_vec_env(envRL2Spot, n_envs=16)

#Check environment
#stable_baselines3.common.env_checker.check_env(env)
#print("Fin du check de l'environnement")

model = PPO("MultiInputPolicy", env, verbose=1, device="cpu")

for i in range(1):
    print("Début de l'apprentissage")
    model.learn(total_timesteps=100000)
    print("Fin de l'apprentissage")

    model.save("PPO_spot")
    print("Fin de l'enregistrement du modèle")

del model # remove to demonstrate saving and loading

env = make_vec_env(envRL2Spot, n_envs=16)
model = PPO("MultiInputPolicy", env, verbose=1, device="cpu")
TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{PPO}/{2048*16*iters}")"""

env = envRL2Spot()

model = PPO.load("PPO/393216")
print("Fin du chargement du modèle")

obs = env.reset()
print("Fin du reset de l'environnement")

print(model)

while True:
    print("----------------------------------")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print("Timestep : " +  str(env.nbStep))
    print("Action choisie : " + str(action))
    print("StepLength : " + str(env.StepLength))
    print("YawRate : " + str(env.YawRate))
    print("Position : " + str(env.spot.GetBasePosition()) )
    print("Reward : " + str(reward))
    print("Distance murs : " + str(env.col_detector.compute_distances()))
    if done:
        obs = env.reset()

