import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from gym import spaces
import pybullet as p
import pybullet_data

import numpy as np
import time
import math
from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from va_init import InitCoopEnv
from va_reset import ResetCoopEnv
from va_step import StepCoopEnv

class CoopEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CoopEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    num_robots = 2
    low_action = np.array([-1.0, -1.0, 0.0, -3.14, -3.14, -3.14] * num_robots) * 0.5
    high_action = np.array([1.0, 1.0, 1.0, 3.14, 3.14, 3.14] * num_robots) * 0.5
    self.action_space = spaces.Box(low_action, high_action)

    obs_space = np.array([100]*6*num_robots + [2.0, 2.0, 2.0, 3.14, 3.14, 3.14])
    assert len(obs_space) == 18
    self.observation_space = spaces.Box(-obs_space, obs_space)

    self.desired_obj_pose = [0.0, 0.3, 0.4, 0.0, 0.0, 0.0]

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.reset_coop_env = ResetCoopEnv(self.desired_obj_pose, p)
    self.step_coop_env = StepCoopEnv(self.reset_coop_env.robots, self.reset_coop_env.grasped_object, self.reset_coop_env.ft_id, self.desired_obj_pose, p)

    self.num_episodes = 0

  def step(self, action):
    self.step_coop_env.apply_action(action, p)
    observation = self.step_coop_env.GetObservation(p)
    reward = self.step_coop_env.GetReward(p)
    done, info = self.step_coop_env.GetInfo(p)
    print("---------------------------- Step ----------------------------")
    print("Reward:", reward)
    print("")
    print("Observation:", observation)
    print("")
    print("Info:", info)
    return observation, reward, done, info

  def reset(self):
    print("------------- Resetting environment, Episode: {} --------------".format(self.num_episodes))
    self.num_episodes += 1
    self.reset_coop_env.ResetCoop(p)
    observation = self.reset_coop_env.GetObservation(p)
    return observation  # reward, done, info can't be included

  def render(self, mode='human', close=False):
    pass

# if __name__ == '__main__':
#   coop = CoopEnv()
#   coop.reset()
#   action = [0.1]*12
#   coop.step(action)
#   while 1:
#       a = 1
#     # coop.reset(p)
#     # coop.step()