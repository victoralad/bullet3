import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from gym import spaces
import pybullet as p
import pybullet_data
import copy

import numpy as np
import time
import math
from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from swe_init import InitCoopEnv
from swe_reset import ResetCoopEnv
from swe_step import StepCoopEnv

class CoopEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CoopEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    num_robots = 2
    force_vec_len = 6
    max_force = 1.5
    low_action = np.array([-max_force] * force_vec_len)
    high_action = np.array([max_force] * force_vec_len)
    self.action_space = spaces.Box(low_action, high_action)

    # obs_space = [(Fc_1, Fc_2), Measured F_1/T_1, (measured_obj_pose, desired_obj_pose, measured_ee_pose)]
    obs_space = np.array([max_force]*force_vec_len * num_robots + [max_force]*force_vec_len + [2.0, 2.0, 2.0, 3.14, 3.14, 3.14] * 3)
    assert len(obs_space) == 36
    self.observation_space = spaces.Box(-obs_space, obs_space)


    self.desired_obj_pose = [0.0, 0.5, 0.3, 0.0, 0.0, 0.0] # original

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.reset_coop_env = ResetCoopEnv(self.desired_obj_pose, p)
    self.step_coop_env = StepCoopEnv(self.reset_coop_env.robots, self.reset_coop_env.grasped_object, self.reset_coop_env.ft_id, self.desired_obj_pose, p)

    self.num_episodes = 0
    self.time_step = 1

    self.reward_data = [[0.0], [0.0]]
    self.sum_reward = 0.0
    self.overall_reward_sum = 0.0
    self.num_Steps_in_episode = 1
    self.obj_pose_error_norm_sum = 0.0
    self.obj_pose_error_data = [[], []]
    self.obtained_reward, self.obj_pose_error_norm, self.standard_control, self.policy = [], [], [], []

  def step(self, action):
    self.step_coop_env.apply_action(action, self.num_Steps_in_episode, p)
    observation = self.step_coop_env.GetObservation(p)
    reward, obj_pose_error_norm = self.step_coop_env.GetReward(p)

    self.obtained_reward += [reward]
    self.obj_pose_error_norm += [obj_pose_error_norm]

    self.obj_pose_error_norm_sum += obj_pose_error_norm
    done, info = self.step_coop_env.GetInfo(p)
    self.sum_reward += reward
    self.num_Steps_in_episode += 1
    print("---------------------------- Step {} ----------------------------".format(self.time_step))
    self.time_step += 1
    self.reward_data[0] += [self.time_step]
    self.reward_data[1] += [self.sum_reward / self.time_step]
    self.overall_reward_sum = copy.copy(self.sum_reward)
    self.obj_pose_error_data[0] += [self.time_step]
    self.obj_pose_error_data[1] += [self.obj_pose_error_norm_sum / self.time_step]
    print("Observation:", observation)
    print("")
    print("Action:", action)
    print("")
    print("Reward:", reward)
    print("")
    print("Info:", info)
    return observation, reward, done, info

  def reset(self):
    print("------------- Resetting environment, Episode: {} --------------".format(self.num_episodes))
    self.num_episodes += 1.0

    # avg_reward = self.sum_reward / self.num_Steps_in_episode
    # self.reward_data[0] += [self.num_episodes]
    # self.reward_data[1] += [avg_reward]
    # self.sum_reward = 0.0
    
    self.num_Steps_in_episode = 1
    
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