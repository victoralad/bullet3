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
    max_force = 10.5
    low_action = np.array([-max_force] * force_vec_len)
    high_action = np.array([max_force] * force_vec_len)
    self.action_space = spaces.Box(low_action, high_action)

    # obs_space = [(Fc_1, Fc_2), Measured F_1/T_1, (measured_obj_pose, desired_eeA_pose, measured_ee_pose)]
    obs_space = np.array([2.0, 2.0, 2.0, 3.14, 3.14, 3.14] * 2)
    assert len(obs_space) == 12
    self.observation_space = spaces.Box(-obs_space, obs_space)

    self.desired_eeA_pose = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]
    # self.desired_eeA_pose = [0.7, 0.0, 0.4, 0.0, 0.0, 0.0] # original

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.reset_coop_env = ResetCoopEnv(self.desired_eeA_pose, p)
    self.step_coop_env = StepCoopEnv(self.reset_coop_env.robots, self.reset_coop_env.grasped_object, self.reset_coop_env.ft_id, self.desired_eeA_pose, p)

    self.num_episodes = 0
    self.time_step = 1

    self.reward_data = [[0.0], [0.0]]
    self.sum_reward = 0.0
    self.num_steps_in_episode = 1
    self.obj_pose_error_norm_episode_sum = 0.0
    self.mean_obj_pose_error_norm_data = [[0.0], [self.obj_pose_error_norm_episode_sum]]

  def step(self, action):
    self.step_coop_env.apply_action(action, p)
    observation = self.step_coop_env.GetObservation(p)
    reward, obj_pose_error_norm = self.step_coop_env.GetReward(p, self.num_steps_in_episode)
    print("******************************")
    print(self.num_steps_in_episode)
    print(self.mean_obj_pose_error_norm_data[1][-1])
    print(self.reward_data[1][-1])

    done, info = self.step_coop_env.GetInfo(p, self.num_steps_in_episode)
    self.obj_pose_error_norm_episode_sum += obj_pose_error_norm
    self.sum_reward += reward
    self.num_steps_in_episode += 1
    print("---------------------------- Step {} ----------------------------".format(self.time_step))
    self.time_step += 1
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

    
    self.mean_obj_pose_error_norm_data[0] += [self.num_episodes]
    self.mean_obj_pose_error_norm_data[1] += [self.obj_pose_error_norm_episode_sum / self.num_steps_in_episode]
    self.obj_pose_error_norm_episode_sum = 0.0

    self.reward_data[0] += [self.num_episodes]
    self.reward_data[1] += [self.sum_reward / self.num_steps_in_episode]
    self.sum_reward = 0.0

    self.num_steps_in_episode = 1
    
    self.reset_coop_env.ResetCoop(p)
    observation = self.reset_coop_env.GetObservation(p)
    return observation  # reward, done, info can't be included

  def render(self, mode='human', close=False):
    pass
