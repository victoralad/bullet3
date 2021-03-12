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
    # Example when using discrete actions:
    N_DISCRETE_ACTIONS = 5
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    HEIGHT = 5
    WIDTH = 5
    N_CHANNELS = 10
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # InitCoopEnv(p)
    self.reset_coop_env = ResetCoopEnv(p)
    self.step_coop_env = StepCoopEnv(p)
  def step(self, action):
    pass
    # return observation, reward, done, info
  def reset(self, arg):
    self.reset_coop_env.ResetCoop(arg)
    # return observation  # reward, done, info can't be included
  def render(self, mode='human', close=False):
    pass

if __name__ == '__main__':
  coop = CoopEnv()
  while 1:
    coop.reset(p)
    # coop.step()