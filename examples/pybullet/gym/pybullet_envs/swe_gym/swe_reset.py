import time
import math
import numpy as np

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from swe_init import InitCoopEnv

class ResetCoopEnv(InitCoopEnv):

  def __init__(self, desired_eeA_pose, p):
    super().__init__(p)
    self.desired_eeA_pose = desired_eeA_pose
    self.env_state = {}
    self.ComputeEnvState(p)

  def ResetCoop(self, p):
    # Reset the object to the grasp location
    p.resetBasePositionAndOrientation(self.grasped_object, [2.7, 0.0, 0.02], [0, 0, 1, 1])
    # # Reset the robots to a position where the grippers can grasp the object
    # robot_A_reset = [1.6215659536342868, 0.9575781843548509, -0.14404269719109372, -1.496128956979969, 0.18552992566925916, 2.4407372489326353,
    #  1.8958616972085343, 0.01762362413070885, 0.017396558579594615, 0.0, 0.0, 0.0]
    
    # robot_B_reset = [2.470787979046169, 1.5992683071619733, -1.3190493822244016, -1.3970919589354867, 1.5466399312306398, 1.8048923566089303,
    #  1.8741340429221176, 0.04180727854471872, 0.03980317811581496, 0.0, 0.0, 0.0]
    
    # robot_A_reset = [0.10073155, 0.86246903, -0.03992083, -1.61260852, -0.10303224, 2.39708345, -0.80150425, 0.0162362413070885, 0.047396558579594615, 0.04, 0.04, 0.0]
    # robot_B_reset = [0.10073155, 0.86246903, -0.03992083, -1.61260852, -0.10303224, 2.39708345, -0.60150425, 0.0180727854471872, 0.05980317811581496, 0.04, 0.04, 0.0]

  
  def GetObservation(self, p):
    # ----------------------------- Get model input ----------------------------------
    self.model_input = np.array([])
    self.ComputeEnvState(p)
    self.model_input = np.append(self.model_input, np.array(self.desired_eeA_pose))
    self.model_input = np.append(self.model_input, np.array(self.env_state["robot_A_ee_pose"]))
    assert len(self.model_input) == 12
    return self.model_input
  
  def ComputeEnvState(self, p):

    # Compute the pose of both end effectors in the world frame.
    robot_A_ee_state = p.getLinkState(self.kukaId_A, self.kukaEndEffectorIndex, 1)
    robot_A_ee_pose = list(robot_A_ee_state[0]) + list(p.getEulerFromQuaternion(robot_A_ee_state[1]))
    robot_A_ee_vel = list(robot_A_ee_state[-2]) + list(robot_A_ee_state[-1])

    self.env_state["robot_A_ee_pose"] = robot_A_ee_pose
    self.env_state["robot_A_ee_vel"] = robot_A_ee_vel

  
  def GetEnvState(self):
    return self.env_state
    

  

  