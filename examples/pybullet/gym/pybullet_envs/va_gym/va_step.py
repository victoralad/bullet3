import time
import math
import numpy as np

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from va_reset import ResetCoopEnv

class StepCoopEnv(ResetCoopEnv):

  def __init__(self, robots, grasped_object, ft_id, p):
    self.robot_A = robots[0]
    self.robot_B = robots[1]
    self.kukaEndEffectorIndex = 7
    self.totalNumJoints = p.getNumJoints(self.robot_A)
    self.numJoints = 7 # number of joints for just the arm
    self.jd = [0.01] * self.totalNumJoints
    self.useSimulation = 1
    self.ft_id = ft_id # F/T sensor joint id.
    self.grasped_object = grasped_object

  def apply_action(self, action, p):
    assert len(action) == 12
    desired_ee_pos_A = action[:3]
    desired_ee_orn_A = p.getQuaternionFromEuler(action[3:6])
    desired_ee_pos_B = action[6:9]
    desired_ee_orn_B = p.getQuaternionFromEuler(action[9:])

    joint_pos_A = p.calculateInverseKinematics(self.robot_A,
                                          self.kukaEndEffectorIndex,
                                          desired_ee_pos_A,
                                          desired_ee_orn_A,
                                          jointDamping=self.jd)
    
    joint_pos_B = p.calculateInverseKinematics(self.robot_B,
                                          self.kukaEndEffectorIndex,
                                          desired_ee_pos_B,
                                          desired_ee_orn_B,
                                          jointDamping=self.jd)
    
    if (self.useSimulation):
      for i in range(self.numJoints):
        p.setJointMotorControl2(bodyIndex=self.robot_A,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_pos_A[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.1,
                                velocityGain=0.5)

        p.setJointMotorControl2(bodyIndex=self.robot_B,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_pos_B[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.1,
                                velocityGain=0.5)
    p.stepSimulation()
  
  def GetObservation(self, p):
    # ----------------------------- Get model input ----------------------------------
    obj_pose_error = [None] * 6
    wrench_A = [None] * 6
    wrench_B = [None] * 6
    desired_obj_pose = [0.0, 0.3, 0.4, 0.0, 0.0, 0.0]

    # Get object pose
    obj_pose = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose[0]) + list(p.getEulerFromQuaternion(obj_pose[1]))
    for i in range(len(obj_pose)):
      obj_pose_error[i] = desired_obj_pose[i] - obj_pose[i]

    # Get Wrench measurements at wrist
    _, _, ft_A, _ = p.getJointState(self.robot_A, self.ft_id)
    _, _, ft_B, _ = p.getJointState(self.robot_B, self.ft_id)
    wrench_A = list(ft_A)
    wrench_B = list(ft_B)

    self.model_input = wrench_A + wrench_B + obj_pose_error
    assert len(self.model_input) == 18
    return self.model_input
  
  def GetReward(self, p):
    reward = None
    u = np.array(self.model_input[-6:])
    Q = np.eye(len(u))
    obj_pose_error_reward =  -u.T @ u
    fI = np.array(self.model_input[:6]) - np.array(self.model_input[6:12]) # Internal stress = f_A - f_B. The computed value is wrong and must be corrected ASAP.
    wrench_reward = -fI.T @ fI
    print(wrench_reward)
    reward = obj_pose_error_reward + wrench_reward
    return reward
  
  def GetInfo(self, p):
    done = False
    info = None
    obj_pose_error = self.model_input[-6:]
    norm = np.linalg.norm(obj_pose_error)
    if norm > 2.0:
      done = True
      info = "The norm of the object pose error, {}, is significant enough to reset the training episode.".format(norm)
    return done, info