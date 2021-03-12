import time
import math

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from va_reset import ResetCoopEnv

class StepCoopEnv(ResetCoopEnv):

  def __init__(self, robots, numJoints, totalNumJoints, useSimulation):
    self.robot_A = robots[0]
    self.robot_B = robots[1]
    self.numJoints = numJoints
    self.useSimulation = useSimulation
    # joint damping coefficents
    self.jd = [0.01] * totalNumJoints

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
    self.p.stepSimulation()
  
  def GetObservation(self, p):
    return ResetCoopEnv.GetObservation(p)
  
  def GetReward(self, p):
    reward = None
    return reward