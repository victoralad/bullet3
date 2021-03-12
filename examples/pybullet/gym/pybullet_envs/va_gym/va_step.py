import time
import math

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from va_init import InitCoopEnv

class StepCoopEnv(InitCoopEnv):

  def __init__(self, p):
    super().__init__(p)

  def apply_action(self):
    for i in range(20):
      if (self.useSimulation):
        for i in range(self.numJoints):
          p.setJointMotorControl2(bodyIndex=self.kukaId_A,
                                  jointIndex=i,
                                  controlMode=self.p.POSITION_CONTROL,
                                  targetPosition=joint_pos_A[i],
                                  targetVelocity=0,
                                  force=500,
                                  positionGain=0.1,
                                  velocityGain=0.5)

          p.setJointMotorControl2(bodyIndex=self.kukaId_B,
                                  jointIndex=i,
                                  controlMode=self.p.POSITION_CONTROL,
                                  targetPosition=joint_pos_B[i],
                                  targetVelocity=0,
                                  force=500,
                                  positionGain=0.1,
                                  velocityGain=0.5)
      self.p.stepSimulation()
  
  def get_observation(self):
    pass