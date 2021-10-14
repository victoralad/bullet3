import pybullet as p
import pybullet_data
import time
import math

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

class InitCoopEnv:

  def __init__(self, p):
    p.loadURDF("plane.urdf", [0, 0, 0.0], useFixedBase=True)
    self.kukaId_A = p.loadURDF("va_kuka_robot/va_iiwa_model.urdf", [-0.3, 0, 0], useFixedBase=True)
    self.kukaId_B = p.loadURDF("va_kuka_robot/va_iiwa_model.urdf", [0.3, 0, 0], useFixedBase=True)
    self.grasped_object = p.loadURDF("va_kuka_robot/grasp_object.urdf", [0, 0.7, 0.02], useFixedBase=False)
    p.setGravity(0, 0, -9.81)
    self.kukaEndEffectorIndex = 7
    self.totalNumJoints = p.getNumJoints(self.kukaId_A)

    # number of joints for just the arm
    self.numJoints = 7

    self.joints_A = self.GetJointInfo(self.kukaId_A, p)
    self.joints_B = self.GetJointInfo(self.kukaId_B, p)

    # Enable force torque sensor for the sensor joint.
    self.ft_id = self.joints_A["iiwa7_joint_ft300_sensor"].id # The value is the same for robot B.
    p.enableJointForceTorqueSensor(self.kukaId_A, self.ft_id, 1)
    p.enableJointForceTorqueSensor(self.kukaId_B, self.ft_id, 1)

    p.changeDynamics(self.grasped_object, -1, lateralFriction = 5)
    self.robots = [self.kukaId_A, self.kukaId_B]
    for robot in self.robots:
      p.setJointMotorControl2(robot, 15, p.VELOCITY_CONTROL, force=0.0)
      p.setJointMotorControl2(robot, 17, p.VELOCITY_CONTROL, force=0.0)
      self.SetGripperConstraint(robot, p)

    self.useSimulation = 1
    self.useRealTimeSimulation = 0
    p.setRealTimeSimulation(self.useRealTimeSimulation)


  # Computes joint attributes that are useful in other computations.
  def GetJointInfo(self, kukaId, p):
    joints = AttrDict()
    jointInfo = namedtuple("jointInfo", ["id","name","lowerLimit","upperLimit","maxForce","maxVelocity"])

    # get jointInfo and index of dummy_center_indicator_link
    for i in range(self.totalNumJoints):
      info = p.getJointInfo(kukaId, i)
      jointID = info[0]
      jointName = info[1].decode("utf-8")
      jointLowerLimit = info[8]
      jointUpperLimit = info[9]
      jointMaxForce = info[10]
      jointMaxVelocity = info[11]
      singleInfo = jointInfo(jointID, jointName, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
      joints[singleInfo.name] = singleInfo
    
    return joints
  
  # Constrains the joints of the grippers so they appear real in simulation.
  def SetGripperConstraint(self, kukaId, p):
    a = p.createConstraint(kukaId,
                  10,
                  kukaId,
                  15,
                  jointType=p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    p.changeConstraint(a, gearRatio=-1, erp=0.1, maxForce=50)

    b = p.createConstraint(kukaId,
                  12,
                  kukaId,
                  17,
                  jointType=p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    p.changeConstraint(b, gearRatio=-1, erp=0.1, maxForce=50)

    c = p.createConstraint(kukaId,
                  14,
                  kukaId,
                  15,
                  jointType=p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    d = p.createConstraint(kukaId,
                  16,
                  kukaId,
                  17,
                  jointType=p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    p.changeConstraint(d, gearRatio=-1, erp=0.1, maxForce=50)