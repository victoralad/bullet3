# import pybullet as p
# import pybullet_data

import time
import math

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

class ResetCoopEnv:

  def __init__(self, p):
    # self.p.connect(self.p.GUI)
    # self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.p = p
    self.p.loadURDF("plane.urdf", [0, 0, 0.0], useFixedBase=True)
    self.kukaId_A = self.p.loadURDF("va_kuka_robot/va_iiwa_model.urdf", [-0.3, 0, 0], useFixedBase=True)
    self.kukaId_B = self.p.loadURDF("va_kuka_robot/va_iiwa_model.urdf", [0.3, 0, 0], useFixedBase=True)
    self.grasped_object = self.p.loadURDF("va_kuka_robot/grasp_object.urdf", [0, 0.7, 0.02], useFixedBase=False)
    self.p.setGravity(0, 0, -9.81)
    self.kukaEndEffectorIndex = 7
    self.totalNumJoints = self.p.getNumJoints(self.kukaId_A)
    # joint damping coefficents
    self.jd = [0.01] * self.totalNumJoints
    # number of joints for just the arm
    self.numJoints = 7

    self.joints_A = self.GetJointInfo(self.kukaId_A)
    self.joints_B = self.GetJointInfo(self.kukaId_B)

    # Enable force torque sensor for the sensor joint.
    self.ft_id = self.joints_A["iiwa7_joint_ft300_sensor"].id # The value is the same for robot B.
    self.p.enableJointForceTorqueSensor(self.kukaId_A, self.ft_id, 1)
    self.p.enableJointForceTorqueSensor(self.kukaId_B, self.ft_id, 1)

    self.p.changeDynamics(self.grasped_object, -1, lateralFriction = 5)
    robots = [self.kukaId_A, self.kukaId_B]
    for robot in robots:
      self.p.setJointMotorControl2(robot, 15, self.p.VELOCITY_CONTROL, force=0.0)
      self.p.setJointMotorControl2(robot, 17, self.p.VELOCITY_CONTROL, force=0.0)
      self.SetGripperConstraint(robot)

    self.t = 0.
    self.useSimulation = 1
    self.useRealTimeSimulation = 0
    self.p.setRealTimeSimulation(self.useRealTimeSimulation)

    self.model_input = None

    robot_A_reset = [-1.184011299413845, -1.4364158475353175, -1.0899721376131706, 1.0667906797236881, 1.2044237679252714, 
      -1.2280706100083119, 0.988134098323069, 0.0, 0.0, 0.0, 0.003781686634043995, 0.0, 0.014366518891656452, 
      0.0, -0.16958579599715132, 0.1427849400696791, -0.22500275319458551, 0.19728657436618674]
    
    robot_B_reset = [-1.0663547431079572, -1.4050373708258017, -1.017445912897535, 1.0546136723514878, 1.1190637184529868, 
      -1.2119703753287736, 1.157260237744829, 0.0, 0.0, 0.0, -0.029547676578768497, 0.0, -0.002485325591840869, 
      0.0, -0.1716278040056272, 0.017623415295276643, -0.09600218730971453, 0.07728500814094336]

    for i in range(self.totalNumJoints):
      self.p.resetJointState(self.kukaId_A, i, robot_A_reset[i])
      self.p.resetJointState(self.kukaId_B, i, robot_B_reset[i])

    for i in range(20):
      self.gripper(self.kukaId_A, self.joints_A, 0.0)
      self.gripper(self.kukaId_B, self.joints_B, 0.0)
      self.p.stepSimulation()
    
    # initPose = [0, 0, -0.5 * math.pi, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # for i in range(self.numJoints):
    #   p.resetJointState(self.kukaId_A, i, initPose[i])
    #   p.resetJointState(self.kukaId_B, i, initPose[i])
    # # self.p.stepSimulation()

  def GetJointInfo(self, kukaId):
    joints = AttrDict()
    jointInfo = namedtuple("jointInfo", ["id","name","lowerLimit","upperLimit","maxForce","maxVelocity"])

    # get jointInfo and index of dummy_center_indicator_link
    for i in range(self.totalNumJoints):
      info = self.p.getJointInfo(kukaId, i)
      jointID = info[0]
      jointName = info[1].decode("utf-8")
      jointLowerLimit = info[8]
      jointUpperLimit = info[9]
      jointMaxForce = info[10]
      jointMaxVelocity = info[11]
      singleInfo = jointInfo(jointID, jointName, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
      joints[singleInfo.name] = singleInfo
    
    return joints

  def gripper(self, kukaId, joints, gripper_opening_length):
    '''
    Gripper commands need to be mirrored to simulate behavior of the actual
    UR5. Converts one command input to 6 joint positions, used for the
    robotiq gripper. This is a rough simulation of the way the robotiq
    gripper works in practice, in the absence of a plugin like the one we
    use in Gazebo.

    Parameters:
    -----------
    cmd: 1x1 array of floating point position commands in [-0.8, 0]
    mode: PyBullet control mode
    '''

    gripper_main_control_joint_name = "robotiq_85_left_knuckle_joint"
    mimic_joint_name = ["robotiq_85_right_knuckle_joint",
                        "robotiq_85_left_inner_knuckle_joint",
                        "robotiq_85_right_inner_knuckle_joint",
                        "robotiq_85_left_finger_tip_joint",
                        "robotiq_85_right_finger_tip_joint"]
    mimic_multiplier = [1, 1, 1, -1, -1]

    # gripper control
    gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)    # angle calculation

    self.p.setJointMotorControl2(kukaId,
                            joints[gripper_main_control_joint_name].id,
                            self.p.POSITION_CONTROL,
                            targetPosition=gripper_opening_angle,
                            force=joints[gripper_main_control_joint_name].maxForce,
                            maxVelocity=joints[gripper_main_control_joint_name].maxVelocity)
    
    for i in range(len(mimic_joint_name)):
      joint = joints[mimic_joint_name[i]]
      self.p.setJointMotorControl2(kukaId, joint.id, self.p.POSITION_CONTROL,
                              targetPosition=gripper_opening_angle * mimic_multiplier[i],
                              force=joint.maxForce,
                              maxVelocity=joint.maxVelocity)
  
  def SetGripperConstraint(self, kukaId):
    a = self.p.createConstraint(kukaId,
                  10,
                  kukaId,
                  15,
                  jointType=self.p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    self.p.changeConstraint(a, gearRatio=-1, erp=0.1, maxForce=50)

    b = self.p.createConstraint(kukaId,
                  12,
                  kukaId,
                  17,
                  jointType=self.p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    self.p.changeConstraint(b, gearRatio=-1, erp=0.1, maxForce=50)

    c = self.p.createConstraint(kukaId,
                  14,
                  kukaId,
                  15,
                  jointType=self.p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    d = self.p.createConstraint(kukaId,
                  16,
                  kukaId,
                  17,
                  jointType=self.p.JOINT_FIXED,
                  jointAxis=[0, 0, 1],
                  parentFramePosition=[0, 0, 0],
                  childFramePosition=[0, 0, 0])
    self.p.changeConstraint(d, gearRatio=-1, erp=0.1, maxForce=50)