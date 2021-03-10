import pybullet as p
import pybullet_data

import time
import math

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

#clid = p.connect(p.SHARED_MEMORY)

class ObjDyn:

  def __init__(self):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0.0], useFixedBase=True)
    self.kukaId_A = p.loadURDF("va_kuka_robot/va_iiwa_model.urdf", [-0.3, 0, 0], useFixedBase=True)
    self.kukaId_B = p.loadURDF("va_kuka_robot/va_iiwa_model.urdf", [0.3, 0, 0], useFixedBase=True)
    self.grasped_object = p.loadURDF("va_kuka_robot/grasp_object.urdf", [0, 0.7, 0.02], useFixedBase=False)
    p.setGravity(0, 0, -9.81)
    self.kukaEndEffectorIndex = 7
    self.totalNumJoints = p.getNumJoints(self.kukaId_A)
    # joint damping coefficents
    self.jd = [0.01] * self.totalNumJoints
    # number of joints for just the arm
    self.numJoints = 7

    self.joints_A = self.GetJointInfo(self.kukaId_A)
    self.joints_B = self.GetJointInfo(self.kukaId_B)

    # Enable force torque sensor for the sensor joint.
    self.ft_id = self.joints_A["iiwa7_joint_ft300_sensor"].id # The value is the same for robot B.
    p.enableJointForceTorqueSensor(self.kukaId_A, self.ft_id, 1)
    p.enableJointForceTorqueSensor(self.kukaId_B, self.ft_id, 1)


    p.changeDynamics(self.grasped_object, -1, lateralFriction = 5)
    robots = [self.kukaId_A, self.kukaId_B]
    for robot in robots:
      p.setJointMotorControl2(robot, 15, p.VELOCITY_CONTROL, force=0.0)
      p.setJointMotorControl2(robot, 17, p.VELOCITY_CONTROL, force=0.0)
      self.SetGripperConstraint(robot)

    self.t = 0.
    self.useSimulation = 1
    self.useRealTimeSimulation = 0
    p.setRealTimeSimulation(self.useRealTimeSimulation)

    initPose = [0, 0, -0.5 * math.pi, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    self.model_input = None

    for i in range(self.numJoints):
      p.resetJointState(self.kukaId_A, i, initPose[i])
      p.resetJointState(self.kukaId_B, i, initPose[i])

    self.prevPose_A = [0, 0, 0]
    self.prevPose_B = [0, 0, 0]
    self.prevPose1_A = [0, 0, 0]
    self.prevPose1_B = [0, 0, 0]
    self.hasPrevPose = 0

    self.delta_z_A = 0.0
    self.delta_z_B = 0.0

    # logId1 = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "LOG0001.txt", [0, 1, 2])
    # logId2 = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "LOG0002.txt", bodyUniqueIdA=2)
    

  def GetJointInfo(self, kukaId):
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

  # --------------------------- Run Simulation --------------------------------
  def Run(self):
    if (self.useRealTimeSimulation):
      dt = datetime.now()
      self.t = (dt.second / 60.) * 2. * math.pi
    else:
      self.t += 0.1

    if (self.useSimulation and self.useRealTimeSimulation == 0):
      p.stepSimulation()

    desired_ee_pos_A = [-0.25, 0.7, 0.2 + self.delta_z_A] #[0.2 * math.cos(self.t), 0.2 * math.sin(self.t), 0.4]
    desired_ee_pos_B = [0.25, 0.7, 0.2 + self.delta_z_B]
    #end effector points down, not up (when orientation is used)
    desired_ee_orn_euler_A = [-3.141090814084376, 0.0015622492927442, 0] #[0, -math.pi, 0]
    desired_ee_orn_euler_B = [3.121090814084376, 0.0015622492927442, 0]
    desired_ee_orn_A = p.getQuaternionFromEuler(desired_ee_orn_euler_A)
    desired_ee_orn_B = p.getQuaternionFromEuler(desired_ee_orn_euler_B)

    jointPoses_A = p.calculateInverseKinematics(self.kukaId_A,
                                              self.kukaEndEffectorIndex,
                                              desired_ee_pos_A,
                                              desired_ee_orn_A,
                                              jointDamping=self.jd)
    
    jointPoses_B = p.calculateInverseKinematics(self.kukaId_B,
                                          self.kukaEndEffectorIndex,
                                          desired_ee_pos_B,
                                          desired_ee_orn_B,
                                          jointDamping=self.jd)
    
    if (self.useSimulation):
      a = 2
      for i in range(self.numJoints):
        p.setJointMotorControl2(bodyIndex=self.kukaId_A,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses_A[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.1,
                                velocityGain=0.5)

        p.setJointMotorControl2(bodyIndex=self.kukaId_B,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses_B[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.1,
                                velocityGain=0.5)

    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(self.numJoints):
        p.resetJointState(self.kukaId_A, i, jointPoses_A[i])
        p.resetJointState(self.kukaId_B, i, jointPoses_B[i])
    
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
    _, _, ft_A, _ = p.getJointState(self.kukaId_A, self.ft_id)
    _, _, ft_B, _ = p.getJointState(self.kukaId_B, self.ft_id)
    wrench_A = list(ft_A)
    wrench_B = list(ft_B)

    self.model_input = wrench_A + wrench_B + obj_pose_error
    assert len(self.model_input) == 18

    # ----------------------------------------------------------------------------------
    ls_A = p.getLinkState(self.kukaId_A, self.kukaEndEffectorIndex)
    ls_B = p.getLinkState(self.kukaId_B, self.kukaEndEffectorIndex)

    if (self.hasPrevPose):
      #self.trailDuration is duration (in seconds) after debug lines will be removed automatically
      #use 0 for no-removal
      trailDuration = 15
      p.addUserDebugLine(self.prevPose_A, desired_ee_pos_A, [0, 0, 0.3], 1, trailDuration)
      p.addUserDebugLine(self.prevPose1_A, ls_A[4], [1, 0, 0], 1, trailDuration)
      p.addUserDebugLine(self.prevPose_B, desired_ee_pos_B, [0, 0, 0.3], 1, trailDuration)
      p.addUserDebugLine(self.prevPose1_B, ls_B[4], [1, 0, 0], 1, trailDuration)
    self.prevPose_A = desired_ee_pos_A
    self.prevPose1_A = ls_A[4]
    self.prevPose_B = desired_ee_pos_B
    self.prevPose1_B = ls_B[4]
    self.hasPrevPose = 1

    # ------------------------------- Get Keyboard events -----------------------------
    keys = p.getKeyboardEvents()
    robotA = ord('a')
    robotB = ord('b')
    robotAB = ord('t') # The two robots.
    close_gripper = ord('c')
    open_gripper = ord('o')
    inc_ee_pos = ord('p') # increment end effector pos (p for plus)
    dec_ee_pos = ord('m') # decrement end effector pos (m for minus)
    delta_value = 0.005
    
    if robotA in keys:
      if open_gripper in keys:
        self.gripper(self.kukaId_A, self.joints_A, 0.085)
      elif close_gripper in keys:
        self.gripper(self.kukaId_A, self.joints_A, 0.0)
      elif inc_ee_pos in keys:
        self.delta_z_A += delta_value
      elif dec_ee_pos in keys:
        self.delta_z_A -= delta_value
    elif robotB in keys:
      if open_gripper in keys:
        self.gripper(self.kukaId_B, self.joints_B, 0.085)
      elif close_gripper in keys:
        self.gripper(self.kukaId_B, self.joints_B, 0.0)
      elif inc_ee_pos in keys:
        self.delta_z_B += delta_value
      elif dec_ee_pos in keys:
        self.delta_z_B -= delta_value
    elif robotAB in keys:
      if open_gripper in keys:
        self.gripper(self.kukaId_A, self.joints_A, 0.085)
        self.gripper(self.kukaId_B, self.joints_B, 0.085)
      elif close_gripper in keys:
        self.gripper(self.kukaId_A, self.joints_A, 0.0)
        self.gripper(self.kukaId_B, self.joints_B, 0.0)
      elif inc_ee_pos in keys:
        self.delta_z_A += delta_value
        self.delta_z_B += delta_value
      elif dec_ee_pos in keys:
        self.delta_z_A -= delta_value
        self.delta_z_B -= delta_value

  def gripper(self, kukaId, joints, gripper_opening_length, mode=p.POSITION_CONTROL):
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

        p.setJointMotorControl2(kukaId,
                                joints[gripper_main_control_joint_name].id,
                                p.POSITION_CONTROL,
                                targetPosition=gripper_opening_angle,
                                force=joints[gripper_main_control_joint_name].maxForce,
                                maxVelocity=joints[gripper_main_control_joint_name].maxVelocity)
        
        for i in range(len(mimic_joint_name)):
          joint = joints[mimic_joint_name[i]]
          p.setJointMotorControl2(kukaId, joint.id, p.POSITION_CONTROL,
                                  targetPosition=gripper_opening_angle * mimic_multiplier[i],
                                  force=joint.maxForce,
                                  maxVelocity=joint.maxVelocity)
  
  def SetGripperConstraint(self, kukaId):
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
    

if __name__ == '__main__':
  iiwa = ObjDyn()
  model_input = None
  while 1:
    iiwa.Run()
    model_input = iiwa.model_input
    # print(model_input)
# p.disconnect()