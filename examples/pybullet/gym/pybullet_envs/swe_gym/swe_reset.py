import time
import math
import numpy as np

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from swe_init import InitCoopEnv

class ResetCoopEnv(InitCoopEnv):

  def __init__(self, desired_obj_pose, p):
    super().__init__(p)
    self.desired_obj_pose = desired_obj_pose
    self.env_state = {}
    self.ComputeEnvState(p)

  def ResetCoop(self, p):
    # Reset the object to the grasp location
    p.resetBasePositionAndOrientation(self.grasped_object, [0, 0.7, 0.02], p.getQuaternionFromEuler([0, 0, 0]))
    # Reset the robots to a position where the grippers can grasp the object
    robot_A_reset = [-1.184011299413845, -1.4364158475353175, -1.0899721376131706, 1.0667906797236881, 1.2044237679252714, 
      -1.2280706100083119, 0.988134098323069, 0.0, 0.0, 0.0, 0.003781686634043995, 0.0]
    
    robot_B_reset = [-1.0663547431079572, -1.4050373708258017, -1.017445912897535, 1.0546136723514878, 1.1190637184529868, 
      -1.2119703753287736, 1.157260237744829, 0.0, 0.0, 0.0, -0.029547676578768497, 0.0]

    for i in range(self.totalNumJoints):
      p.resetJointState(self.kukaId_A, i, robot_A_reset[i])
      p.resetJointState(self.kukaId_B, i, robot_B_reset[i])

    # Grasp the object. Require multiple time steps to do so. Hence 20 "ticks" is used.
    for i in range(20):
      self.gripper(self.kukaId_A, 0.02, p)
      self.gripper(self.kukaId_B, 0.02, p)
      p.stepSimulation()
    
    # Move the object away from the floor after grasping it

    joint_pos_A = [-1.1706906129781278, -1.1734894538763323, -1.1843647849213839, 1.0369803397881985, 1.0339485888804945, -1.4692204508121034, 1.0414560340680936]
    joint_pos_B = [-1.0486248920719832, -1.1473636221157095, -1.1177883364427017, 1.024559282045054, 0.9666073630561682, -1.4632516957457011, 1.208288290582244]
    
    for i in range(5000):
      if (self.useSimulation):
        for i in range(self.numJoints):
          p.setJointMotorControl2(bodyIndex=self.kukaId_A,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=joint_pos_A[i],
                                  targetVelocity=0,
                                  force=100,
                                  positionGain=0.1,
                                  velocityGain=0.5,
                                  maxVelocity=0.01)

          p.setJointMotorControl2(bodyIndex=self.kukaId_B,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=joint_pos_B[i],
                                  targetVelocity=0,
                                  force=100,
                                  positionGain=0.1,
                                  velocityGain=0.5,
                                  maxVelocity=0.01)
      p.stepSimulation()

  # Controls the gripper (open and close commands)
  def gripper(self, robot, finger_target, p):
        '''
        Gripper commands need to be mirrored to simulate behavior of the actual
        UR5. Converts one command input to 6 joint positions, used for the
        robotiq gripper. This is a rough simulation of the way the robotiq
        gripper works in practice, in the absence of a plugin like the one we
        use in Gazebo.

        Parameters:
        -----------
        robot: which robot is being commanded
        cmd: 1x1 array of floating point position commands in [-0.8, 0]
        p: PyBullet client
        '''

        for i in [9,10]:
          p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, finger_target, force= 10)
  
  def GetObservation(self, p):
    # ----------------------------- Get model input ----------------------------------
    self.model_input = np.array([])
    self.ComputeEnvState(p)
    self.model_input = np.append(self.model_input, self.ComputeWrenchFromGraspMatrix(self.kukaId_A, p))
    self.model_input = np.append(self.model_input, self.ComputeWrenchFromGraspMatrix(self.kukaId_B, p))
    self.model_input = np.append(self.model_input, np.array(self.env_state["measured_force_torque_A"]))
    self.model_input = np.append(self.model_input, np.array(self.env_state["object_pose"]))
    self.model_input = np.append(self.model_input, np.array(self.desired_obj_pose))
    assert len(self.model_input) == 30
    return self.model_input
  
  def ComputeEnvState(self, p):
    # Get object pose & velocity
    obj_pose = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose[0]) + list(p.getEulerFromQuaternion(obj_pose[1]))
    obj_vel = p.getBaseVelocity(self.grasped_object)
    obj_vel = list(obj_vel[1]) + list(obj_vel[1])

    # Compute the pose of both end effectors
    robot_A_ee_state = p.getLinkState(self.kukaId_A, self.kukaEndEffectorIndex)
    robot_B_ee_state = p.getLinkState(self.kukaId_B, self.kukaEndEffectorIndex)
    robot_A_ee_pose = list(robot_A_ee_state[0]) + list(p.getEulerFromQuaternion(robot_A_ee_state[1]))
    robot_B_ee_pose = list(robot_B_ee_state[0]) + list(p.getEulerFromQuaternion(robot_B_ee_state[1]))

    # Get Wrench measurements at wrist
    _, _, ft_A, _ = p.getJointState(self.kukaId_A, self.ft_id)
    _, _, ft_B, _ = p.getJointState(self.kukaId_B, self.ft_id)
    force_torque_A = list(ft_A)
    force_torque_B = list(ft_B)

    self.env_state["measured_force_torque_A"] = force_torque_A
    self.env_state["measured_force_torque_B"] = force_torque_B
    self.env_state["object_pose"] = obj_pose
    self.env_state["object_velocity"] = obj_vel
    self.env_state["robot_A_ee_pose"] = robot_A_ee_pose
    self.env_state["robot_B_ee_pose"] = robot_B_ee_pose
  
  def GetEnvState(self):
    return self.env_state
    
  def ComputeWrenchFromGraspMatrix(self, robot, p):
    # TODO (Victor): compute F_T = G_inv * F_o
    desired_obj_wrench = self.ComputeDesiredObjectWrench(p)
    grasp_matrix = self.ComputeGraspMatrix(p)
    wrench = np.linalg.pinv(grasp_matrix).dot(desired_obj_wrench)
    if robot == self.kukaId_A:
      return wrench[:6]
    else:
      return wrench[6:]
  
  def ComputeDesiredObjectWrench(self, p):
    Kp = 5.5 * np.array([5, 5, 5, 2, 2, 2])
    Kv = 5.2 * np.array([1.5, 1.5, 1.5, 0.2, 0.2, 0.2])
    # State of object.
    obj_state = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose_error = [0.0]*6
    for i in range(len(obj_pose_error)):
      obj_pose_error[i] = self.desired_obj_pose[i] - (self.env_state["object_pose"])[i]
    obj_vel_error = self.env_state["object_velocity"]
    for i in range(len(obj_vel_error)):
      obj_vel_error[i] = -obj_vel_error[i]
    desired_obj_wrench = Kp * obj_pose_error + Kv * obj_vel_error
    return desired_obj_wrench
  
  def ComputeGraspMatrix(self, p):
    rp_A = self.env_state["robot_A_ee_pose"]
    rp_B = self.env_state["robot_B_ee_pose"]
    top_three_rows = np.hstack((np.eye(3), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))))
    bottom_three_rows = np.hstack((self.skew(rp_A), np.eye(3), self.skew(rp_B), np.eye(3)))
    grasp_matrix = np.vstack((top_three_rows, bottom_three_rows))
    return grasp_matrix

  def skew(self, vector): 
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])