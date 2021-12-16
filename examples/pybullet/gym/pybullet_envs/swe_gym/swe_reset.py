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
    p.resetBasePositionAndOrientation(self.grasped_object, [0.7, 0.0, 0.02], [0, 0, 1, 1])
    # # Reset the robots to a position where the grippers can grasp the object
    # robot_A_reset = [1.6215659536342868, 0.9575781843548509, -0.14404269719109372, -1.496128956979969, 0.18552992566925916, 2.4407372489326353,
    #  1.8958616972085343, 0.01762362413070885, 0.017396558579594615, 0.0, 0.0, 0.0]
    
    # robot_B_reset = [2.470787979046169, 1.5992683071619733, -1.3190493822244016, -1.3970919589354867, 1.5466399312306398, 1.8048923566089303,
    #  1.8741340429221176, 0.04180727854471872, 0.03980317811581496, 0.0, 0.0, 0.0]
    
    robot_A_reset = [0.10073155, 0.86246903, -0.03992083, -1.61260852, -0.10303224, 2.39708345, -0.80150425, 0.0162362413070885, 0.047396558579594615, 0.04, 0.04, 0.0]
    robot_B_reset = [0.10073155, 0.86246903, -0.03992083, -1.61260852, -0.10303224, 2.39708345, -0.60150425, 0.0180727854471872, 0.05980317811581496, 0.04, 0.04, 0.0]

    for i in range(self.totalNumJoints):
      p.resetJointState(self.kukaId_A, i, robot_A_reset[i])
      p.resetJointState(self.kukaId_B, i, robot_B_reset[i])


    # # Open gripper
    # # Grasp the object. Require multiple time steps to do so. Hence 20 "ticks" is used.
    # for i in range(20):
    #   self.gripper(self.kukaId_A, 0.08, p)
    #   self.gripper(self.kukaId_B, 0.08, p)
    #   p.stepSimulation()


    # Grasp the object. Require multiple time steps to do so. Hence 20 "ticks" is used.
    for i in range(20):
      self.gripper(self.kukaId_A, 0.02, p)
      self.gripper(self.kukaId_B, 0.02, p)
      p.stepSimulation()
    
    # while(1):
    #   a = 1

    # Move the object away from the floor after grasping it

    # joint_pos_A = [-1.1706906129781278, -1.1734894538763323, -1.1843647849213839, 1.0369803397881985, 1.0339485888804945, -1.4692204508121034, 1.0414560340680936]
    # joint_pos_B = [-1.1706906129781278, -1.1734894538763323, -1.1843647849213839, 1.0369803397881985, 1.0339485888804945, -1.4692204508121034, 1.0414560340680936]

    joint_pos_A = [0.10073155, -0.14246903, -0.33992083, -2.61260852, -0.20303224, 2.39708345, -0.80150425]
    joint_pos_B = [0.10073155, -0.14246903, -0.33992083, -2.61260852, -0.20303224, 2.39708345, -0.80150425]


    for i in range(10000):
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
    # while(1):
    #   a = 1

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
    self.model_input = np.append(self.model_input, np.array(self.env_state["robot_A_ee_pose"]))
    assert len(self.model_input) == 36
    return self.model_input
  
  def ComputeEnvState(self, p):
    # Get object pose & velocity
    obj_pose_state = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose_state[0]) + list(p.getEulerFromQuaternion(obj_pose_state[1]))
    obj_vel = p.getBaseVelocity(self.grasped_object)
    obj_vel = list(obj_vel[1]) + list(obj_vel[1])

    # Compute the pose of both end effectors in the world frame.
    robot_A_ee_state = p.getLinkState(self.kukaId_A, self.kukaEndEffectorIndex)
    robot_B_ee_state = p.getLinkState(self.kukaId_B, self.kukaEndEffectorIndex)
    robot_A_ee_pose = list(robot_A_ee_state[0]) + list(p.getEulerFromQuaternion(robot_A_ee_state[1]))
    robot_B_ee_pose = list(robot_B_ee_state[0]) + list(p.getEulerFromQuaternion(robot_B_ee_state[1]))

    # Compute the pose of both end effectors in the object's frame.
    obj_orn_matrix = np.array(p.getMatrixFromQuaternion(obj_pose_state[1]))
    obj_orn_matrix = np.reshape(obj_orn_matrix, (3, 3))
    robot_A_ee_pose_obj_frame = np.linalg.inv(obj_orn_matrix).dot((np.array(robot_A_ee_state[0]) - np.array(obj_pose_state[0])))
    robot_B_ee_pose_obj_frame = np.linalg.inv(obj_orn_matrix).dot((np.array(robot_B_ee_state[0]) - np.array(obj_pose_state[0])))

    # Get Wrench measurements at wrist
    _, _, ft_A, _ = p.getJointState(self.kukaId_A, self.ft_id)
    _, _, ft_B, _ = p.getJointState(self.kukaId_B, self.ft_id)
    force_torque_A = list(ft_A)
    force_torque_B = list(ft_B)

    self.env_state["measured_force_torque_A"] = force_torque_A
    self.env_state["measured_force_torque_B"] = force_torque_B
    self.env_state["object_pose"] = obj_pose
    self.env_state["object_orn_matrix"] = obj_orn_matrix
    self.env_state["object_velocity"] = obj_vel
    self.env_state["robot_A_ee_pose"] = robot_A_ee_pose
    self.env_state["robot_B_ee_pose"] = robot_B_ee_pose
    self.env_state["robot_A_ee_pose_obj_frame"] = robot_A_ee_pose_obj_frame
    self.env_state["robot_B_ee_pose_obj_frame"] = robot_B_ee_pose_obj_frame
  
  def GetEnvState(self):
    return self.env_state
    
  def ComputeWrenchFromGraspMatrix(self, robot, p):
    # TODO (Victor): compute F_T = G_inv * F_o
    desired_obj_wrench = self.ComputeDesiredObjectWrench(p)
    grasp_matrix = self.ComputeGraspMatrix(p)
    inv_grasp_matrix = np.linalg.pinv(grasp_matrix)
    # grasp_matrix_sq = grasp_matrix.dot(grasp_matrix.T)
    # inv_grasp_matrix = grasp_matrix.T.dot(np.linalg.inv(grasp_matrix_sq))
    wrench = inv_grasp_matrix.dot(desired_obj_wrench)
    if robot == self.kukaId_A:
      return wrench[:6]
    else:
      return wrench[6:]
  
  def ComputeDesiredObjectWrench(self, p):
    Kp = 0.6 * np.array([10, 10, 12, 3.5, 0.1, 0.1])
    Kv = 0.5 * np.array([1.2, 1.2, 1.5, 0.2, 0.1, 0.1])
    obj_pose_error = [0.0]*6
    for i in range(len(obj_pose_error)):
      obj_pose_error[i] = self.desired_obj_pose[i] - (self.env_state["object_pose"])[i]
    for i in range(3):
      obj_pose_error[i + 3] = math.fmod(obj_pose_error[i+3] + math.pi + 2*math.pi, 2*math.pi) - math.pi

    obj_vel_error = self.env_state["object_velocity"]
    for i in range(len(obj_vel_error)):
      obj_vel_error[i] = -obj_vel_error[i]
    obj_mass_matrix, obj_coriolis_vector, obj_gravity_vector = self.getObjectDynamics(p)
    obj_mass_matrix = np.eye(6)
    desired_obj_wrench = obj_mass_matrix.dot(Kp * obj_pose_error + Kv * obj_vel_error) + obj_coriolis_vector + np.array(obj_gravity_vector)
    return desired_obj_wrench
  
  def ComputeGraspMatrix(self, p):
    # TODO(VICTOR): Need to convert rp_A and rp_B to center of mass frame.
    rp_A = self.env_state["robot_A_ee_pose_obj_frame"]
    rp_B = self.env_state["robot_B_ee_pose_obj_frame"]
    top_three_rows = np.hstack((np.eye(3), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))))
    bottom_three_rows = np.hstack((self.skew(rp_A), np.eye(3), self.skew(rp_B), np.eye(3)))
    grasp_matrix = np.vstack((top_three_rows, bottom_three_rows))
    return grasp_matrix

  def skew(self, vector): 
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])
  
  def getObjectDynamics(self, p):
    dynamics_info = p.getDynamicsInfo(self.grasped_object, -1)
    obj_mass = dynamics_info[0]
    obj_inertia_vector = list(dynamics_info[2])
    obj_inertia_matrix = np.diag(obj_inertia_vector) # needs to be transformed
    obj_inertia_matrix = ((self.env_state["object_orn_matrix"]).dot(obj_inertia_matrix)).dot((self.env_state["object_orn_matrix"]).T)
    mass_matrix_top_row = np.hstack((obj_mass * np.eye(3), np.zeros((3, 3))))
    mass_matrix_bottom_row = np.hstack((np.zeros((3, 3)), obj_inertia_matrix))
    obj_mass_matrix = np.vstack((mass_matrix_top_row, mass_matrix_bottom_row))
    body_omega = (self.env_state["object_orn_matrix"]).dot((self.env_state["object_velocity"])[3:])
    body_omega_transformed = (self.skew(body_omega).dot(obj_inertia_matrix)).dot(body_omega)
    obj_coriolis_vector = np.hstack((np.zeros(3), body_omega_transformed))
    obj_gravity_vector = obj_mass * np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
    return obj_mass_matrix, obj_coriolis_vector, obj_gravity_vector