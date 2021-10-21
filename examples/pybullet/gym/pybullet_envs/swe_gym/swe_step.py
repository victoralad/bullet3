import time
import math
import numpy as np

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from swe_reset import ResetCoopEnv

class StepCoopEnv(ResetCoopEnv):

  def __init__(self, robots, grasped_object, ft_id, desired_obj_pose, p):
    self.robotId_A = robots[0]
    self.robotId_B = robots[1]
    self.kukaEndEffectorIndex = 11
    self.totalNumJoints = p.getNumJoints(self.robotId_A)
    self.nDof = p.computeDofCount(self.robotId_A)
    self.numJoints = 7 # number of joints for just the arm
    self.jd = [0.01] * self.totalNumJoints
    self.useSimulation = 1
    self.ft_id = ft_id # F/T sensor joint id.
    self.grasped_object = grasped_object
    self.desired_obj_pose = desired_obj_pose
    self.constraint_set = False
    self.ee_constraint = 0
    self.ee_constraint_reward = 0 # This helps ensure that the grasp constraint is not violated.
    self.env_state = {}
    self.ComputeEnvState(p)


  def apply_action(self, action, p):
    assert len(action) == 12
    self.ComputeEnvState(p)
    computed_joint_torques_robot_A = self.GetJointTorques(self.robotId_A, action, p)
    computed_joint_torques_robot_B = self.GetJointTorques(self.robotId_B, action, p)
    
    if (self.useSimulation):
      for i in range(200):
        for i in range(self.numJoints):
          p.setJointMotorControl2(self.robotId_A, i, p.VELOCITY_CONTROL, force=0.5)
          p.setJointMotorControl2(bodyIndex=self.robotId_A,
                                jointIndex=i,
                                controlMode=p.TORQUE_CONTROL,
                                force=computed_joint_torques_robot_A[i])

          p.setJointMotorControl2(self.robotId_B, i, p.VELOCITY_CONTROL, force=0.5)
          p.setJointMotorControl2(bodyIndex=self.robotId_B,
                                jointIndex=i,
                                controlMode=p.TORQUE_CONTROL,
                                force=computed_joint_torques_robot_B[i])
        p.stepSimulation()
  
  def GetObservation(self, p):
    # ----------------------------- Get model input ----------------------------------
    self.model_input = []
    self.ComputeEnvState(p)
    self.model_input += self.ComputeWrenchFromGraspMatrix(self.robotId_A, p)
    self.model_input += self.ComputeWrenchFromGraspMatrix(self.robotId_B, p)
    self.model_input += self.env_state["measured_force_torque_A"]
    self.model_input += self.env_state["object_pose"]
    self.model_input += self.desired_obj_pose
    assert len(self.model_input) == 30
    return self.model_input
  
  def GetReward(self, p):
    reward = 0.0
    # u = np.array(self.model_input[-6:])
    # Q = 100*np.eye(len(u))
    # # Q[4][4] = 1000*Q[4][4]
    # # Q[2][2] = 1000*Q[2][2]
    # obj_pose_error_reward =  -1 * u.T @ (Q @ u)

    # if not self.constraint_set:
    #   self.ee_constraint = self.GetConstraint(p)
    #   self.constraint_set = True
    # curr_ee_constraint = self.GetConstraint(p)
    # self.ee_constraint_reward = (curr_ee_constraint - self.ee_constraint)**2 # Squared constraint violation error
    # ee_constr_reward = -self.ee_constraint_reward

    # fI = np.array(self.model_input[:6]) - np.array(self.model_input[6:12]) # Internal stress = f_A - f_B. The computed value is wrong and must be corrected ASAP.
    # R = np.eye(len(fI))
    # wrench_reward = -1 * fI.T @ (R @ fI)

    # reward = obj_pose_error_reward + ee_constr_reward + wrench_reward
    return reward
  
  def GetInfo(self, p):
    done = False
    info = {1: 'Still training'}
    obj_pose_error = self.model_input[-6:]
    norm = np.linalg.norm(obj_pose_error)
    if norm > 2.0 or self.ee_constraint_reward > 0.05:
      done = True
      info = {1 : 'The norm of the object pose error, {}, is significant enough to reset the training episode.'.format(norm),
              2 : 'The fixed grasp constraint has been violated by this much: {}'.format(self.ee_constraint_reward)}
    return done, info
  
  def GetConstraint(self, p):
    robot_A_ee_pose = self.env_state["robot_A_ee_pose"]
    robot_B_ee_pose = self.env_state["robot_B_ee_pose"]
    ee_constraint = np.array(robot_A_ee_pose) - np.array(robot_B_ee_pose)
    # subroutine to handle situations where joint angles cross PI or -PI
    for i in range(3):
      ee_constraint[i+3] = math.fmod(ee_constraint[i+3] + math.pi + 2*math.pi, 2*math.pi) - math.pi
    norm_ee_constraint = np.linalg.norm(ee_constraint)
    return norm_ee_constraint

  def GetJointTorques(self, robotId, action, p):
    # State of end effector.
    ee_state = p.getLinkState(robotId, self.kukaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = ee_state

    # Get the joint and link state directly from Bullet.
    joints_pos, joints_vel, joints_torq = self.getJointStates(robotId, p)

    # Get the Jacobians for the CoM of the end-effector link.
    # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    # The localPosition is always defined in terms of the link frame coordinates.
    
    zero_vec = [0.0] * len(joints_pos)
    jac_t, jac_r = p.calculateJacobian(robotId, self.kukaEndEffectorIndex, frame_pos, joints_pos, zero_vec, zero_vec)
    
    jac = np.vstack((np.array(jac_t), np.array(jac_r)))
    nonlinear_forces = p.calculateInverseDynamics(robotId, joints_pos, joints_vel, zero_vec)
    if robotId == 0:
      desired_ee_wrench = np.array(self.ComputeWrenchFromGraspMatrix(robotId, p)) + np.array(action[:6])
    else:
      desired_ee_wrench = self.ComputeWrenchFromGraspMatrix(robotId, p)
    desired_joint_torques = jac.T.dot(np.array(desired_ee_wrench)) + np.array(nonlinear_forces)
    return desired_joint_torques[:self.numJoints]
  
  def getJointStates(self, robotId, p):
    joint_states = p.getJointStates(robotId, range(self.nDof))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques
  
  def ComputeEnvState(self, p):
    # Get object pose & velocity
    obj_pose = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose[0]) + list(p.getEulerFromQuaternion(obj_pose[1]))
    obj_vel = p.getBaseVelocity(Self.grasped_object)
    obj_vel = list(obj_vel[1]) + list(obj_vel[1])

    # Compute the pose of both end effectors
    robot_A_ee_state = p.getLinkState(self.robotId_A, self.kukaEndEffectorIndex)
    robot_B_ee_state = p.getLinkState(self.robotId_B, self.kukaEndEffectorIndex)
    robot_A_ee_pose = list(robot_A_ee_state[0]) + list(p.getEulerFromQuaternion(robot_A_ee_state[1]))
    robot_B_ee_pose = list(robot_B_ee_state[0]) + list(p.getEulerFromQuaternion(robot_B_ee_state[1]))

    # Get Wrench measurements at wrist
    _, _, ft_A, _ = p.getJointState(self.robotId_A, self.ft_id)
    _, _, ft_B, _ = p.getJointState(self.robotId_B, self.ft_id)
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
    if robot == self.robotId_A:
      return wrench[:6]
    else:
      return wrench[6:]
  
  def ComputeDesiredObjectWrench(self, p):
    Kp = 5.5 * np.array([5, 5, 5, 2, 2, 2])
    Kv = 5.2 * np.array([1.5, 1.5, 1.5, 0.2, 0.2, 0.2])
    # State of object.
    obj_state = p.getBasePositionAndOrientation(self.grasped_object)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = ee_state
    obj_pose_error = self.desired_obj_pose - self.env_state["object_pose"]
    obj_vel_error = -self.env_state["object_velocity"]
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