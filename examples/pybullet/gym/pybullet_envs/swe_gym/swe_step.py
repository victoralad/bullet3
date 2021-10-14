import time
import math
import numpy as np

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from va_reset import ResetCoopEnv

class StepCoopEnv(ResetCoopEnv):

  def __init__(self, robots, grasped_object, ft_id, desired_obj_pose, p):
    self.robot_A = robots[0]
    self.robot_B = robots[1]
    self.kukaEndEffectorIndex = 11
    self.totalNumJoints = p.getNumJoints(self.robot_A)
    self.numJoints = 7 # number of joints for just the arm
    self.jd = [0.01] * self.totalNumJoints
    self.useSimulation = 1
    self.ft_id = ft_id # F/T sensor joint id.
    self.grasped_object = grasped_object
    self.desired_obj_pose = desired_obj_pose
    self.constraint_set = False
    self.ee_constraint = 0
    self.ee_constraint_reward = 0 # This helps ensure that the grasp constraint is not violated.


  def apply_action(self, action, p):
    assert len(action) == 12
    computed_joint_torques_robot_A = action[:6]
    computed_joint_torques_robot_B = self.GetJointTorques(self.robot_B, p)
    
    if (self.useSimulation):
      for i in range(200):
        for i in range(self.numJoints):
          p.setJointMotorControl2(self.kukaId, i, p.VELOCITY_CONTROL, force=0.5)
          p.setJointMotorControl2(bodyIndex=self.robot_A,
                                jointIndex=i,
                                controlMode=p.TORQUE_CONTROL,
                                force=computed_joint_torques_robot_A[i])

          p.setJointMotorControl2(self.kukaId, i, p.VELOCITY_CONTROL, force=0.5)
          p.setJointMotorControl2(bodyIndex=self.robot_B,
                                jointIndex=i,
                                controlMode=p.TORQUE_CONTROL,
                                force=computed_joint_torques_robot_B[i])
        p.stepSimulation()
  
  def GetObservation(self, p):
    # ----------------------------- Get model input ----------------------------------
    obj_pose_error = [None] * 6
    wrench_A = [None] * 6
    wrench_B = [None] * 6

    # Get object pose
    obj_pose = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose[0]) + list(p.getEulerFromQuaternion(obj_pose[1]))
    for i in range(len(obj_pose)):
      obj_pose_error[i] = self.desired_obj_pose[i] - obj_pose[i]

    # Get Wrench measurements at wrist
    _, _, ft_A, self.applied_ft_A = p.getJointState(self.robot_A, self.ft_id)
    _, _, ft_B, self.applied_ft_B = p.getJointState(self.robot_B, self.ft_id)
    wrench_A = list(ft_A)
    wrench_B = list(ft_B)

    self.model_input = wrench_A + wrench_B + obj_pose_error
    normed_wrench = self.model_input[:12] / np.linalg.norm(self.model_input[:12])
    self.model_input[:12] = normed_wrench
    assert len(self.model_input) == 18
    return self.model_input
  
  def GetReward(self, p):
    reward = None
    u = np.array(self.model_input[-6:])
    Q = 100*np.eye(len(u))
    # Q[4][4] = 1000*Q[4][4]
    # Q[2][2] = 1000*Q[2][2]
    obj_pose_error_reward =  -1 * u.T @ (Q @ u)

    if not self.constraint_set:
      self.ee_constraint = self.GetConstraint(p)
      self.constraint_set = True
    curr_ee_constraint = self.GetConstraint(p)
    self.ee_constraint_reward = (curr_ee_constraint - self.ee_constraint)**2 # Squared constraint violation error
    ee_constr_reward = -self.ee_constraint_reward

    fI = np.array(self.model_input[:6]) - np.array(self.model_input[6:12]) # Internal stress = f_A - f_B. The computed value is wrong and must be corrected ASAP.
    R = np.eye(len(fI))
    wrench_reward = -1 * fI.T @ (R @ fI)

    reward = obj_pose_error_reward + ee_constr_reward + wrench_reward
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
    robot_A_ee_state = p.getLinkState(self.robot_A, self.kukaEndEffectorIndex)
    robot_B_ee_state = p.getLinkState(self.robot_B, self.kukaEndEffectorIndex)
    robot_A_ee_pose = list(robot_A_ee_state[0]) + list(p.getEulerFromQuaternion(robot_A_ee_state[1]))
    robot_B_ee_pose = list(robot_B_ee_state[0]) + list(p.getEulerFromQuaternion(robot_B_ee_state[1]))
    ee_constraint = np.array(robot_A_ee_pose) - np.array(robot_B_ee_pose)
    # subroutine to handle situations where joint angles cross PI or -PI
    for i in range(3):
      ee_constraint[i+3] = math.fmod(ee_constraint[i+3] + math.pi + 2*math.pi, 2*math.pi) - math.pi
    norm_ee_constraint = np.linalg.norm(ee_constraint)
    return norm_ee_constraint

  def GetJointTorques(self, robot, p):
    # State of end effector.
    ee_state = p.getLinkState(robot, self.kukaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = ee_state

    # Get the joint and link state directly from Bullet.
    joints_pos, joints_vel, joints_torq = self.getJointStates(robot)

    # Get the Jacobians for the CoM of the end-effector link.
    # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    # The localPosition is always defined in terms of the link frame coordinates.
    
    zero_vec = [0.0] * len(joints_pos)
    jac_t, jac_r = p.calculateJacobian(robot, self.kukaEndEffectorIndex, frame_pos, joints_pos, zero_vec, zero_vec)
    
    jac = np.vstack((np.array(jac_t), np.array(jac_r)))
    nonlinear_forces = p.calculateInverseDynamics(self.kukaId, joints_pos, joints_vel, zero_vec)
    desired_ee_wrench = self.ComputeWrenchFromGraspMatrix(robot, p)
    desired_joint_torques = jac.T.dot(np.array(desired_ee_wrench)) + np.array(nonlinear_forces)
    return desired_joint_torques