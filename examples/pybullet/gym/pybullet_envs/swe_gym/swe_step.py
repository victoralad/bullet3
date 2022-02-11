import time
import math
import numpy as np
import copy

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from swe_reset import ResetCoopEnv

class StepCoopEnv(ResetCoopEnv):

  def __init__(self, robots, grasped_object, ft_id, desired_eeA_pose, p):
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
    self.desired_eeA_pose = desired_eeA_pose
    self.constraint_set = False
    self.ee_constraint = 0
    self.ee_constraint_reward = 0 # This helps ensure that the grasp constraint is not violated.
    self.desired_obj_wrench = None
    self.desired_eeA_wrench = None
    self.desired_eeB_wrench = None
    self.action = None
    self.grasp_matrix = None
    self.mean_dist = [0.0]*6
    cov_dist_vec = [0.08]*6
    self.cov_dist = np.diag(cov_dist_vec)
    self.terminal_reward = 0.0
    self.horizon = 20000
    self.env_state = {}
    self.ComputeEnvState(p)
    self.antag_joint_pos = np.load('antagonist/data/12_joints.npy')
    self.antag_data_idx = 0
    self.time_mod = 0.0 # This enables the simulation trajectory to match the teleoperated trajectory for the antagonist.
    self.hard_to_sim_ratio = 10
    self.interpol_pos = self.antag_joint_pos[self.antag_data_idx]
    self.reset_eps = False
    self.use_hard_data = False

    self.prev_obj_pose = [0, 0, 0]
    self.hasPrevPose = 1

    self.eeA_pose_error = None
    self.eeA_pose_error_norm = None
    self.prev_eeA_pos = np.zeros((6,))
    self.standard_control = True
    self.obj_pose_error = None
    self.desired_obj_pose = desired_eeA_pose

    # p.setRealTimeSimulation(1)


  def apply_action(self, action, p):
    assert len(action) == 6
    self.action = np.array(action)
    self.ComputeEnvState(p)
    computed_joint_torques_robot_A = self.GetJointTorques(self.robotId_A, action, p)
    # computed_joint_torques_robot_B = self.GetJointTorques(self.robotId_B, action, p)
    
    if (self.useSimulation):
      for _ in range(1):
        p.setJointMotorControlArray(self.robotId_A, list(range(self.numJoints)), p.VELOCITY_CONTROL, forces=[0.02]*self.numJoints)
        p.setJointMotorControlArray(bodyIndex=self.robotId_A,
                              jointIndices=list(range(self.numJoints)),
                              controlMode=p.TORQUE_CONTROL,
                              forces=computed_joint_torques_robot_A)

        p.stepSimulation()
  
  def GetObservation(self, p):
    # ----------------------------- Get model input ----------------------------------
    self.model_input = []
    self.model_input = np.append(self.model_input, np.array(self.desired_eeA_pose))
    self.model_input = np.append(self.model_input, np.array(self.env_state["object_pose"]))

    if (self.hasPrevPose):
      #self.trailDuration is duration (in seconds) after debug lines will be removed automatically
      #use 0 for no-removal
      trailDuration = 10000
      p.addUserDebugLine((self.desired_eeA_pose)[:3], (self.env_state["robot_A_ee_pose"])[:3], [0.8, 0, 0.8], 2, trailDuration)
      # p.addUserDebugLine(self.prevPose1_A, ls_A[4], [1, 0, 0], 1, trailDuration)
      # self.prev_obj_pose = (self.env_state["object_pose"])[:3]
      # self.prevPose1_A = ls_A[4]
      self.hasPrevPose = 0

    assert len(self.model_input) == 12
    return self.model_input
  
  def GetReward(self, p, num_steps):
    reward = 0.0
    

    # Get pose error of the bar and done condition
    self.eeA_pose_error = self.GetPoseError()
    self.eeA_pose_error_norm = np.linalg.norm(self.eeA_pose_error) #min(np.linalg.norm(self.eeA_pose_error), 2.0)
    eeA_pose_error_norm_reward = -1.0*self.eeA_pose_error_norm

    # reward to penalize high velocities
    velocity = np.array(self.env_state["robot_A_ee_vel"])
    velocity_norm = np.linalg.norm(velocity)
    velocity_norm_reward = -0.1*velocity_norm

    # terminal OPEN reward
    terminal_eeA_reward = 0.0
    if num_steps > self.horizon:
      terminal_eeA_reward = -2.0*self.eeA_pose_error_norm**2

    reward = 20.0 + eeA_pose_error_norm_reward + velocity_norm_reward + terminal_eeA_reward
    return reward, self.eeA_pose_error_norm

  def GetPoseError(self):
    eeA_pose_error = [0.0] * 6
    for i in range(len(eeA_pose_error)):
      eeA_pose_error[i] = self.desired_eeA_pose[i] - (self.env_state["robot_A_ee_pose"])[i]
    for i in range(3):
      eeA_pose_error[i + 3] = self.desired_eeA_pose[i+3] - (self.env_state["robot_A_ee_pose"])[i+3]
      eeA_pose_error[i + 3] = math.fmod(eeA_pose_error[i+3] + math.pi + 2*math.pi, 2*math.pi) - math.pi

    return eeA_pose_error

  def CheckDone(self, norm, num_steps):

    done = False
    info = {0: 'Still training'}

    if num_steps > self.horizon:
      done = True
      info = {1: 'Episode completed successfully.'}
    # elif norm > 2.0 and self.ee_constraint_reward > 0.1:
    #   done = True
    #   info = {2: 'The norm of the object pose error, {}, is significant enough to reset the training episode.'.format(norm),
    #           3: 'The fixed grasp constraint has been violated by this much: {}'.format(self.ee_constraint_reward)}
    # elif norm > 2.0:
    #   done = True
    #   info = {2: 'The norm of the object pose error, {}, is significant enough to reset the training episode.'.format(norm)}
    # elif self.ee_constraint_reward > 0.1:
    #   done = True
    #   info = {3: 'The fixed grasp constraint has been violated by this much: {}'.format(self.ee_constraint_reward)}
    
    self.reset_eps = done
    return done, info

  def GetInfo(self, p, num_steps):
    done, info = self.CheckDone(self.eeA_pose_error_norm, num_steps)
    return done, info

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
    jac = jac[:, :7]
    nonlinear_forces = p.calculateInverseDynamics(robotId, joints_pos, zero_vec, zero_vec)
    nonlinear_forces = nonlinear_forces[:7]
    if self.standard_control:
      desired_ee_wrench = self.ComputeDesiredEEWrench(p)
      # desired_ee_wrench = np.zeros((6,))
      # desired_ee_wrench[2] = 0.681
      # print("###################")
      # print(desired_ee_wrench)
    else:
      desired_ee_wrench = np.array(action[:6])
    # desired_ee_wrench = np.array(action[:6])
    robot_inertia_matrix = np.array(p.calculateMassMatrix(robotId, joints_pos))
    robot_inertia_matrix = robot_inertia_matrix[:7, :7]
    # dyn_ctnt_inv = np.linalg.inv(jac.dot(robot_inertia_matrix.dot(jac.T)))
    dyn_ctnt_inv = np.eye(6)
    desired_joint_torques = (jac.T.dot(dyn_ctnt_inv)).dot(np.array(desired_ee_wrench)) + np.array(nonlinear_forces)
    # desired_joint_torques = np.zeros((7,))
    # desired_joint_torques = nonlinear_forces
    return desired_joint_torques[:self.numJoints]
  
  def getJointStates(self, robotId, p):
    joint_states = p.getJointStates(robotId, range(self.nDof))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques
  
  def ComputeEnvState(self, p):
    # Compute the pose of both end effectors in the world frame.
    robot_A_ee_state = p.getLinkState(self.robotId_A, self.kukaEndEffectorIndex, 1)
    robot_A_ee_pose = list(robot_A_ee_state[0]) + list(p.getEulerFromQuaternion(robot_A_ee_state[1]))
    robot_A_ee_vel = list(robot_A_ee_state[-2]) + list(robot_A_ee_state[-1])
    # Get object pose & velocity
    obj_pose_state = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose_state[0]) + list(p.getEulerFromQuaternion(obj_pose_state[1]))
    # print("****************************", obj_pose)
    obj_vel = p.getBaseVelocity(self.grasped_object)
    obj_vel = list(obj_vel[1]) + list(obj_vel[1])
    # Compute the pose of both end effectors in the object's frame.
    obj_orn_matrix = np.array(p.getMatrixFromQuaternion(obj_pose_state[1]))
    obj_orn_matrix = np.reshape(obj_orn_matrix, (3, 3))

    self.env_state["robot_A_ee_pose"] = robot_A_ee_pose
    self.env_state["robot_A_ee_vel"] = robot_A_ee_vel
    self.env_state["object_pose"] = obj_pose
    self.env_state["object_orn_matrix"] = obj_orn_matrix
    self.env_state["object_velocity"] = obj_vel
  
  
  def GetEnvState(self):
    return self.env_state
  
  def GetObjPoseError(self):
    obj_pose_error = [0.0] * 6
    for i in range(len(obj_pose_error)):
      obj_pose_error[i] = self.desired_obj_pose[i] - (self.env_state["object_pose"])[i]
    for i in range(3):
      obj_pose_error[i + 3] = self.desired_obj_pose[i+3] - (self.env_state["object_pose"])[i+3]
      obj_pose_error[i + 3] = math.fmod(obj_pose_error[i+3] + math.pi + 2*math.pi, 2*math.pi) - math.pi
    return obj_pose_error
  
  def ComputeDesiredEEWrench(self, p):
    Kp = 0.6 * np.array([5, 5, 5, 1.5, 1.5, 1.5])
    Kv = 0.2 * np.array([1.2, 1.2, 1.2, 0.1, 0.1, 0.1])
    self.obj_pose_error = np.array(self.GetObjPoseError())# + self.action
    obj_vel_error = self.env_state["object_velocity"]
    for i in range(len(obj_vel_error)):
      obj_vel_error[i] = -obj_vel_error[i]
    # desired_eeA_wrench = Kp * self.eeA_pose_error + Kv * eeA_vel_error
    obj_mass_matrix, obj_coriolis_vector, obj_gravity_vector = self.getObjectDynamics(p)
    obj_mass_matrix = np.eye(6)
    desired_eeA_wrench = obj_mass_matrix.dot(Kp * self.obj_pose_error + Kv * obj_vel_error) + obj_coriolis_vector + obj_gravity_vector
    # desired_eeA_wrench = obj_gravity_vector
    print(desired_eeA_wrench)
    # quit()
    return desired_eeA_wrench
  
  def getObjectDynamics(self, p):
    dynamics_info = p.getDynamicsInfo(self.grasped_object, -1)
    obj_mass = dynamics_info[0]
    obj_inertia_vector = list(dynamics_info[2])
    obj_inertia_matrix = np.diag(obj_inertia_vector) # needs to be transformed
    obj_inertia_matrix = ((self.env_state["object_orn_matrix"]).dot(obj_inertia_matrix)).dot((self.env_state["object_orn_matrix"]).T)
    mass_matrix_top_row = np.hstack((obj_mass * np.eye(3), np.zeros((3, 3))))
    mass_matrix_bottom_row = np.hstack((np.zeros((3, 3)), obj_inertia_matrix))
    obj_mass_matrix = np.vstack((mass_matrix_top_row, mass_matrix_bottom_row))
    body_omega = (self.env_state["object_orn_matrix"]).T.dot((self.env_state["object_velocity"])[3:])
    body_omega_transformed = (self.skew(body_omega).dot(obj_inertia_matrix)).dot(body_omega)
    obj_coriolis_vector = np.hstack((np.zeros(3), body_omega_transformed))
    # obj_gravity_vector = obj_mass * np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
    obj_gravity_vector = obj_mass_matrix.dot(np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0]))
    return obj_mass_matrix, obj_coriolis_vector, obj_gravity_vector

  def skew(self, vector): 
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])