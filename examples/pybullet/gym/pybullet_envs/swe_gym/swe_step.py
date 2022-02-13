import time
import math
import numpy as np
import copy

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
    self.desired_obj_wrench = None
    self.desired_eeA_wrench = None
    self.desired_eeB_wrench = None
    self.action = None
    self.grasp_matrix = None
    self.mean_dist = [0.0]*6
    cov_dist_vec = [0.08]*6
    self.cov_dist = np.diag(cov_dist_vec)
    self.horizon = 9000
    self.env_state = {}
    self.ComputeEnvState(p)
    self.antag_joint_pos = np.load('antagonist/data/11_joints.npy')
    self.antag_data_idx = 0
    self.time_mod = 0.0 # This enables the simulation trajectory to match the teleoperated trajectory for the antagonist.
    self.hard_to_sim_ratio = 10
    self.interpol_pos = self.antag_joint_pos[self.antag_data_idx]
    self.reset_eps = False
    self.use_hard_data = False

    self.obj_pose_error = [0.0] * 6
    self.obj_pose_error_norm = 0.0
    self.axis = 2
    self.robotA_base = p.getBasePositionAndOrientation(self.robotId_A)
    self.robotB_base = p.getBasePositionAndOrientation(self.robotId_B)

    self.prev_obj_pose = [0, 0, 0]
    self.hasPrevPose = 1


    # p.setRealTimeSimulation(1)


  def apply_action(self, action, p):
    assert len(action) == 1
    self.action = np.array(action)
    self.ComputeEnvState(p)
    computed_joint_torques_robot_A = self.GetJointTorques(self.robotId_A, action, p)
    computed_joint_torques_robot_B = self.GetJointTorques(self.robotId_B, action, p)
    
    if self.use_hard_data:
      if (self.useSimulation):
        for i in range(1):
          for i in range(self.numJoints):
            p.setJointMotorControl2(self.robotId_A, i, p.VELOCITY_CONTROL, force=0.02)
            p.setJointMotorControl2(bodyIndex=self.robotId_A,
                                  jointIndex=i,
                                  controlMode=p.TORQUE_CONTROL,
                                  force=computed_joint_torques_robot_A[i])

            p.setJointMotorControl2(bodyIndex=self.robotId_B,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.interpol_pos[i],
                                    targetVelocity=0,
                                    force=100,
                                    positionGain=0.1,
                                    velocityGain=0.5,
                                    maxVelocity=0.01)
          p.stepSimulation()
        if self.antag_data_idx < len(self.antag_joint_pos) - 2:
          if self.time_mod < self.hard_to_sim_ratio:
            self.time_mod += 1.0
            interpos_ratio = self.time_mod / self.hard_to_sim_ratio
            scaled_pos = interpos_ratio * (self.antag_joint_pos[self.antag_data_idx + 1] - self.antag_joint_pos[self.antag_data_idx])
            self.interpol_pos = self.antag_joint_pos[self.antag_data_idx] + scaled_pos
            # print("***************************")
            # print(self.time_mod)
            # print(self.antag_joint_pos[self.antag_data_idx])
            # print(self.antag_joint_pos[self.antag_data_idx + 1])
            # print(self.interpol_pos)
          else:
            self.antag_data_idx += 1
            self.time_mod = 0
        # if self.antag_data_idx < len(self.antag_joint_pos) - 1:
        #   self.antag_data_idx += 1
        if self.reset_eps:
          self.antag_data_idx = 0

    else:
      if (self.useSimulation):
        for _ in range(1):
          p.setJointMotorControlArray(self.robotId_A, list(range(self.numJoints)), p.VELOCITY_CONTROL, forces=[0.02]*self.numJoints)
          p.setJointMotorControlArray(bodyIndex=self.robotId_A,
                             jointIndices=list(range(self.numJoints)),
                             controlMode=p.TORQUE_CONTROL,
                             forces=computed_joint_torques_robot_A)
          
          p.setJointMotorControlArray(self.robotId_B, list(range(self.numJoints)), p.VELOCITY_CONTROL, forces=[0.02]*self.numJoints)
          p.setJointMotorControlArray(bodyIndex=self.robotId_B,
                             jointIndices=list(range(self.numJoints)),
                             controlMode=p.TORQUE_CONTROL,
                             forces=computed_joint_torques_robot_B)
          p.stepSimulation()
  
  def GetObservation(self, p):
    # ----------------------------- Get model input ----------------------------------
    self.model_input = []
    # self.ComputeEnvState(p)
    self.model_input = np.append(self.model_input, self.desired_eeA_wrench)
    self.model_input = np.append(self.model_input, self.desired_eeB_wrench)
    self.model_input = np.append(self.model_input, np.array(self.env_state["measured_force_torque_A"]))
    self.model_input = np.append(self.model_input, np.array(self.env_state["object_pose"]))
    self.model_input = np.append(self.model_input, np.array(self.desired_obj_pose))
    self.model_input = np.append(self.model_input, np.array(self.env_state["robot_A_ee_pose"]))

    if (self.hasPrevPose):
      #self.trailDuration is duration (in seconds) after debug lines will be removed automatically
      #use 0 for no-removal
      trailDuration = 10000
      p.addUserDebugLine((self.desired_obj_pose)[:3], (self.env_state["object_pose"])[:3], [0.8, 0, 0.8], 2, trailDuration)

      self.hasPrevPose = 0

    assert len(self.model_input) == 36
    return self.model_input
  
  def GetReward(self, p, num_steps):
    reward = 0.0

    # Reward to force the agent to try to maintain a rigid grasp. @Alberta, you may ignore this when computing the overall reward.
    if not self.constraint_set:
      self.ee_constraint = self.GetConstraint(p)
      self.constraint_set = True
    curr_ee_constraint = self.GetConstraint(p)
    self.ee_constraint_reward = (curr_ee_constraint - self.ee_constraint)**2 # Squared constraint violation error
    ee_constr_reward = -self.ee_constraint_reward

    # Get pose error of the bar and done condition
    self.obj_pose_error_norm = min(np.linalg.norm(self.obj_pose_error[self.axis]), 2.0)

    # Rewards
    obj_pose_error_norm_reward = -self.obj_pose_error_norm
    
    scale_factor = 0.05
    episode_len_reward = 0.05*num_steps
    
    terminal_reward_final_error = 0.0
    if num_steps > self.horizon:
      terminal_reward_final_error = -5.0*self.obj_pose_error_norm
    
    terminal_reward_full_horizon = 0.0
    if num_steps > self.horizon:
      terminal_reward_full_horizon = 5.0

    reward = 0.0 + obj_pose_error_norm_reward + episode_len_reward + terminal_reward_final_error + terminal_reward_full_horizon
    return reward, self.obj_pose_error_norm

  def GetPoseError(self):
    obj_pose_error = [0.0] * 6
    for i in range(len(obj_pose_error)):
      obj_pose_error[i] = self.desired_obj_pose[i] - (self.env_state["object_pose"])[i]
    for i in range(3):
      obj_pose_error[i + 3] = math.fmod(obj_pose_error[i+3] + math.pi + 2*math.pi, 2*math.pi) - math.pi
    # print("******************************")
    # print(self.env_state["object_pose"])
    # print(obj_pose_error)
    # while(1):
    #   a = 1
    return obj_pose_error

  def CheckDone(self, norm, num_steps):

    done = False
    info = {0: 'Still training'}

    if num_steps > self.horizon:
      done = True
      info = {1: 'Episode completed successfully.'}
    # elif norm > 10.0 and self.ee_constraint_reward > 0.05:
    #   done = True
    #   info = {2: 'The norm of the object pose error, {}, is significant enough to reset the training episode.'.format(norm),
    #           3: 'The fixed grasp constraint has been violated by this much: {}'.format(self.ee_constraint_reward)}
    elif norm > 10.0:
      done = True
      info = {2: 'The norm of the object pose error, {}, is significant enough to reset the training episode.'.format(norm)}
    # elif self.ee_constraint_reward > 0.05:
    #   done = True
    #   info = {3: 'The fixed grasp constraint has been violated by this much: {}'.format(self.ee_constraint_reward)}
    
    self.reset_eps = done
    return done, info

  def GetInfo(self, p, num_steps):
    done, info = self.CheckDone(self.obj_pose_error_norm, num_steps)

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
    jac = jac[:, :7]
    nonlinear_forces = p.calculateInverseDynamics(robotId, joints_pos, zero_vec, zero_vec)
    nonlinear_forces = nonlinear_forces[:7]
    if robotId == self.robotId_A:
      self.ComputeWrenchFromGraspMatrix(p)
      desired_ee_wrench = self.desired_eeA_wrench# + np.array(action[:6])
      # desired_ee_wrench = np.zeros((6,)) #self.desired_eeA_wrench
      # desired_ee_wrench[self.axis] = action[0]
    else:
      disturbance = np.random.multivariate_normal(self.mean_dist, self.cov_dist)
      desired_ee_wrench = self.desired_eeB_wrench# + disturbance
    # robot_inertia_matrix = np.array(p.calculateMassMatrix(robotId, joints_pos))
    # robot_inertia_matrix = robot_inertia_matrix[:7, :7]
    # dyn_ctnt_inv = np.linalg.inv(jac.dot(robot_inertia_matrix.dot(jac.T)))
    # dyn_ctnt_inv = np.eye(6)
    desired_joint_torques = (jac.T).dot(desired_ee_wrench) + np.array(nonlinear_forces)
    return desired_joint_torques[:self.numJoints]
  
  def getJointStates(self, robotId, p):
    joint_states = p.getJointStates(robotId, range(self.nDof))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques
  
  def ComputeEnvState(self, p):
    # Get object pose & velocity
    obj_pose_state = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose_state[0]) + list(p.getEulerFromQuaternion(obj_pose_state[1]))
    # print("****************************")
    # print(obj_pose)
    # while(1):
    #   a = 1
    obj_vel = p.getBaseVelocity(self.grasped_object)
    obj_vel = list(obj_vel[1]) + list(obj_vel[1])

    # Compute the pose of both end effectors in the world freame.
    robot_A_ee_state = p.getLinkState(self.robotId_A, self.kukaEndEffectorIndex)
    robot_B_ee_state = p.getLinkState(self.robotId_B, self.kukaEndEffectorIndex)
    robot_A_ee_pose = list(robot_A_ee_state[0]) + list(p.getEulerFromQuaternion(robot_A_ee_state[1]))
    robot_B_ee_pose = list(robot_B_ee_state[0]) + list(p.getEulerFromQuaternion(robot_B_ee_state[1]))

    # Compute the pose of both end effectors in the object's frame.
    obj_orn_matrix = np.array(p.getMatrixFromQuaternion(obj_pose_state[1]))
    obj_orn_matrix = np.reshape(obj_orn_matrix, (3, 3))
    robot_A_ee_pose_obj_frame = np.linalg.inv(obj_orn_matrix).dot((np.array(robot_A_ee_state[0]) - np.array(obj_pose_state[0])))
    robot_B_ee_pose_obj_frame = np.linalg.inv(obj_orn_matrix).dot((np.array(robot_B_ee_state[0]) - np.array(obj_pose_state[0])))

    # Get Wrench measurements at wrist
    _, _, ft_A, _ = p.getJointState(self.robotId_A, self.ft_id)
    _, _, ft_B, _ = p.getJointState(self.robotId_B, self.ft_id)
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
    
  def ComputeWrenchFromGraspMatrix(self, p):
    # TODO (Victor): compute F_T = G_inv * F_o
    desired_obj_wrench = self.ComputeDesiredObjectWrench(p)
    grasp_matrix = self.ComputeGraspMatrix(p)
    inv_grasp_matrix = np.linalg.pinv(grasp_matrix)
    wrench = inv_grasp_matrix.dot(desired_obj_wrench)
    assert len(wrench) == 12
    
    # self.desired_eeA_wrench = np.zeros((6,))
    # self.desired_eeA_wrench[self.axis] = self.action[0] 
    self.desired_eeA_wrench = wrench[:6] #self.ToBaseFrame(wrench[:6], "robotA", p)
    self.desired_eeB_wrench = wrench[6:] #self.ToBaseFrame(wrench[6:], "robotB", p)
  
  def ToBaseFrame(self, wrench, robotId, p):
    if robotId == "robotA":
      base_disp_vec = np.array((self.robotA_base)[0])
    elif robotId == "robotB":
      base_disp_vec = np.array((self.robotB_base)[0])
    top_three_rows = np.hstack((np.eye(3), np.zeros((3, 3))))
    bottom_three_rows = np.hstack((self.skew(base_disp_vec), np.eye(3)))
    rotWorldToBase = np.vstack((top_three_rows, bottom_three_rows))
    transformed_wrench = rotWorldToBase.dot(np.array(wrench))
    # print(rotWorldToBase)
    # print(wrench)
    # print(transformed_wrench)
    # quit()
    return transformed_wrench
  
  def ComputeDesiredObjectWrench(self, p):
    Kp = 5.6 * np.array([2, 2, 2, 1.5, 10.5, 1.5])
    Kv = 0.2 * np.array([1.2, 1.2, 2.5, 0.1, 0.2, 0.1])
    # Kp = 5.6 * np.array([2, 2, 2, 10.5, 1.5, 1.5])
    # Kv = 0.2 * np.array([1.2, 1.2, 2.5, 0.2, 0.1, 0.1])
    self.obj_pose_error = self.GetPoseError()
    obj_vel_error = self.env_state["object_velocity"]
    for i in range(len(obj_vel_error)):
      obj_vel_error[i] = -obj_vel_error[i]
    obj_mass_matrix, obj_coriolis_vector, obj_gravity_vector = self.getObjectDynamics(p)
    obj_mass_matrix = np.eye(6)
    desired_obj_wrench = obj_mass_matrix.dot(Kp * self.obj_pose_error + Kv * obj_vel_error) + obj_coriolis_vector + obj_gravity_vector
    # desired_obj_wrench = Kp * self.obj_pose_error + Kv * obj_vel_error + obj_gravity_vector
    # desired_obj_wrench = obj_gravity_vector
    # print("#####################")
    # print(self.env_state["object_pose"])
    # print(obj_gravity_vector)
    # quit()
    return desired_obj_wrench
  
  def ComputeGraspMatrix(self, p):
    rp_A = self.env_state["robot_A_ee_pose_obj_frame"]
    rp_B = self.env_state["robot_B_ee_pose_obj_frame"]
    top_three_rows = np.hstack((np.eye(3), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))))
    bottom_three_rows = np.hstack((self.skew(rp_A), np.eye(3), self.skew(rp_B), np.eye(3)))
    grasp_matrix = np.vstack((top_three_rows, bottom_three_rows))
    self.grasp_matrix = grasp_matrix
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
    body_omega = (self.env_state["object_orn_matrix"]).T.dot((self.env_state["object_velocity"])[3:])
    body_omega_transformed = (self.skew(body_omega).dot(obj_inertia_matrix)).dot(body_omega)
    obj_coriolis_vector = np.hstack((np.zeros(3), body_omega_transformed))
    obj_gravity_vector = obj_mass * np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
    return obj_mass_matrix, obj_coriolis_vector, obj_gravity_vector