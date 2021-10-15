import time
import math

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

from swe_init import InitCoopEnv

class ResetCoopEnv(InitCoopEnv):

  def __init__(self, desired_obj_pose, p):
    super().__init__(p)
    self.desired_obj_pose = desired_obj_pose

  def ResetCoop(self, p):
    # Reset the object to the grasp location
    p.resetBasePositionAndOrientation(self.grasped_object, [0, 0.7, 0.02], p.getQuaternionFromEuler([0, 0, 0]))
    # Reset the robots to a position where the grippers can grasp the object
    robot_A_reset = [-1.184011299413845, -1.4364158475353175, -1.0899721376131706, 1.0667906797236881, 1.2044237679252714, 
      -1.2280706100083119, 0.988134098323069, 0.0, 0.0, 0.0, 0.003781686634043995]
    
    robot_B_reset = [-1.0663547431079572, -1.4050373708258017, -1.017445912897535, 1.0546136723514878, 1.1190637184529868, 
      -1.2119703753287736, 1.157260237744829, 0.0, 0.0, 0.0, -0.029547676578768497]

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
    self.model_input = []
    env_state = self.GetEnvState(p)
    self.model_input.append(env_state["grasp_matrix_force_torque_A"])
    self.model_input.append(env_state["grasp_matrix_force_torque_B"])
    self.model_input.append(env_state["measured_force_torque_A"])
    self.model_input.append(env_state["object_pose"])
    self.model_input.append(env_state["desired_object_pose"])
    assert len(self.model_input) == 30
    return self.model_input
  
  def GetEnvState(self, p):
    env_state = {}

    # Get object pose
    obj_pose = p.getBasePositionAndOrientation(self.grasped_object)
    obj_pose = list(obj_pose[0]) + list(p.getEulerFromQuaternion(obj_pose[1]))

    # Get Wrench measurements at wrist
    _, _, ft_A, _ = p.getJointState(self.kukaId_A, self.ft_id)
    _, _, ft_B, _ = p.getJointState(self.kukaId_B, self.ft_id)
    force_torque_A = list(ft_A)
    force_torque_B = list(ft_B)

    env_state["measured_force_torque_A"] = force_torque_A
    env_state["measured_force_torque_B"] = force_torque_B
    env_state["grasp_matrix_force_torque_A"] = self.ComputeWrenchFromGraspMatrix(self.kukaId_A, p)
    env_state["grasp_matrix_force_torque_B"] = self.ComputeWrenchFromGraspMatrix(self.kukaId_B, p)
    env_state["object_pose"] = obj_pose
    env_state["desired_object_pose"] = self.desired_obj_pose

  def ComputeWrenchFromGraspMatrix(self, robot, p):
    # TODO (Victor): compute F_T = G_inv * F_o
    wrench = [0.0] * 12
    if robot == self.kukaId_A:
      return wrench[:6]
    else:
      return wrench[6:]
  
  def ComputeDesiredObjectWrench(self, p):
    pass