import pybullet as p
import pybullet_data

import time
import math
import numpy as np

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

#clid = p.connect(p.SHARED_MEMORY)

class ObjDyn:

  def __init__(self):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    self.kukaId = p.loadURDF("franka_panda/panda.urdf", [-0.3, 0, 0], useFixedBase=True)
    p.resetBasePositionAndOrientation(self.kukaId, [0, 0, 0], [0, 0, 0, 1])
    p.setGravity(0, 0, -9.81)
    self.kukaEndEffectorIndex = 7
    self.totalNumJoints = p.getNumJoints(self.kukaId)
    # joint damping coefficents
    self.jd = [0.01] * self.totalNumJoints
    # number of joints for just the arm
    self.numJoints = 7
    # num of degrees-of-freedom. nDof = totalNumJoints - numFixedJoints
    self.nDof = p.computeDofCount(self.kukaId)

    # initJointPos = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    initJointPos = [0, 0, -0.5 * math.pi, 0.5 * math.pi, math.pi, math.pi * 0.5, 0.8 * math.pi]
    for i in range(self.numJoints):
      p.resetJointState(self.kukaId, i, initJointPos[i])

    self.t = 0.
    self.useSimulation = 1
    self.useRealTimeSimulation = 0
    p.setRealTimeSimulation(self.useRealTimeSimulation)

    self.Kp = 10.5 * np.array([5, 5, 5, 2, 2, 2])
    self.Kv = 10.2 * np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])

    self.prevPose = [0, 0, 0]
    self.prevPose1 = [0, 0, 0]
    self.hasPrevPose = 0

    
    self.joints = AttrDict()
    jointInfo = namedtuple("jointInfo", ["id","name","lowerLimit","upperLimit","maxForce","maxVelocity"])

    # get jointInfo and index of dummy_center_indicator_link
    for i in range(self.totalNumJoints):
      info = p.getJointInfo(self.kukaId, i)
      jointID = info[0]
      jointName = info[1].decode("utf-8")
      jointLowerLimit = info[8]
      jointUpperLimit = info[9]
      jointMaxForce = info[10]
      jointMaxVelocity = info[11]
      singleInfo = jointInfo(jointID, jointName, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
      self.joints[singleInfo.name] = singleInfo

    for i in range(2):
      print("Body %d's name is %s." % (i, p.getBodyInfo(i)[1]))

  # --------------------------- Run Simulation --------------------------------
  def Run(self):
    if (self.useRealTimeSimulation):
      dt = datetime.now()
      self.t = (dt.second / 60.) * 2. * math.pi
    else:
      self.t += 0.1

    if (self.useSimulation and self.useRealTimeSimulation == 0):
      p.stepSimulation()

    desired_ee_pos = [-0.5, 0.0, 0.4] #[0.2 * math.cos(self.t), 0.2 * math.sin(self.t), 0.4]
    #end effector points down, not up (when orientation is used)
    desired_orn_euler = [0.1, -math.pi / 2, 0.1]

    # State of end effector.
    ee_state = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = ee_state

    # Get the joint and link state directly from Bullet.
    joints_pos, joints_vel, joints_torq = self.getJointStates(self.kukaId)

    # Get the Jacobians for the CoM of the end-effector link.
    # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    # The localPosition is always defined in terms of the link frame coordinates.
    
    zero_vec = [0.0] * len(joints_pos)
    jac_t, jac_r = p.calculateJacobian(self.kukaId, self.kukaEndEffectorIndex, frame_pos, joints_pos, zero_vec, zero_vec)
    
    jac = np.vstack((np.array(jac_t), np.array(jac_r)))
    ee_pose = np.concatenate((np.array(frame_pos), np.array(p.getEulerFromQuaternion(frame_rot))))
    desired_ee_pose = np.concatenate((np.array(desired_ee_pos), np.array(desired_orn_euler)))
    ee_vel = np.concatenate((np.array(link_vt), np.array(link_vr)))
    desired_ee_vel = np.zeros(len(ee_vel))

    ee_pose_error = desired_ee_pose - ee_pose
    ee_vel_error = desired_ee_vel - ee_vel

    desired_ee_wrench = self.Kp * ee_pose_error + self.Kv * ee_vel_error

    nonlinear_forces = p.calculateInverseDynamics(self.kukaId, joints_pos, joints_vel, zero_vec)
    desired_joint_torques = jac.T.dot(desired_ee_wrench) + np.array(nonlinear_forces)

    print("-----------------------")
    # print(ee_pose_error)
    for i in range(6):
      print(ee_pose_error[i])

    if (self.useSimulation):
      for i in range(self.numJoints):
        p.setJointMotorControl2(self.kukaId, i, p.VELOCITY_CONTROL, force=0.5)
        p.setJointMotorControl2(bodyIndex=self.kukaId,
                                jointIndex=i,
                                controlMode=p.TORQUE_CONTROL,
                                force=desired_joint_torques[i])
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(self.numJoints):
        p.resetJointState(self.kukaId, i, jointPoses[i])
    

    if (self.hasPrevPose):
      #self.trailDuration is duration (in seconds) after debug lines will be removed automatically
      #use 0 for no-removal
      trailDuration = 15
      p.addUserDebugLine(self.prevPose, desired_ee_pos, [0, 0, 0.3], 1, trailDuration)
      p.addUserDebugLine(self.prevPose1, ee_state[4], [1, 0, 0], 1, trailDuration)
    self.prevPose = desired_ee_pos
    self.prevPose1 = ee_state[4]
    self.hasPrevPose = 1

    keys = p.getKeyboardEvents()
    close_cmd = ord('c')
    open_cmd = ord('o')
    if close_cmd in keys:
      # print("Close gripper")
      self.gripper(0.0)
    elif open_cmd in keys:
      # print("Open gripper")
      self.gripper(0.085)

  def getJointStates(self, kukaId):
    joint_states = p.getJointStates(kukaId, range(self.nDof))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques

  def gripper(self, finger_target, mode=p.POSITION_CONTROL):
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

        for i in [9,10]:
          p.setJointMotorControl2(self.kukaId, i, mode, finger_target, force= 10)



if __name__ == '__main__':
  iiwa = ObjDyn()
  while 1:
    a = 1 # debug. Used to comment out the line below.
    iiwa.Run()
# p.disconnect()