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
    p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
    self.kukaId = p.loadURDF("va_kuka/va_iiwa_model.urdf", [0, 0, 0], useFixedBase=True)
    p.resetBasePositionAndOrientation(self.kukaId, [0, 0, 0], [0, 0, 0, 1])
    p.setGravity(0, 0, -9.81)
    self.kukaEndEffectorIndex = 7
    self.totalNumJoints = p.getNumJoints(self.kukaId)
    # joint damping coefficents
    self.jd = [0.01] * self.totalNumJoints
    # number of joints for just the arm
    self.numJoints = 7

    self.t = 0.
    self.useSimulation = 1
    self.useRealTimeSimulation = 0
    p.setRealTimeSimulation(self.useRealTimeSimulation)

    self.prevPose = [0, 0, 0]
    self.prevPose1 = [0, 0, 0]
    self.hasPrevPose = 0

    logId1 = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "LOG0001.txt", [0, 1, 2])
    logId2 = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "LOG0002.txt", bodyUniqueIdA=2)
    
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

    pos = [0.2 * math.cos(self.t), 0.2 * math.sin(self.t), 0.4]
    #end effector points down, not up (when orientation is used)
    orn_euler = [0, -math.pi, 0]
    orn = p.getQuaternionFromEuler(orn_euler)

    # jacobian = p.calculateJacobian(self.kukaId, self.kukaEndEffectorIndex, )

    jointPoses = p.calculateInverseKinematics(self.kukaId,
                                              self.kukaEndEffectorIndex,
                                              pos,
                                              orn,
                                              jointDamping=self.jd)
    
    if (self.useSimulation):
      for i in range(self.numJoints):
        p.setJointMotorControl2(bodyIndex=self.kukaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(self.numJoints):
        p.resetJointState(self.kukaId, i, jointPoses[i])
    
    ls = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)

    if (self.hasPrevPose):
      #self.trailDuration is duration (in seconds) after debug lines will be removed automatically
      #use 0 for no-removal
      trailDuration = 15
      p.addUserDebugLine(self.prevPose, pos, [0, 0, 0.3], 1, trailDuration)
      p.addUserDebugLine(self.prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    self.prevPose = pos
    self.prevPose1 = ls[4]
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


  def gripper(self, gripper_opening_length, mode=p.POSITION_CONTROL):
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

        p.setJointMotorControl2(self.kukaId,
                                self.joints[gripper_main_control_joint_name].id,
                                p.POSITION_CONTROL,
                                targetPosition=gripper_opening_angle,
                                force=self.joints[gripper_main_control_joint_name].maxForce,
                                maxVelocity=self.joints[gripper_main_control_joint_name].maxVelocity)
        for i in range(len(mimic_joint_name)):
            joint = self.joints[mimic_joint_name[i]]
            p.setJointMotorControl2(self.kukaId, joint.id, p.POSITION_CONTROL,
                                    targetPosition=gripper_opening_angle * mimic_multiplier[i],
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)



if __name__ == '__main__':
  iiwa = ObjDyn()
  while 1:
    iiwa.Run()
# p.disconnect()