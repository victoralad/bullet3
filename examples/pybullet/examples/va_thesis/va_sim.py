import pybullet as p
import time
import math
from datetime import datetime

#clid = p.connect(p.SHARED_MEMORY)
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
kukaId = p.loadURDF("va_kuka/va_iiwa_model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 7
totalNumJoints = p.getNumJoints(kukaId)


#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.01] * totalNumJoints
# number of joints for just the arm
numJoints = 7

for i in range(numJoints):
  p.resetJointState(kukaId, i, rp[i])

p.setGravity(0, 0, -9.81)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0

count = 0
useOrientation = 0
useSimulation = 1
useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15

logId1 = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "LOG0001.txt", [0, 1, 2])
logId2 = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "LOG0002.txt", bodyUniqueIdA=2)

for i in range(2):
  print("Body %d's name is %s." % (i, p.getBodyInfo(i)[1]))

while 1:
  if (useRealTimeSimulation):
    dt = datetime.now()
    t = (dt.second / 60.) * 2. * math.pi
  else:
    t = t + 0.1

  if (useSimulation and useRealTimeSimulation == 0):
    p.stepSimulation()

  for i in range(1):
    pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
    #end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul,
                                                  jr, rp)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  orn,
                                                  jointDamping=jd)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos)
    
    if (useSimulation):
      for i in range(numJoints):
        p.setJointMotorControl2(bodyIndex=kukaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(numJoints):
        p.resetJointState(kukaId, i, jointPoses[i])
  
  ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  if (hasPrevPose):
    p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  prevPose = pos
  prevPose1 = ls[4]
  hasPrevPose = 1

  keys = p.getKeyboardEvents()
  close_cmd = ord('c')
  open_cmd = ord('o')
  if close_cmd in keys:
    print("Close gripper")
  elif open_cmd in keys:
    print("Open gripper")


  # def gripper(self, cmd, mode=pb.POSITION_CONTROL):
  #       '''
  #       Gripper commands need to be mirrored to simulate behavior of the actual
  #       UR5. Converts one command input to 6 joint positions, used for the
  #       robotiq gripper. This is a rough simulation of the way the robotiq
  #       gripper works in practice, in the absence of a plugin like the one we
  #       use in Gazebo.

  #       Parameters:
  #       -----------
  #       cmd: 1x1 array of floating point position commands in [-0.8, 0]
  #       mode: PyBullet control mode
  #       '''

  #       cmd = cmd[0]
  #       # This is actually only a 1-DOF gripper
  #       if cmd < -0.1:
  #           cmd_array = [-cmd + 0.1, -cmd + 0.1, cmd + 0.15,
  #                   -cmd + 0.1, -cmd + 0.1, cmd + 0.15]
  #       else:
  #           cmd_array = [-cmd , -cmd, cmd, -cmd, -cmd, cmd]
  #       forces = [25., 25., 25., 25., 25., 25.]
  #       gains = [0.1, 0.1, 0.15, 0.1, 0.1, 0.15]
  #       #if abs(cmd) < -0.01:
  #       #    mode = pb.TORQUE_CONTROL
  #       #    forces = [0.] * len(cmd_array)
  #       #else:

  #       #gripper_indices = [left_knuckle, left_inner_knuckle,
  #       #               left_fingertip, right_knuckle, right_inner_knuckle,
  #       #               right_fingertip]

  #       pb.setJointMotorControlArray(self.handle, self.gripper_indices, mode,
  #                                    cmd_array,
  #                                    forces=forces,
  #                                    positionGains=gains)
