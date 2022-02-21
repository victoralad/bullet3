import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


exp_run = 1

ax = plt.axes(projection ='3d')

with open('data/rl/traj_3d_pose_{}.data'.format(1), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_1 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(2), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_2 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(3), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_3 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(5), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_5 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(6), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_6 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(7), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_7 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(8), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_8 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(9), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_9 = pickle.load(filehandle)

with open('data/rl/traj_3d_pose_{}.data'.format(10), 'rb') as filehandle:
    # read the data as binary data stream
    traj_3d_plot_10 = pickle.load(filehandle)

x1 = traj_3d_plot_1[0]
y1 = traj_3d_plot_1[1]
z1 = traj_3d_plot_1[2]

x2 = traj_3d_plot_2[0]
y2 = traj_3d_plot_2[1]
z2 = traj_3d_plot_2[2]

x3 = traj_3d_plot_3[0]
y3 = traj_3d_plot_3[1]
z3 = traj_3d_plot_3[2]

x5 = traj_3d_plot_5[0]
y5 = traj_3d_plot_5[1]
z5 = traj_3d_plot_5[2]

x6 = traj_3d_plot_6[0]
y6 = traj_3d_plot_6[1]
z6 = traj_3d_plot_6[2]

x7 = traj_3d_plot_7[0]
y7 = traj_3d_plot_7[1]
z7 = traj_3d_plot_7[2]

x8 = traj_3d_plot_8[0]
y8 = traj_3d_plot_8[1]
z8 = traj_3d_plot_8[2]

x9 = traj_3d_plot_9[0]
y9 = traj_3d_plot_9[1]
z9 = traj_3d_plot_9[2]

x10 = traj_3d_plot_10[0]
y10 = traj_3d_plot_10[1]
z10 = traj_3d_plot_10[2]

x_mean_final_pose = np.mean([x6[-1], x7[-1], x8[-1], x9[-1], x10[-1]])
y_mean_final_pose = np.mean([y6[-1], y7[-1], y8[-1], y9[-1], y10[-1]])
z_mean_final_pose = np.mean([z6[-1], z7[-1], z8[-1], z9[-1], z10[-1]])

print("Yello!")
print(x_mean_final_pose)
print(y_mean_final_pose)
print(z_mean_final_pose)

# -------- Plotting ----------
ax.plot3D(x1, y1, z1)
ax.plot3D(x2, y2, z2)
ax.plot3D(x3, y3, z3)
ax.plot3D(x5, y5, z5)
ax.plot3D(x6, y6, z6)
ax.plot3D(x7, y7, z7)
ax.plot3D(x8, y8, z8)
ax.plot3D(x9, y9, z9)
ax.plot3D(x10, y10, z10)

# naming the x axis
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.legend(["Trajectory 1", "Trajectory 2", "Trajectory 3", "Trajectory 5", "Trajectory 6", "Trajectory 7", "Trajectory 8", "Trajectory 9", "Trajectory 10"])
# plt.legend(["Trajectory 1", "Trajectory 2", "Trajectory 3", "Trajectory 4", "Trajectory 5"])

# Saving the figure.
plt.savefig("data/joint_plot/traj_3d_plot_{}.jpg".format(exp_run))

plt.show()


