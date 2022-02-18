import pickle
import matplotlib.pyplot as plt
import numpy as np


exp_run = 1

with open('data/rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/no_rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data_no_rl = pickle.load(filehandle)


plt.plot(list(range(len(obj_pose_error_data))), obj_pose_error_data)
plt.plot(list(range(len(obj_pose_error_data_no_rl))), obj_pose_error_data_no_rl)
# naming the x axis
plt.xlabel('Num-time-steps')
# naming the y axis
plt.ylabel('Obj-pose-error-norm')

plt.legend(["Residual RL", "Standard"])
  
# giving a title to my graph
plt.title('Plot of Object pose error norm over a single episode \n Avg object pose error norm: Residual RL = {:.3f}, Standard = {:.3f}'.format(np.mean(obj_pose_error_data), np.mean(obj_pose_error_data_no_rl)))

# Saving the figure.
plt.savefig("data/joint_plot/eps_OPEN_plot_{}.jpg".format(exp_run))

plt.show()
