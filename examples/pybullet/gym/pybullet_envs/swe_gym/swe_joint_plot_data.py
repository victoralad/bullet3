import pickle
import matplotlib.pyplot as plt
import numpy as np


exp_run = 1

with open('data/rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/no_rl/obj_pose_error_traj_{}.data'.format(6), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data_no_rl = pickle.load(filehandle)


plt.plot(list(range(len(obj_pose_error_data))), obj_pose_error_data)
plt.plot(list(range(len(obj_pose_error_data_no_rl))), obj_pose_error_data_no_rl)

residual_rl_mean = np.mean(obj_pose_error_data)
standard_mean = np.mean(obj_pose_error_data_no_rl)

# naming the x axis
plt.xlabel('Num-time-steps')
# naming the y axis
plt.ylabel('Obj-pose-error-norm')

plt.legend(["Residual RL", "Standard"])
  
# giving a title to my graph
plt.title('Plot of Object pose error norm \n Avg obj pose error norm: Residual RL = {:.3f}, Standard = {:.3f}'.format(residual_rl_mean, standard_mean))
# plt.title('Plot of average object pose error norm \n Overall avg obj pose error norm: Residual RL = {:.3f}'.format(residual_rl_mean))
# plt.title('Plot of average object pose error norm \n Overall avg obj pose error norm: Standard = {:.3f}'.format(obj_pose_error_data_no_rl[1][-1]))

# Saving the figure.
plt.savefig("data/joint_plot/multi_OPEN_seed_{}.jpg".format(exp_run))

plt.show()



# exp_run = 1

# with open('data/rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
#     # read the data as binary data stream
#     obj_pose_error_data = pickle.load(filehandle)

# with open('data/no_rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
#     # read the data as binary data stream
#     obj_pose_error_data_no_rl = pickle.load(filehandle)


# plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
# plt.plot(obj_pose_error_data_no_rl[0], obj_pose_error_data_no_rl[1])
# # naming the x axis
# plt.xlabel('Num-time-steps')
# # naming the y axis
# plt.ylabel('Avg-obj-pose-error-norm')

# plt.legend(["Residual RL", "Standard"])
  
# # giving a title to my graph
# plt.title('Plot of average object pose error norm \n Overall avg obj pose error norm: Residual RL = {:.3f}, Standard = {:.3f}'.format(obj_pose_error_data[1][-1], obj_pose_error_data_no_rl[1][-1]))

# # Saving the figure.
# plt.savefig("data/joint_plot/OPEN_plot_{}.jpg".format(exp_run))

# plt.show()
