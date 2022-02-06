import pickle
import matplotlib.pyplot as plt


exp_run = 1

with open('data/rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/no_rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data_no_rl = pickle.load(filehandle)

mean_OPEN_residual = sum(obj_pose_error_data[1]) / len(obj_pose_error_data[0])
mean_OPEN_standard = sum(obj_pose_error_data_no_rl[1]) / len(obj_pose_error_data_no_rl[0])

plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
plt.plot(obj_pose_error_data_no_rl[0], obj_pose_error_data_no_rl[1])
# naming the x axis
plt.xlabel('num-of-episodes')
# naming the y axis
plt.ylabel('Mean OPEN per episode')

plt.legend(["Residual RL", "Standard"])
  
# giving a title to my graph
plt.title('Overall mean OPEN for residual RL = {:.3f} \n Overall mean OPEN for standard controler = {:.3f}'.format(mean_OPEN_residual, mean_OPEN_standard))

# Saving the figure.
plt.savefig("data/joint_plot/ppo2_obj_error_norm_plot_{}.jpg".format(exp_run))

plt.show()
