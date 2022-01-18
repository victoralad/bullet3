import pickle
import matplotlib.pyplot as plt


exp_run = 12

with open('data/rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/no_rl/obj_pose_error_{}.data'.format(exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data_no_rl = pickle.load(filehandle)


plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
plt.plot(obj_pose_error_data_no_rl[0], obj_pose_error_data_no_rl[1])
# naming the x axis
plt.xlabel('Num-time-steps')
# naming the y axis
plt.ylabel('Avg-obj-pose-error-norm')

plt.legend(["Residual RL", "Standard"])
  
# giving a title to my graph
plt.title('Plot of average object pose error norm \n Overall avg obj pose error norm: Residual RL = {:.3f}, Standard = {:.3f}'.format(obj_pose_error_data[1][-1], obj_pose_error_data_no_rl[1][-1]))

# Saving the figure.
plt.savefig("data/joint_plot/ppo2_obj_error_norm_plot_{}.jpg".format(exp_run))

plt.show()
