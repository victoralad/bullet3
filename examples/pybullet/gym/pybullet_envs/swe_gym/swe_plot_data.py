import pickle
import matplotlib.pyplot as plt


with open('data/obj_pose_error.data', 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/obtained_reward.data', 'rb') as filehandle:
    # read the data as binary data stream
    obtained_reward_data = pickle.load(filehandle)

with open('data/obj_pose_error_not_avg.data', 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_not_avg_data = pickle.load(filehandle)


plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
plt.plot(obj_pose_error_data[0], obtained_reward_data)
plt.plot(obj_pose_error_data[0], obj_pose_error_not_avg_data)

# naming the x axis
plt.xlabel('num-time-steps')
# # naming the y axis
# plt.ylabel('avg-obj-pose-error-norm')

plt.legend(["Avg object-pose-error-norm", "Reward", "object-pose-error-norm"])
  
# giving a title to my graph
plt.title('Plot of average object pose error norm \n Number of steps = {}, overall avg obj pose error norm = {:.3f}'.format(obj_pose_error_data[0][-1], obj_pose_error_data[1][-1]))

# Saving the figure.
plot_num = 13
plt.savefig("data/ppo2_obj_error_norm_TRAIN_plot_{}.jpg".format(plot_num))

plt.show()
