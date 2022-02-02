import pickle
import matplotlib.pyplot as plt


with open('data/obj_pose_error.data', 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/obtained_reward.data', 'rb') as filehandle:
    # read the data as binary data stream
    obtained_reward_data = pickle.load(filehandle)



plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
plt.plot(obtained_reward_data[0], obtained_reward_data[1])

# naming the x axis
plt.xlabel('num-of-episodes')
# # naming the y axis
# plt.ylabel('avg-obj-pose-error-norm')

# OPEN stands for "Object Pose Error Norm."
plt.legend(["Mean OPEN per episode", "Mean reward per episode"])
  
# giving a title to my graph
# plt.title('Plot of average object pose error norm \n Number of steps = {}, overall avg obj pose error norm = {:.3f}'.format(obj_pose_error_data[0][-1], obj_pose_error_data[1][-1]))

# Saving the figure.
plot_num = 1101
plt.savefig("data/ppo2_obj_error_norm_TRAIN_plot_{}.jpg".format(plot_num))

plt.show()
