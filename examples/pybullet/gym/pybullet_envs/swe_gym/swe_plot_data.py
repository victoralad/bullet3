import pickle
import matplotlib.pyplot as plt
import numpy as np


with open('data/obj_pose_error.data', 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/obtained_reward.data', 'rb') as filehandle:
    # read the data as binary data stream
    obtained_reward_data = pickle.load(filehandle)

with open('data/actions.data', 'rb') as filehandle:
    # read the data as binary data stream
    action_data = pickle.load(filehandle)

mean_OPEN = sum(obj_pose_error_data[1]) / len(obj_pose_error_data[0])
mean_reward = sum(obtained_reward_data[1]) / len(obtained_reward_data[0])
action_data_np = np.array(action_data)
mean_action = np.mean(action_data_np)
var_action = np.var(action_data_np)
print(mean_action)
print(var_action)
print(len(action_data))

plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
plt.plot(obtained_reward_data[0], obtained_reward_data[1])
plt.plot(range(len(action_data)), action_data)

# naming the x axis
plt.xlabel('num-of-episodes')
# # naming the y axis
# plt.ylabel('avg-obj-pose-error-norm')

# OPEN stands for "Object Pose Error Norm."
plt.legend(["Mean OPEN per episode", "Mean reward per episode"])
# plt.legend(["Mean OPEN per episode"])
  
# giving a title to my graph
plt.title('Overall mean OPEN = {:.3f}, Overall mean reward = {:.3f} \n Mean action = {:.3f}, Action variance = {:.3f}'.format(mean_OPEN, mean_reward, mean_action[0], var_action[0]))
# plt.title('Plot of average object pose error norm \n Number of steps = {}, overall avg obj pose error norm = {:.3f}'.format(obj_pose_error_data[0][-1], obj_pose_error_data[1][-1]))

# Saving the figure.
plot_num = 1105
plt.savefig("data/OPEN_TRAIN_plot_{}.jpg".format(plot_num))

plt.show()
