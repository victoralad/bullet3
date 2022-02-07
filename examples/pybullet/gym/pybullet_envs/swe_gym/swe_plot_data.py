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
final_mean_action = action_data[0][-1]
final_var_action = action_data[1][-1]

plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
plt.plot(obtained_reward_data[0], obtained_reward_data[1])
plt.plot(range(len(action_data[0])), action_data[0])
plt.plot(range(len(action_data[1])), action_data[1])
print(len(action_data[1]))

# naming the x axis
plt.xlabel('num-of-episodes')
# # naming the y axis
# plt.ylabel('avg-obj-pose-error-norm')

# OPEN stands for "Object Pose Error Norm."
plt.legend(["Mean OPEN per episode", "Mean reward per episode", "Mean action per episode", "Var per episode"])
# plt.legend(["Mean OPEN per episode"])
  
# giving a title to my graph
plt.title('Overall mean OPEN = {:.3f}, Overall mean reward = {:.3f} \n Final mean action = {:.3f}, Final action variance = {:.3f}'.format(mean_OPEN, mean_reward, final_mean_action, final_var_action))
# plt.title('Plot of average object pose error norm \n Number of steps = {}, overall avg obj pose error norm = {:.3f}'.format(obj_pose_error_data[0][-1], obj_pose_error_data[1][-1]))

# Saving the figure.
plot_num = 1105
plt.savefig("data/OPEN_TRAIN_plot_{}.jpg".format(plot_num))

plt.show()
