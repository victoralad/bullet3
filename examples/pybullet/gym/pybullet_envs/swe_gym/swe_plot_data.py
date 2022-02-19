import pickle
import matplotlib.pyplot as plt
import numpy as np


with open('data/obj_pose_error.data', 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/obtained_reward.data', 'rb') as filehandle:
    # read the data as binary data stream
    obtained_reward_data = pickle.load(filehandle)


plt.plot(list(range(len(obj_pose_error_data))), obj_pose_error_data)
# plt.plot(obj_pose_error_data[0], obtained_reward_data)

# naming the x axis
plt.xlabel('num-time-steps')
# # naming the y axis
# plt.ylabel('Avg-obj-pose-error-norm')

plt.legend(["Object-pose-error-norm", "Reward"])
  
# giving a title to my graph
plt.title('Overall avg obj pose error norm = {:.3f}'.format(np.mean(obj_pose_error_data)))

# Saving the figure.
plot_num = 1
plt.savefig("data/OPEN_TRAIN_plot_{}.jpg".format(plot_num))

plt.show()
