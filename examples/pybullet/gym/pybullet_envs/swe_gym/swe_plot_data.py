import pickle
import matplotlib.pyplot as plt


with open('data/obj_pose_error.data', 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/obtained_reward.data', 'rb') as filehandle:
    # read the data as binary data stream
    obtained_reward_data = pickle.load(filehandle)

# Aesthetic artifact
min_pose_error = min(obj_pose_error_data[1])
max_pose_error = max(obj_pose_error_data[1])


plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
plt.plot(obj_pose_error_data[0], obtained_reward_data)

# naming the x axis
plt.xlabel('num-time-steps')
# # naming the y axis
# plt.ylabel('Avg-obj-pose-error-norm')

plt.legend(["Avg object-pose-error-norm", "Reward"])
  
# giving a title to my graph
plt.title('Overall avg obj pose error norm = {:.3f}'.format(obj_pose_error_data[1][-1]))

# Saving the figure.
plot_num = 1
plt.savefig("data/OPEN_TRAIN_plot_{}.jpg".format(plot_num))

plt.show()
