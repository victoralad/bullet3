import pickle
import matplotlib.pyplot as plt

# with open('data/summary_reward.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     summary_reward_data = pickle.load(filehandle)

# with open('data/reward.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     reward_data = pickle.load(filehandle)

# # Aesthetic artifact
# reward_data[1][0] = min(reward_data[1])
# reward_data[1][1] = reward_data[1][0]

# # Get the reward data summary.
# print(summary_reward_data)
# print(reward_data[1][-1])
# overall_avg_reward = summary_reward_data[2]

# plt.plot(reward_data[0], reward_data[1])
# # naming the x axis
# plt.xlabel('num-time-steps')
# # naming the y axis
# plt.ylabel('average-reward')
  
# # giving a title to my graph
# # (temp. wrong) Average reward is the total reward obtained in an episode divided by the total number of steps in that episode.
# plt.title('Plot of running avg reward (NO RL) \n num_time_steps = {}, Overall avg reward = {:.3f}'.format(summary_reward_data[0], summary_reward_data[2]))

# # Saving the figure.
# plot_num = 15
# plt.savefig("data/ppo2_rewards_plot_{}_No_RL.jpg".format(plot_num))

# plt.show()


with open('data/obj_pose_error.data', 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

# Aesthetic artifact
min_pose_error = min(obj_pose_error_data[1])
max_pose_error = max(obj_pose_error_data[1])


plt.plot(obj_pose_error_data[0], obj_pose_error_data[1])
# naming the x axis
plt.xlabel('num-time-steps')
# naming the y axis
plt.ylabel('avg-obj-pose-error-norm')
  
# giving a title to my graph
plt.title('Plot of average object pose error norm \n Number of steps = {}, overall avg obj pose error norm = {:.3f}'.format(obj_pose_error_data[0][-1], obj_pose_error_data[1][-1]))

# Saving the figure.
plot_num = 19
plt.savefig("data/ppo2_rewards_plot_{}.jpg".format(plot_num))

plt.show()
