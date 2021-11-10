import pickle
import matplotlib.pyplot as plt

with open('data/reward.data', 'rb') as filehandle:
    # read the data as binary data stream
    reward_data = pickle.load(filehandle)

# Aesthetic artifact
reward_data[1][0] = min(reward_data[1])
reward_data[1][1] = reward_data[1][0]

print(reward_data)

plt.plot(reward_data[0], reward_data[1])
# naming the x axis
plt.xlabel('num-episodes')
# naming the y axis
plt.ylabel('average-reward')
  
# giving a title to my graph
# Average reward is the total reward obtained in an episode divided by the total number of steps in that episode.
plt.title('Plot of average reward per episode (NO RL)')

# Saving the figure.
plot_num = 8
plt.savefig("data/ppo2_rewards_plot_{}.jpg".format(plot_num))

plt.show()

