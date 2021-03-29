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
plt.ylabel('cummulative-reward')
  
# giving a title to my graph
plt.title('Plot of cumulative rewards per episode')

# Saving the figure.
plot_num = 3
plt.savefig("data/rewards_plot_{}.jpg".format(plot_num))

plt.show()

