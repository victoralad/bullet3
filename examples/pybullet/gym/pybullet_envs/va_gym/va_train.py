import gym

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG

# # Custom MLP policy of two layers of size 16 each
# class CustomDDPGPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
#                                            layers=[16, 16],
#                                            layer_norm=False,
#                                            feature_extraction="mlp")

env = gym.make('CoopEnv-v0')
model = DDPG('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=1000)
print("")
print("---------- Done training -----------------")