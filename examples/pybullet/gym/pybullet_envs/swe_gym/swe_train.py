import gym
import numpy as np
import pickle
import tensorflow as tf

from stable_baselines import DDPG, PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

env = gym.make('CoopEnv-v0')

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# # model = DDPG('MlpPolicy', env, verbose=1, action_noise=action_noise)
# # model = DDPG('MlpPolicy', env, verbose=1)

# model = PPO2('MlpPolicy', env, verbose=1)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, act_fun=tf.nn.tanh,
                                           net_arch=[32, 32, dict(pi=[32, 32, 12], vf=[32, 32, 1])],
                                           feature_extraction="mlp")

model = PPO2(CustomPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=2000)

print("")
print("---------- Done training -----------------")

model.save("ppo_coop_manip")
del model

with open('data/reward.data', 'wb') as filehandle:
    pickle.dump(env.reward_data, filehandle)