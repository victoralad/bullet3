import gym
import numpy as np
import pickle

from stable_baselines import DDPG
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

env = gym.make('CoopEnv-v0')

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG('MlpPolicy', env, verbose=1, action_noise=action_noise)
# model = DDPG('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=3000)
print("")
print("---------- Done training -----------------")

model.save("ddpg_coop_manip")

with open('data/reward.data', 'wb') as filehandle:
    pickle.dump(env.reward_data, filehandle)