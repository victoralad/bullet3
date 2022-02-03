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

# from stable_baselines.ddpg.policies import FeedForwardPolicy
# Custom MLP policy of two layers of size 16 each
# class CustomDDPGPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
#                                            layers=[32, 32],
#                                            layer_norm=False,
#                                            feature_extraction="mlp")

# model = DDPG(CustomDDPGPolicy, env, verbose=1, action_noise=action_noise)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, act_fun=tf.nn.tanh,
                                           net_arch=[32, 32, dict(pi=[32, 32, 6], vf=[32, 32, 1])],
                                           feature_extraction="mlp")

# model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log="./data/ppo2_coop_manip_tensorboard/")
model = PPO2(CustomPolicy, env, learning_rate=2.5e-4, verbose=1)

# Train the agent
total_timesteps = 10000
model.learn(total_timesteps=total_timesteps)

print("")
print("---------- Done training -----------------")

model.save("ppo_coop_manip")
del model

# # time_step is the total time steps for the entire simulation.
# summary_reward_data = [env.time_step, env.overall_reward_sum, env.overall_reward_sum / env.time_step]
# with open('data/summary_reward.data', 'wb') as filehandle:
#     pickle.dump(summary_reward_data, filehandle)

# with open('data/reward.data', 'wb') as filehandle:
#     pickle.dump(env.reward_data, filehandle)

with open('data/obj_pose_error.data', 'wb') as filehandle:
    pickle.dump(env.mean_obj_pose_error_norm_data, filehandle)

with open('data/obtained_reward.data', 'wb') as filehandle:
    pickle.dump(env.reward_data, filehandle)
