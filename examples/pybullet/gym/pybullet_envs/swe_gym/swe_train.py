import gym
import numpy as np
import pickle
import tensorflow as tf

from stable_baselines import DDPG, PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, CnnPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm

env = gym.make('CoopEnv-v0')

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=8, filter_size=3, stride=1, **kwargs))
    layer_3 = conv_to_fc(layer_3)
    mlp_activ = tf.nn.softmax
    return mlp_activ(linear(layer_3, 'fc1', n_hidden=128))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")

model = PPO2(CustomPolicy, env, learning_rate=2.5e-3, verbose=1)

# class CustomPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs, act_fun=tf.nn.tanh,
#                                            net_arch=[32, 32, dict(pi=[32, 32, 6], vf=[32, 32, 1])],
#                                            feature_extraction="cnn")

# model = PPO2(CustomPolicy, env, learning_rate=2.5e-1, verbose=1, tensorboard_log="./data/ppo2_coop_manip_tensorboard/")
# # model = PPO2(CustomPolicy, env, learning_rate=2.5e-4, verbose=1)


# Train the agent
total_timesteps = 30000
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

with open('data/actions.data', 'wb') as filehandle:
    pickle.dump(env.action_data, filehandle)