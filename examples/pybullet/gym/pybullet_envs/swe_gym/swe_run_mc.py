import gym
import pickle
import config
import numpy as np

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG, PPO2

config.switch_goal_pose = False
config.goal_pose_idx = 0
config.time_step = 1

model = PPO2.load("ppo_coop_manip")
env = gym.make('CoopEnv-v0')
obs = env.reset()
max_test_steps = 1000000
overall_OPEN_data = []
for goal_pose in env.obj_goal_poses:
    obj_pose_error_data = []
    while config.time_step < max_test_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        # grasp fail is mapped to the number 3 in the info dictionary.
        obj_pose_error_data += [env.obj_pose_error_data[-1]]
        if done:
            if 1 in info:
                break
            elif 3 in info:
                obj_pose_error_data = []
                env.reset()
        env.render()
    config.time_step = 1
    config.switch_goal_pose = True
    config.goal_pose_idx += 1
    if config.goal_pose_idx < len(env.obj_goal_poses):
        env.reset()
    overall_OPEN_data += [obj_pose_error_data]

folder = "no_rl"
exp_run = 1
with open('data/{}/obj_pose_error_{}.data'.format(folder, exp_run), 'wb') as filehandle:
    pickle.dump(overall_OPEN_data, filehandle)
