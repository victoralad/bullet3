import gym
import pickle

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG, PPO2

model = PPO2.load("ppo_coop_manip")

env = gym.make('CoopEnv-v0')
obs = env.reset()
max_test_steps = 1000000
while env.time_step < max_test_steps:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    # grasp fail is mapped to the number 3 in the info dictionary.
    if done:
        if 1 in info or 3 in info:
            env.reset()
    env.render()

folder = "rl"
exp_run = 1
with open('data/{}/obj_pose_error_{}.data'.format(folder, exp_run), 'wb') as filehandle:
    pickle.dump(env.obj_pose_error_data, filehandle)