import gym
import pickle

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG, PPO2

seed = 1
model = PPO2.load("ppo_coop_manip_seed_{}".format(seed))

env = gym.make('CoopEnv-v0')
obs = env.reset()
max_test_steps = 1000000
obj_pose_error_data = []
while env.time_step < max_test_steps:
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

folder = "no_rl"
exp_run = 102
with open('data/{}/obj_pose_error_{}_seed_{}.data'.format(folder, exp_run, seed), 'wb') as filehandle:
    pickle.dump(obj_pose_error_data, filehandle)