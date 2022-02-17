import gym
import pickle
import numpy as np

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG, PPO2

from swe_monte_carlo import MonteCarlo

num_simulations = 10
monte_c = MonteCarlo(num_simulations)
monte_c.RunSimulation()
goal_poses = monte_c.GetGoalPoses()
quit()
np.save('goal_poses', goal_poses)

model = PPO2.load("ppo_coop_manip")
env = gym.make('CoopEnv-v0')
obs = env.reset()
max_test_steps = 1000000
for goal_pose in goal_poses:
    obj_pose_error_data = []
    while env.time_step < max_test_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        # grasp fail is mapped to the number 3 in the info dictionary.
        obj_pose_error_data += env.obj_pose_error_data[-1]]
        if done:
            if 1 in info:
                break
            elif 3 in info:
                obj_pose_error_data = []
                env.reset()
        env.render()

folder = "rl"
exp_run = 1
with open('data/{}/obj_pose_error_{}.data'.format(folder, exp_run), 'wb') as filehandle:
    pickle.dump(obj_pose_error_data, filehandle)
