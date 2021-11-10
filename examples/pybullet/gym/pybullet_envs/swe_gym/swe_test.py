import gym

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG, PPO2

model = PPO2.load("ppo_coop_manip")

env = gym.make('CoopEnv-v0')
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done and 3 in info:
        env.reset()
    env.render()