import gym

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG

model = DDPG.load("ddpg_coop_manip")

env = gym.make('CoopEnv-v0')
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()