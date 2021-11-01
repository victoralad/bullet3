import numpy as np
import gym
from tqdm import tqdm
from agents import PPO
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from torch.utils.tensorboard import SummaryWriter

def training(params, logdir, device):

    writer = SummaryWriter(log_dir=logdir)

    agent = PPO(params, logdir, device)

    env = gym.make('CoopEnv-v0')

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    for epoch in range(params['epochs']):

        cumulative_reward = 0
        final_reward = 0

        st = env.reset()

        for t in range(params['episodes']):
            a, logprob = agent.get_action(st, test=False)
            st1, r, done, info = env.step(a)

            cumulative_reward += r
            final_reward = r

            if t == params['episodes'] - 1:
                done = True

            agent.push_data(st, a, logprob, r, done, st1)

            if done:
                break

            st = st1

        writer.add_scalar("Cumulative reward", cumulative_reward, int(epoch))
        writer.add_scalar("Final reward", final_reward, int(epoch))

        agent.update(epoch)





