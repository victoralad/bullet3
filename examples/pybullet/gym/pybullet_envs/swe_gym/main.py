
import numpy as np
import argparse
import torch
import pickle
from train import training


from stable_baselines import DDPG, PPO2
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument('--epochs', default=100000, type=int, help='Number trials')
parser.add_argument('--episode_per_update', default=10, type=int, help='')
parser.add_argument('--epoch_test', default=100, type=int, help='Number trials')
parser.add_argument('--episodes', default=1000, type=int, help='Number of episodes per epoch during training')
parser.add_argument('--seed', default=1234, type=int, help='Seed')

# PPO parameters
parser.add_argument('--norm_A', default=1, type=int, help='normalize advantage in the outer loop for meta PPO')
parser.add_argument('--c1', default=0.5, type=float, help='scaling constant for value loss in PPO')
parser.add_argument('--c2', default=0.0, type=float, help='scaling constant for entropy bonus in PPO')
parser.add_argument('--gradient_clipping', default=1, type=int, help='clip gradients in PPO')
parser.add_argument('--eps_adapt', default=0.12, type=float, help='epsilon adaptation')


if __name__ == '__main__':
    args = parser.parse_args()
    params = args.__dict__

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logdir = 'logs/'

    training(params, logdir, device)
