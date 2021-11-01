import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal



"""
General Two-Head Architecture
"""
class Two_Head(nn.Module):

    def __init__(self, state_dim, action_dim=6, activation=nn.Tanh):
        super(Two_Head, self).__init__()

        # TODO: do we want tanh or ReLU?
        self.body = nn.Sequential(
            nn.Linear(state_dim, 32, bias=True),
            activation(),
            nn.Linear(32, 32, bias=True),
            activation(),
        )

        self.actor = nn.Sequential(
            nn.Linear(32, 32, bias=True),
            activation(),
            nn.Linear(32, 32, bias=True),
            activation(),
            nn.Linear(32, action_dim, bias=True)     # 2 as we are predicting mean and deviation of density probability function
        )

        self.critic = nn.Sequential(
            nn.Linear(32, 32, bias=True),
            activation(),
            nn.Linear(32, 32, bias=True),
            activation(),
            nn.Linear(32, 1, bias=True)
        )

    def forward_pi(self, x):
        return self.actor(self.body(x))

    def forward_V(self, x):
        return self.critic(self.body(x))

    def forward(self, x):
        return self.actor(self.body(x)), self.critic(self.body(x))


"""
PPO network
"""
class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim=6, learn_cov=False):
        super(ActorCritic, self).__init__()

        self.model = Two_Head(state_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

        # TODO: at the moment we are not learning the covariance matrix but just the mean
        self.cov_var = torch.full(size=(action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, state):
        # Get action means from the model
        action_parameters = self.model.forward_pi(state)

        # get mean and std, get normal distribution
        dist = MultivariateNormal(action_parameters.squeeze(), self.cov_mat.squeeze())

        # mu, sigma = action_parameters[:, :1], torch.exp(action_parameters[:, 1:])
        # m = torch.normal(mu[:, 0], sigma[:, 0])

        # sample action, get log probability
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        actions = torch.cat((action, state[0, 6:12]))


        return actions, action_logprob #actions.item(), action_logprob

    def evaluate(self, state, action):
        action_parameters, state_value = self.model.forward(state)

        # TODO:  do I want to lean the variance?
        # TODO: do not learn cocariance at the beginning, but learning it helps with the entropy (get reasonable min max)
        mu, sigma = action_parameters[:, :1], torch.exp(action_parameters[:, 1:])
        # TODO: check if this generates a multivariate gaussian
        m = torch.normal(mu[:, 0], sigma[:, 0])
        #get log probability
        action_logprobs = m.log_prob(action)

        # dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value)#, dist_entropy

