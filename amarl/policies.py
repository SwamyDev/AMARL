import abc
import torch
import torch.nn as nn
import numpy as np

from amarl.models import A2CNet, A2CLinearNet


class Policy(abc.ABC):
    @abc.abstractmethod
    def compute_actions(self, observations):
        pass

    @abc.abstractmethod
    def learn_on_batch(self, batch):
        pass


class RandomPolicy(Policy):
    def __init__(self, action_space):
        self._action_space = action_space

    def compute_actions(self, observations):
        return np.array([self._action_space.sample() for _ in range(len(observations))]), None

    def learn_on_batch(self, batch):
        pass


class A2CPolicy(Policy):
    def __init__(self, observation_space, action_space, gamma=0.99, entropy_loss_weight=0.01, value_loss_weight=0.5,
                 gradient_clip=5, device='cpu'):
        self._gamma = gamma
        self._entropy_loss_weight = entropy_loss_weight
        self._value_loss_weight = value_loss_weight
        self._gradient_clip = gradient_clip
        self._device = device

        self._model = A2CNet(observation_space.shape, action_space.n).to(self._device)
        self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=1e-4, alpha=0.99, eps=1e-5)

    def compute_actions(self, observations):
        acts_dist, vs = self._model(torch.from_numpy(observations).to(self._device))
        a = acts_dist.sample()
        return a, {'vs': vs.squeeze(dim=1), 'act_dists': acts_dist}

    def learn_on_batch(self, rollout):
        _, next_vs = self._model(torch.from_numpy(rollout['last_obs']).to(self._device))
        next_return = next_vs.squeeze(dim=1).detach()
        rewards = rollout['rewards']
        dones = rollout['dones']
        vs = rollout['vs']

        returns = [None] * len(rollout)
        advantages = [None] * len(rollout)
        for i in reversed(range(len(rollout))):
            r = torch.from_numpy(rewards[i]).to(self._device)
            d = torch.from_numpy(dones[i]).to(torch.uint8).to(self._device)
            next_return = r + self._gamma * (1 - d) * next_return
            advantages[i] = next_return - vs[i].detach()
            returns[i] = next_return.detach()

        adv = torch.cat(advantages)
        ret = torch.cat(returns)
        v = torch.cat(vs)
        actions = rollout['actions']
        act_dist = rollout['act_dists']
        log_prob = torch.cat([d.log_prob(a) for a, d in zip(actions, act_dist)], dim=0)
        entropy = torch.cat([d.entropy() for d in act_dist], dim=0)

        policy_loss = -(log_prob * adv).mean()
        value_loss = (ret - v).pow(2).mean()
        entropy_loss = entropy.mean()

        self._optimizer.zero_grad()
        total_loss = policy_loss - self._entropy_loss_weight * entropy_loss + self._value_loss_weight * value_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip)
        self._optimizer.step()


class CommunicationPolicy(Policy):
    pass
