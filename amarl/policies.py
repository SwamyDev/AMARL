import abc
import torch
import torch.nn as nn
import numpy as np

from amarl.models import A2CNet


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
        return [self._action_space.sample() for _ in range(len(observations))]

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
        acts_dist, vs = self._model(observations.to(self._device))
        a = acts_dist.sample()
        return a, {'vs': vs.squeeze(dim=1), 'act_dists': acts_dist}

    def learn_on_batch(self, batch):
        _, next_vs = self._model(batch['last_obs'].to(self._device))
        next_return = next_vs.squeeze(dim=1)
        rewards = torch.stack(batch['rewards']).to(self._device)
        dones = torch.stack(batch['dones']).to(torch.uint8).to(self._device)
        vs = torch.stack(batch['vs'])

        returns = torch.empty_like(rewards)
        advantages = torch.empty_like(rewards)
        for i in reversed(range(len(rewards))):
            next_return = rewards[i] + self._gamma * (1 - dones[i]) * next_return
            returns[i] = next_return
            advantages[i] = next_return - vs[i]

        actions = batch['actions']
        act_dist = batch['act_dists']
        log_prob = torch.stack([d.log_prob(a) for a, d in zip(actions, act_dist)])
        entropy = torch.stack([d.entropy() for d in act_dist])

        policy_loss = -(log_prob * advantages).mean()
        value_loss = (returns - vs).pow(2).mean()
        entropy_loss = entropy.mean()
        total_loss = policy_loss - self._entropy_loss_weight * entropy_loss + self._value_loss_weight * value_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip)
        self._optimizer.step()


class CommunicationPolicy(Policy):
    pass
