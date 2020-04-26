import abc
import torch
import torch.nn as nn
import numpy as np

from amarl.messenger import subscribe_to, Message
from amarl.models import A2CNet, A2CLinearNet, A2CSocialInfluenceNet, A2CLinearLSTMNet


class Policy(abc.ABC):
    @abc.abstractmethod
    def compute_actions(self, observations, dones=None):
        pass

    @abc.abstractmethod
    def learn_on_batch(self, batch):
        pass

    @staticmethod
    def _make_optimizer_from_dict(model, optimizer_dict):
        d = optimizer_dict.copy()
        opt_class, kwargs = d.popitem()
        return getattr(torch.optim, opt_class)(model.parameters(), **kwargs)


class RandomPolicy(Policy):
    def __init__(self, action_space):
        self._action_space = action_space

    def compute_actions(self, observations, dones=None):
        return np.array([self._action_space.sample() for _ in range(len(observations))]), None

    def learn_on_batch(self, batch):
        pass


class A2CPolicy(Policy):
    def __init__(self, observation_space, action_space, optimizer=None, gamma=0.99, entropy_loss_weight=0.01,
                 value_loss_weight=0.5, gradient_clip=5, device='cpu'):
        self._gamma = gamma
        self._entropy_loss_weight = entropy_loss_weight
        self._value_loss_weight = value_loss_weight
        self._gradient_clip = gradient_clip
        self._device = device

        # self._model = A2CNet(observation_space.shape, action_space.n).to(self._device)
        self._model = A2CLinearNet(observation_space.shape[0], action_space.n).to(self._device)
        if optimizer:
            self._optimizer = self._make_optimizer_from_dict(self._model, optimizer)
        else:
            self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=1e-4, alpha=0.99, eps=1e-5)

    def compute_actions(self, observations, dones=None):
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


class A2CLSTMPolicy(Policy):
    def __init__(self, observation_space, action_space, optimizer=None, gamma=0.99, entropy_loss_weight=0.01,
                 value_loss_weight=0.5, gradient_clip=5, device='cpu'):
        self._gamma = gamma
        self._entropy_loss_weight = entropy_loss_weight
        self._value_loss_weight = value_loss_weight
        self._gradient_clip = gradient_clip
        self._device = device

        self._model = A2CLinearLSTMNet(observation_space.shape[0], action_space.n).to(self._device)
        # self._model = A2CSocialInfluenceNet(observation_space.shape, action_space.n).to(self._device)
        if optimizer:
            self._optimizer = self._make_optimizer_from_dict(self._model, optimizer)
        else:
            self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=1e-4, alpha=0.99, eps=1e-5)
        # self._hidden_x, self._cell_x = self._model.get_initial_state()
        self._model.reset_hidden()

    # def _reset_hidden_at_done(self, idx_dones):
    #     self._hidden_x = torch.autograd.Variable(self._hidden_x.data).float().to(self._device)
    #     self._cell_x = torch.autograd.Variable(self._cell_x.data).float().to(self._device)
    #     for i in idx_dones:
    #         self._hidden_x[i] = torch.zeros_like(self._hidden_x[i])
    #         self._cell_x[i] = torch.zeros_like(self._cell_x[i])

    def compute_actions(self, observations, dones=None):
        acts_dist, vs, prv_hidden = self._model(torch.from_numpy(observations).to(self._device),
                                                torch.from_numpy(np.array(dones)).to(torch.uint8).to(self._device))
        a = acts_dist.sample()
        return a, {'vs': vs.squeeze(dim=1), 'act_dists': acts_dist, 'prv_hidden': prv_hidden}

    def learn_on_batch(self, rollout):
        rewards = rollout['rewards']
        dones = rollout['dones']
        vs = rollout['vs']

        next_return = torch.zeros(*rewards[0].shape).to(self._device)
        is_terminal = rollout['dones'][-1][0]
        if not is_terminal:
            _, next_vs, _ = self._model(torch.from_numpy(rollout['last_obs']).to(self._device),
                                        torch.from_numpy(rollout['dones'][-1]).to(torch.uint8).to(self._device))

            next_return = next_vs.squeeze(dim=1).detach()

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

        # prev_x, prev_c = self._model.hidden_x, self._model.hidden_c
        # self._model.hidden_x, self._model.hidden_c = rollout['initial_hidden']

        self._optimizer.zero_grad()
        total_loss = policy_loss - self._entropy_loss_weight * entropy_loss + self._value_loss_weight * value_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip)
        self._optimizer.step()

        if is_terminal:
            self._model.reset_hidden()
        else:
            self._model.hidden_x.detach_()
            self._model.hidden_c.detach_()


class CommunicationPolicy(Policy):
    pass
