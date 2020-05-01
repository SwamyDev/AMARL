import abc
import torch
import torch.nn as nn
import numpy as np

from amarl.models import A2CNet, A2CLinearNet, A2CLinearLSTMNet


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

    def compute_actions(self, observations, dones=None):
        return np.array([self._action_space.sample() for _ in range(len(observations))]), None

    def learn_on_batch(self, batch):
        pass


class ActorCriticPolicy(Policy, abc.ABC):
    def __init__(self, model, optimizer, gamma, entropy_loss_weight, value_loss_weight,
                 gradient_clip, device='cpu'):
        self._gamma = gamma
        self._entropy_loss_weight = entropy_loss_weight
        self._value_loss_weight = value_loss_weight
        self._gradient_clip = gradient_clip
        self._device = device
        self._model = model.to(self._device)
        if optimizer:
            self._optimizer = self._make_optimizer_from_dict(optimizer)
        else:
            self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=1e-4, alpha=0.99, eps=1e-5)

    def _make_optimizer_from_dict(self, optimizer_dict):
        d = optimizer_dict.copy()
        opt_class, kwargs = d.popitem()
        return getattr(torch.optim, opt_class)(self._model.parameters(), **kwargs)

    def _train_actor_critic_on(self, rollout, next_return):
        advantages, returns = self._calc_advantages_and_return(next_return, rollout)
        total_loss = self._make_actor_critic_loss(rollout, advantages, returns)
        self._train_model_on_loss(total_loss)

    def _calc_advantages_and_return(self, next_return, rollout):
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

        return advantages, returns

    def _make_actor_critic_loss(self, rollout, advantages, returns):
        adv = torch.cat(advantages)
        ret = torch.cat(returns)
        v = torch.cat(rollout['vs'])
        actions = rollout['actions']
        act_dist = rollout['act_dists']
        log_prob = torch.cat([d.log_prob(a) for a, d in zip(actions, act_dist)], dim=0)
        entropy = torch.cat([d.entropy() for d in act_dist], dim=0)
        policy_loss = -(log_prob * adv).mean()
        value_loss = (ret - v).pow(2).mean()
        entropy_loss = entropy.mean()
        total_loss = policy_loss - self._entropy_loss_weight * entropy_loss + self._value_loss_weight * value_loss
        return total_loss

    def _train_model_on_loss(self, total_loss):
        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip)
        self._optimizer.step()


class A2CPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, optimizer=None, gamma=0.99, entropy_loss_weight=0.01,
                 value_loss_weight=0.5, gradient_clip=5, device='cpu'):
        super().__init__(A2CNet(observation_space.shape, action_space.n), optimizer, gamma, entropy_loss_weight,
                         value_loss_weight, gradient_clip, device)

    def compute_actions(self, observations, dones=None):
        acts_dist, vs = self._model(torch.from_numpy(observations).to(self._device))
        a = acts_dist.sample()
        return a, {'vs': vs.squeeze(dim=1), 'act_dists': acts_dist}

    def learn_on_batch(self, rollout):
        _, next_vs = self._model(torch.from_numpy(rollout['last_obs']).to(self._device))
        next_return = next_vs.squeeze(dim=1).detach()
        self._train_actor_critic_on(rollout, next_return)


class A2CLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, optimizer=None, gamma=0.99, entropy_loss_weight=0.01,
                 value_loss_weight=0.5, gradient_clip=30, device='cpu'):
        super().__init__(A2CLinearLSTMNet(observation_space.shape[0], action_space.n), optimizer, gamma,
                         entropy_loss_weight, value_loss_weight, gradient_clip, device)
        self._hidden = self._model.get_initial_state()
        self._pending_reset = False

    def compute_actions(self, observations, dones=None):
        if self._pending_reset:
            self._hidden = self._model.get_initial_state()
        self._pending_reset = dones[0]

        acts_dist, vs, self._hidden = self._model(torch.from_numpy(observations).to(self._device), self._hidden)
        a = acts_dist.sample()
        return a, {'vs': vs.squeeze(dim=1), 'act_dists': acts_dist}

    def learn_on_batch(self, rollout):
        is_terminal = rollout['dones'][-1][0]
        next_return = torch.zeros(*rollout['rewards'][0].shape).to(self._device)
        if not is_terminal:
            _, next_vs, self._hidden = self._model(torch.from_numpy(rollout['last_obs']).to(self._device), self._hidden)
            next_return = next_vs.squeeze(dim=1).detach()

        self._train_actor_critic_on(rollout, next_return)

        self._hidden[0].detach_()
        self._hidden[1].detach_()


class CommunicationPolicy(Policy):
    pass
