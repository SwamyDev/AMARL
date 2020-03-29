from collections import deque

import pytest
import torch
import numpy as np
from gym.spaces import Box, Discrete
from pytest import approx

from amarl.policies import A2CPolicy


@pytest.fixture
def observation_space():
    return Box(0, 1, shape=(32, 32), dtype=np.float32)


@pytest.fixture
def action_space():
    return Discrete(2)


@pytest.fixture
def a2c(observation_space, action_space):
    return A2CPolicy(observation_space, action_space)


@pytest.fixture
def make_observation_batch(observation_space):
    def factory(size=3):
        return torch.empty(size, 3, *observation_space.shape).uniform_(0, 1)

    return factory


def test_a2c_produces_computes_correct_action(a2c, make_observation_batch, action_space):
    actions, _ = a2c.compute_actions(make_observation_batch(100))
    actions = actions.cpu().numpy()
    assert actions.shape == (100,)
    assert actions.min() == 0 and actions.max() == (action_space.n - 1)


def test_a2c_can_be_trained_to_prefer_a_certain_action_when_in_a_certain_state(a2c, make_observation_batch):
    rollout_length = 5
    batch_size = 16
    obs = make_observation_batch(batch_size)

    actions, _ = a2c.compute_actions(obs)
    assert action_distribution(actions.cpu().numpy(), index=1) <= 0.7

    action_dists = []
    training_batch = dict(actions=[], act_dists=[], vs=[], last_obs=obs)
    training_batch['dones'] = [make_sparse_dones(batch_size) for _ in range(rollout_length)]
    for _ in range(1000):
        dist = action_distribution(actions.cpu().numpy(), index=1)
        action_dists.append(dist)
        training_batch.update(make_rollout(a2c, rollout_length, static_obs=obs))
        a2c.learn_on_batch(training_batch)
        actions, _ = a2c.compute_actions(obs)

    start_avg = sum(action_dists[:100]) / 100
    end_avg = sum(action_dists[-100:]) / 100
    action_increase = end_avg / start_avg
    assert action_increase >= 1.2


def make_rollout(a2c, rollout_length, static_obs):
    batch = dict(actions=[], act_dists=[], vs=[])
    for _ in range(rollout_length):
        acts, infos = a2c.compute_actions(static_obs)
        batch['actions'].append(acts)
        batch['act_dists'].append(infos['act_dists'])
        batch['vs'].append(infos['vs'])
    batch['rewards'] = reward_certain_action(batch['actions'], action_to_reward=1)
    return batch


def reward_certain_action(actions, action_to_reward):
    return [(a == action_to_reward) * 2 - 1 for a in actions]


def make_sparse_dones(size):
    return torch.empty(size).uniform_() > 0.9


def action_distribution(actions, index):
    sum_acts_of_index = (actions == index).sum()
    return sum_acts_of_index / len(actions)
