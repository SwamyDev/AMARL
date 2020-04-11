import numpy as np
import pytest
import torch
from gym.spaces import Box, Discrete

from amarl.policies import A2CPolicy
from amarl.workers import RolloutBatch


@pytest.fixture(scope='session')
def device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using torch device: {d}")
    return d


@pytest.fixture
def observation_space():
    return Box(0, 1, shape=(4, 84, 84), dtype=np.float32)


@pytest.fixture
def action_space():
    return Discrete(2)


@pytest.fixture
def a2c(observation_space, action_space, device):
    return A2CPolicy(observation_space, action_space, device=device)


@pytest.fixture
def make_observation_batch(observation_space):
    def factory(size=3):
        return np.random.rand(size, *observation_space.shape).astype(np.float32)

    return factory


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def test_a2c_produces_computes_correct_action(a2c, make_observation_batch, action_space):
    actions, _ = a2c.compute_actions(make_observation_batch(100))
    actions = actions.cpu().numpy()
    assert actions.shape == (100,)
    assert actions.min() == 0 and actions.max() == (action_space.n - 1)


def test_a2c_can_be_trained_to_prefer_a_certain_action_when_in_a_certain_state(a2c, make_observation_batch):
    random_seed(42)
    rollout_length = 5
    batch_size = 16
    obs = make_observation_batch(batch_size)
    dones = [make_sparse_dones(batch_size) for _ in range(rollout_length)]

    initial_dist = get_action_probabilities(*a2c.compute_actions(obs))
    assert initial_dist[1] < 0.6

    for _ in range(100):
        a2c.learn_on_batch(make_rollout(a2c, rollout_length, static_obs=obs, static_dones=dones))

    trained_dist = get_action_probabilities(*a2c.compute_actions(obs))
    assert trained_dist[1] >= 0.9999


def get_action_probabilities(_, additional):
    return additional['act_dists'].probs.detach().cpu().numpy()[0]


def make_rollout(a2c, rollout_length, static_obs, static_dones):
    rollout = RolloutBatch()
    for _ in range(rollout_length):
        acts, additional = a2c.compute_actions(static_obs)
        elements = dict(actions=acts)
        elements.update(additional)
        rollout.append(elements)

    rollout.set_element('rewards', reward_certain_action(rollout['actions'], action_to_reward=1))
    rollout.set_element('dones', static_dones)
    rollout.set_element('last_obs', static_obs)
    return rollout


def reward_certain_action(actions, action_to_reward):
    return [((a == action_to_reward) * 2 - 1).cpu().numpy() for a in actions]


def make_sparse_dones(size):
    return np.random.rand(size) > 0.9


def action_distribution(actions, index):
    sum_acts_of_index = (actions == index).sum()
    return sum_acts_of_index / len(actions)
