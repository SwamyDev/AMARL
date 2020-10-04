import random

import numpy as np
import pytest
import torch
from gym.spaces import Box, Discrete

from amarl.messenger import broadcast, Message
from amarl.policies import A2CPolicy, A2CLSTMPolicy
from amarl.workers import Rollout, IrregularRollout


@pytest.fixture(scope='session')
def device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using torch device: {d}")
    return d


@pytest.fixture(scope='session')
def fixed_seed():
    return random_seed(21)


@pytest.fixture
def img_obs_space():
    return Box(0, 1, shape=(4, 84, 84), dtype=np.float32)


@pytest.fixture
def linear_obs_space():
    return Box(0, 1, shape=(8,), dtype=np.float32)


@pytest.fixture
def basic_action_space():
    return Discrete(2)


@pytest.fixture
def a2c(img_obs_space, basic_action_space, fixed_seed, device):
    return A2CPolicy(img_obs_space, basic_action_space, device=device)


@pytest.fixture
def a2c_lstm(linear_obs_space, basic_action_space, fixed_seed, device):
    return A2CLSTMPolicy(linear_obs_space, basic_action_space, optimizer={'RMSprop': {'lr': 3e-3, 'alpha': 0.99}},
                         device=device)


@pytest.fixture
def make_img_batch(img_obs_space):
    def factory(size=3):
        return torch.rand(size, *img_obs_space.shape, dtype=torch.float32)

    return factory


@pytest.fixture
def make_linear_batch(linear_obs_space):
    def factory(size=3):
        obs = torch.rand(1, *linear_obs_space.shape, dtype=torch.float32)
        return {rank_id: obs.clone() for rank_id in range(size)}

    return factory


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))
    return seed


def test_a2c_produces_computes_correct_action(a2c, make_img_batch, basic_action_space):
    actions, _ = a2c.compute_actions(make_img_batch(100))
    assert actions.shape == (100,)
    assert actions.min() == 0 and actions.max() == (basic_action_space.n - 1)


def test_a2c_can_be_trained_to_prefer_a_certain_action_when_in_a_certain_state(a2c, make_img_batch):
    rollout_length = 5
    batch_size = 16
    obs = make_img_batch(batch_size)
    dones = [make_sparse_dones(batch_size) for _ in range(rollout_length)]

    initial_dist = get_action_probabilities(*a2c.compute_actions(obs))
    assert initial_dist[1] < 0.6

    for _ in range(100):
        a2c.learn_on_batch(make_action_one_rewarded_rollout(a2c, rollout_length, static_obs=obs, static_dones=dones))

    trained_dist = get_action_probabilities(*a2c.compute_actions(obs))
    assert trained_dist[1] >= 0.999


def get_action_probabilities(_, train_data):
    return train_data['act_dists'].probs.detach().cpu().numpy()[0]


def make_action_one_rewarded_rollout(a2c, rollout_length, static_obs, static_dones):
    rollout = Rollout()
    rewards = []
    for _ in range(rollout_length):
        a, train_data = a2c.compute_actions(static_obs)
        rollout.append(train_data)
        rewards.append(reward_certain_action(a, action_to_reward=1))

    rollout.set_element('rewards', rewards)
    rollout.set_element('dones', static_dones)
    rollout.set_element('last_obs', static_obs)
    return rollout


def reward_certain_action(a, action_to_reward):
    return (a == action_to_reward) * 2 - 1


def make_sparse_dones(size):
    return np.random.rand(size) > 0.9


def action_distribution(actions, index):
    sum_acts_of_index = (actions == index).sum()
    return sum_acts_of_index / len(actions)


def test_a2c_lstm_can_be_trained_to_memorize_first_observation_and_condition_an_action_on_it(a2c_lstm,
                                                                                             make_linear_batch,
                                                                                             fixed_seed):
    rollout = 5
    batch_size = 1
    left_obs = make_linear_batch(batch_size)
    right_obs = make_linear_batch(batch_size)

    initial_dist = get_last_action_dists(a2c_lstm, left_obs, rollout)
    assert_is_roughly_uniform(initial_dist)
    initial_dist = get_last_action_dists(a2c_lstm, right_obs, rollout)
    assert_is_roughly_uniform(initial_dist)

    for _ in range(1200):
        first_obs, rewarded_a = (left_obs, 0) if np.random.randint(2) == 0 else (right_obs, 1)
        a2c_lstm.learn_on_batch(
            make_last_action_reward_conditioned_on_first_obs_rollout(a2c_lstm, first_obs, rewarded_a, rollout))

    trained_dist = get_last_action_dists(a2c_lstm, left_obs, rollout)
    assert_all_action_probs_ge(trained_dist, action=0, value=0.99)
    trained_dist = get_last_action_dists(a2c_lstm, right_obs, rollout)
    assert_all_action_probs_ge(trained_dist, action=1, value=0.99)


def assert_is_roughly_uniform(dists):
    for dist in dists.values():
        assert 0.2 < dist[0] < 0.8
        assert 0.2 < dist[1] < 0.8


def get_last_action_dists(policy, obs, rollout):
    train = None
    for i in range(rollout):
        _, train = policy.compute_actions(obs)
        done = i == rollout - 1
        if done:
            send_done_message(obs.keys())
    return {rank_id: dist.probs.detach().cpu().numpy()[0] for rank_id, dist in train['act_dists'].items()}


def send_done_message(indices):
    broadcast(Message.ENV_TERMINATED, index_dones=np.array(indices))


class _LstmEnv:
    def __init__(self, init_obs, rewarded_action, length):
        self._init_obs = {k: init_obs[k].clone() for k in init_obs}
        self._next_obs = init_obs
        self._obs_shape = init_obs[0].shape
        self._rewarded_action = rewarded_action
        self._length = length
        self._num_steps = 0

    def step(self, actions):
        self._num_steps += 1

        obs, reward, done = self._next_obs, 0.0, self._num_steps == self._length
        if done:
            reward = 1.0 if actions[0] == self._rewarded_action else -1.0
            send_done_message(obs.keys())

        self._next_obs = {rank_id: torch.rand(*self._obs_shape, dtype=torch.float32) for rank_id in obs}
        return obs, {rank_id: np.array([reward]) for rank_id in obs}, {rank_id: np.array([done]) for rank_id in obs}, {}

    def reset(self):
        self._next_obs = {k: self._init_obs[k].clone() for k in self._init_obs}
        return self._next_obs


def make_last_action_reward_conditioned_on_first_obs_rollout(policy, observations, rewarded_action, rollout_len):
    rollout = IrregularRollout()
    env = _LstmEnv(observations, rewarded_action, rollout_len)
    observations = env.reset()
    for i in range(rollout_len):
        a, additional = policy.compute_actions(observations)
        observations, rewards, dones, _ = env.step(a)

        elements = dict(rewards=rewards, dones=dones)
        elements.update(additional)
        rollout.append(elements)

    return rollout


def assert_all_action_probs_ge(dists, action, value):
    for dist in dists.values():
        assert dist[action] >= value
