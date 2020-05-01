import numpy as np
import pytest
import torch
from gym.spaces import Box, Discrete

from amarl.policies import A2CPolicy, A2CLSTMPolicy
from amarl.workers import RolloutBatch


@pytest.fixture(scope='session')
def device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using torch device: {d}")
    return d


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
def a2c(img_obs_space, basic_action_space, device):
    return A2CPolicy(img_obs_space, basic_action_space, device=device)


@pytest.fixture
def a2c_lstm(linear_obs_space, basic_action_space, device):
    return A2CLSTMPolicy(linear_obs_space, basic_action_space, device=device)


@pytest.fixture
def make_img_batch(img_obs_space):
    def factory(size=3):
        return np.random.rand(size, *img_obs_space.shape).astype(np.float32)

    return factory


@pytest.fixture
def make_linear_batch(linear_obs_space):
    def factory(size=3):
        return np.random.rand(size, *linear_obs_space.shape).astype(np.float32)

    return factory


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def test_a2c_produces_computes_correct_action(a2c, make_img_batch, basic_action_space):
    actions, _ = a2c.compute_actions(make_img_batch(100))
    actions = actions.cpu().numpy()
    assert actions.shape == (100,)
    assert actions.min() == 0 and actions.max() == (basic_action_space.n - 1)


def test_a2c_can_be_trained_to_prefer_a_certain_action_when_in_a_certain_state(a2c, make_img_batch):
    random_seed(42)
    rollout_length = 5
    batch_size = 16
    obs = make_img_batch(batch_size)
    dones = [make_sparse_dones(batch_size) for _ in range(rollout_length)]

    initial_dist = get_action_probabilities(*a2c.compute_actions(obs))
    assert initial_dist[1] < 0.6

    for _ in range(100):
        a2c.learn_on_batch(make_action_one_rewarded_rollout(a2c, rollout_length, static_obs=obs, static_dones=dones))

    trained_dist = get_action_probabilities(*a2c.compute_actions(obs))
    assert trained_dist[1] >= 0.9999


def get_action_probabilities(_, additional):
    return additional['act_dists'].probs.detach().cpu().numpy()[0]


def make_action_one_rewarded_rollout(a2c, rollout_length, static_obs, static_dones):
    rollout = RolloutBatch()
    for _ in range(rollout_length):
        acts, additional = a2c.compute_actions(static_obs)
        elements = dict(actions=acts)
        elements.update(additional)
        rollout.append(elements)

    rollout.set_element('rewards', reward_certain_actions(rollout['actions'], action_to_reward=1))
    rollout.set_element('dones', static_dones)
    rollout.set_element('last_obs', static_obs)
    return rollout


def reward_certain_actions(actions, action_to_reward):
    return [reward_certain_action(a, action_to_reward) for a in actions]


def reward_certain_action(a, action_to_reward):
    return ((a == action_to_reward) * 2 - 1).cpu().numpy()


def make_sparse_dones(size):
    return np.random.rand(size) > 0.9


def action_distribution(actions, index):
    sum_acts_of_index = (actions == index).sum()
    return sum_acts_of_index / len(actions)


def test_a2c_lstm_can_be_trained_to_memorize_first_observation_and_condition_an_action_on_it(a2c_lstm,
                                                                                             make_linear_batch):
    random_seed(42)
    rollout = 5
    batch_size = 1
    left_obs = make_linear_batch(batch_size)
    right_obs = make_linear_batch(batch_size)

    initial_dist = get_last_action_dist(a2c_lstm, left_obs, rollout)
    assert_is_roughly_uniform(initial_dist)
    initial_dist = get_last_action_dist(a2c_lstm, right_obs, rollout)
    assert_is_roughly_uniform(initial_dist)

    for _ in range(1000):
        first_obs, rewarded_a = (left_obs, 0) if np.random.randint(2) == 0 else (right_obs, 1)
        a2c_lstm.learn_on_batch(
            make_last_action_reward_conditioned_on_first_obs_rollout(a2c_lstm, first_obs, rewarded_a, rollout))

    trained_dist = get_last_action_dist(a2c_lstm, left_obs, rollout)
    assert trained_dist[0] >= 0.99
    trained_dist = get_last_action_dist(a2c_lstm, right_obs, rollout)
    assert trained_dist[1] >= 0.99


def assert_is_roughly_uniform(dist):
    assert 0.2 < dist[0] < 0.8
    assert 0.2 < dist[1] < 0.8


def get_last_action_dist(policy, obs, rollout):
    dist = None
    for i in range(rollout):
        done = i == rollout - 1
        dist = get_action_probabilities(*policy.compute_actions(obs, [done]))
    return dist


def make_last_action_reward_conditioned_on_first_obs_rollout(policy, first_obs, rewarded_action, rollout_len):
    rollout = RolloutBatch()
    for i in range(rollout_len):
        obs = first_obs if i == 0 else np.zeros_like(first_obs)
        done = [i == rollout_len - 1]
        acts, additional = policy.compute_actions(obs, done)
        reward = 0.0
        if done[0]:
            reward = 1.0 if acts[0] == rewarded_action else -1.0
        elements = dict(actions=acts, rewards=np.array([reward]), dones=np.array(done))
        elements.update(additional)
        rollout.append(elements)

    rollout.set_element('last_obs', np.zeros_like(first_obs))
    return rollout
