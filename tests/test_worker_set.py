import gym
import pytest
import numpy as np

from amarl.policies import RandomPolicy
from amarl.workers import RolloutWorker
from amarl.wrappers import MultipleEnvs


class PolicyStub(RandomPolicy):
    pass


class PolicySpy(PolicyStub):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.recorded_actions = list()

    def compute_actions(self, observations):
        a = super().compute_actions(observations)
        self.recorded_actions.append(a)
        return a


class EnvSpyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.received_resets = 0

    def reset(self, **kwargs):
        self.received_resets += 1
        return super().reset(**kwargs)


@pytest.fixture
def env():
    return MultipleEnvs(lambda: gym.make('CartPole-v0'), 1)


@pytest.fixture
def policy(env):
    return PolicySpy(env.action_space)


@pytest.fixture
def workers(env, policy):
    return RolloutWorker(env, policy)


@pytest.fixture
def env_multi():
    return MultipleEnvs(lambda: EnvSpyWrapper(gym.make('CartPole-v0')), 5)


@pytest.fixture
def workers_multi(env_multi, policy):
    return RolloutWorker(env_multi, policy)


def test_workers_roll_out_provided_environment(workers):
    batch = workers.rollout(10)
    assert len(get_rewards(batch)) == 10


def unwrap_batch(batch):
    return batch['actions'], batch['obs'], batch['rewards'], batch['new_obs'], batch['dones'], batch['infos']


def get_rewards(batch):
    a, o, r, no, d, i = unwrap_batch(batch)
    return r


def test_roll_out_produces_batch_in_canonical_form(workers, env):
    batch = workers.rollout(10)
    actions, obs, rewards, new_obs, dones, infos = unwrap_batch(batch)
    assert actions.shape == (10, 1)
    assert obs.shape == (10, 1, *env.observation_space.shape)
    assert rewards.shape == (10, 1)
    assert new_obs.shape == (10, 1, *env.observation_space.shape)
    assert dones.shape == (10, 1)
    assert infos.shape == (10, 1)


def test_roll_out_batch_contains_all_correct_actions(workers, policy):
    batch = workers.rollout(3)
    assert_actions_equal(get_actions(batch), np.array(policy.recorded_actions))


def get_actions(batch):
    a, o, r, no, d, i = unwrap_batch(batch)
    return a


def assert_actions_equal(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_automatically_reset_terminated_environments(workers):
    batch = workers.rollout(1)
    while not any(get_dones(batch)[:, 0]):
        batch = workers.rollout(1)

    terminated_idx = get_dones(batch)[:, 0].argmax()
    batch = workers.rollout(1)
    assert not get_dones(batch)[terminated_idx]


def get_dones(batch):
    a, o, r, no, d, i = unwrap_batch(batch)
    return d


def test_rollout_produces_correct_batch_shapes_with_multiple_environments(workers_multi, env_multi):
    batch = workers_multi.rollout(10)
    actions, obs, rewards, new_obs, dones, infos = unwrap_batch(batch)
    assert actions.shape == (10, len(env_multi))
    assert obs.shape == (10, len(env_multi), *env_multi.observation_space.shape)
    assert rewards.shape == (10, len(env_multi))
    assert new_obs.shape == (10, len(env_multi), *env_multi.observation_space.shape)
    assert dones.shape == (10, len(env_multi))
    assert infos.shape == (10, len(env_multi))


def test_rollout_selectively_only_resets_terminated_environment_in_multi_env_settings(workers_multi, env_multi):
    batch = workers_multi.rollout(1)
    one_done = get_dones(batch).sum(axis=1) == 1
    while not any(one_done):
        batch = workers_multi.rollout(1)
        one_done = get_dones(batch).sum(axis=1) == 1

    resets = [env.received_resets for env in env_multi.envs]
    assert any([r != resets[0] for r in resets])

