import gym
import pytest
import numpy as np
import torch

from amarl.messenger import Message, subscription_to
from amarl.policies import RandomPolicy
from amarl.workers import RolloutWorker
from amarl.wrappers import MultipleEnvs


class PolicyStub(RandomPolicy):
    def __init__(self, action_space):
        super().__init__(action_space)
        self._info_dict = None

    def set_additional_information(self, info_dict):
        self._info_dict = info_dict

    def compute_actions(self, observations):
        acts, _ = super().compute_actions(observations)
        return torch.from_numpy(acts), self._info_dict


class PolicySpy(PolicyStub):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.recorded_actions = list()

    def compute_actions(self, observations):
        a, info = super().compute_actions(observations)
        self.recorded_actions.append(a)
        return a, info


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


@pytest.fixture
def info_listener():
    class _InfoListener:
        def __init__(self):
            self.num_received_infos = 0
            self.last_received_message = None

        def __call__(self, **infos):
            assert infos is not None
            self.num_received_infos += 1
            self.last_received_message = infos

    return _InfoListener()


def test_workers_roll_out_provided_environment(workers):
    rollout = workers.rollout(10)
    assert len(rollout) == 10


def unwrap_rollout(batch):
    return batch['actions'], batch['rewards'], batch['dones'], batch['last_obs'], batch['infos']


def get_rewards(batch):
    a, r, d, l, i = unwrap_rollout(batch)
    return r


def test_roll_out_produces_batch_in_canonical_form(workers, env):
    rollout = workers.rollout(10)
    actions, rewards, dones, last_obs, infos = unwrap_rollout(rollout)
    assert len(actions) == 10 and actions[0].shape == (1,)
    assert len(rewards) == 10 and rewards[0].shape == (1,)
    assert len(dones) == 10 and dones[0].shape == (1,)
    assert last_obs.shape == (1, *env.observation_space.shape)
    assert len(infos) == 10


def test_roll_out_batch_contains_all_correct_actions(workers, policy):
    rollout = workers.rollout(3)
    assert_actions_equal(get_actions(rollout), np.array(policy.recorded_actions))


def get_actions(rollout):
    a, r, d, l, i = unwrap_rollout(rollout)
    return a


def assert_actions_equal(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_automatically_reset_terminated_environments(workers):
    rollout = workers.rollout(1)
    while not any(np.array(get_dones(rollout))[:, 0]):
        rollout = workers.rollout(1)

    terminated_idx = np.array(get_dones(rollout))[:, 0].argmax()
    rollout = workers.rollout(1)
    assert not get_dones(rollout)[terminated_idx]


def get_dones(rollout):
    a, r, d, l, i = unwrap_rollout(rollout)
    return d


def test_rollout_produces_correct_batch_shapes_with_multiple_environments(workers_multi, env_multi):
    rollout = workers_multi.rollout(10)
    actions, rewards, dones, last_obs, infos = unwrap_rollout(rollout)
    assert actions[0].shape == (len(env_multi),)
    assert rewards[0].shape == (len(env_multi),)
    assert dones[0].shape == (len(env_multi),)
    assert last_obs.shape == (len(env_multi), *env_multi.observation_space.shape)
    assert infos[0].shape == (len(env_multi),)


def test_rollout_selectively_only_resets_terminated_environment_in_multi_env_settings(workers_multi, env_multi):
    rollout = workers_multi.rollout(1)
    one_done = np.array(get_dones(rollout)).sum(axis=1) == 1
    while not any(one_done):
        rollout = workers_multi.rollout(1)
        one_done = np.array(get_dones(rollout)).sum(axis=1) == 1

    resets = [env.received_resets for env in env_multi.envs]
    assert any([r != resets[0] for r in resets])


def test_rollout_add_passes_on_additional_information_of_action_compute(workers, env, policy):
    more_info = {'lstm_hidden_state': np.random.rand(2, 10)}
    policy.set_additional_information(more_info)
    rollout = workers.rollout(5)
    assert len(rollout['lstm_hidden_state']) == 5
    assert_tensors_equal(rollout['lstm_hidden_state'][0], more_info['lstm_hidden_state'])


def assert_tensors_equal(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_worker_broadcasts_infos_as_message(workers, info_listener):
    with subscription_to(Message.TRAINING, info_listener):
        workers.rollout(5)
        assert info_listener.num_received_infos == 5


def test_worker_total_steps_account_for_each_environment(workers_multi, env_multi):
    workers_multi.rollout(5)
    assert workers_multi.total_steps == 5 * len(env_multi)


def test_worker_with_multiple_environments_performs(workers):
    workers.rollout(5)
    assert workers.total_steps == 5


def test_worker_sends_terminated_message_when_done(workers_multi, info_listener):
    with subscription_to(Message.ENV_TERMINATED, info_listener):
        rollout = workers_multi.rollout(1)
        dones = get_dones(rollout)[0]
        while not any(dones):
            rollout = workers_multi.rollout(1)
            dones = get_dones(rollout)[0]
        idx_dones = get_idx_done(dones)
        assert_indices_equal(info_listener.last_received_message['index_dones'], idx_dones)


def get_idx_done(dones):
    return [i for i in np.where(dones)[0]]


def assert_indices_equal(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_worker_stop_rollout_when_environment_has_terminated_if_configured_to_do_so(env, policy):
    stopping_worker = RolloutWorker(env, policy, stop_on_done=True)

    rollout = stopping_worker.rollout(5)
    steps = len(rollout)
    while len(rollout) == 5:
        rollout = stopping_worker.rollout(5)
        steps += len(rollout)

    assert stopping_worker.total_steps == steps
