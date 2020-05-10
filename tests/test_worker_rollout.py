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
        self._info_dict = {}

    def set_additional_information(self, info_dict):
        self._info_dict = info_dict

    def compute_actions(self, observations):
        acts, _ = super().compute_actions(observations)
        if type(acts) is dict:
            info = {key: {rank_id: self._info_dict[key] for rank_id in acts} for key in self._info_dict}
        else:
            info = dict(self._info_dict)
        return acts, info


class PolicySpy(PolicyStub):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.recorded_actions = list()
        self.received_observations = None

    def compute_actions(self, observations):
        a, info = super().compute_actions(observations)
        self.recorded_actions.append(a)
        self.received_observations = observations
        return a, info


class EnvSpyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.received_resets = 0
        self.last_init_observation = None

    def reset(self, **kwargs):
        self.received_resets += 1
        self.last_init_observation = super().reset(**kwargs)
        return self.last_init_observation


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
def workers_irregular(multiple_envs_selective, policy):
    return RolloutWorker(multiple_envs_selective, policy, regularized=False)


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
    return batch['rewards'], batch['dones'], batch['last_obs'], batch['infos']


def test_roll_out_produces_batch_in_canonical_form(workers, env):
    rollout = workers.rollout(10)
    rewards, dones, last_obs, infos = unwrap_rollout(rollout)
    assert len(rewards) == 10 and rewards[0].shape == (1,)
    assert len(dones) == 10 and dones[0].shape == (1,)
    assert last_obs.shape == (1, *env.observation_space.shape)
    assert len(infos) == 10


def test_automatically_reset_terminated_environments(workers):
    rollout = workers.rollout(1)
    while not any(np.array(get_dones(rollout))[:, 0]):
        rollout = workers.rollout(1)

    terminated_idx = np.array(get_dones(rollout))[:, 0].argmax()
    rollout = workers.rollout(1)
    assert not get_dones(rollout)[terminated_idx]


def get_dones(rollout):
    r, d, l, i = unwrap_rollout(rollout)
    return d


def test_rollout_produces_correct_batch_shapes_with_multiple_environments(workers_multi, env_multi):
    rollout = workers_multi.rollout(10)
    rewards, dones, last_obs, infos = unwrap_rollout(rollout)
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


def test_rollout_selectively_sets_init_observation_in_multi_env_settings_after_reset(workers_multi, env_multi, policy):
    rollout = workers_multi.rollout(1)
    while not any(get_dones(rollout)[0]):
        rollout = workers_multi.rollout(1)

    reset_envs = [(i, e) for i, e in enumerate(env_multi.envs) if e.received_resets == 2]
    workers_multi.rollout(1)
    for idx, env in reset_envs:
        assert_tensors_equal(policy.received_observations[idx], env.last_init_observation.astype(np.float32))


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


def test_irregular_worker_does_not_advance_terminated_environments(workers_irregular, multiple_envs_selective, policy):
    setup_env_lengths(multiple_envs_selective, lengths=[9, 2, 6, 4, 1])

    rollout = workers_irregular.rollout(5)

    assert_irregular_rollout_shape(rollout, (5, 2, 5, 4, 1))


def setup_env_lengths(multiple_envs_selective, lengths):
    for i, l in enumerate(lengths):
        multiple_envs_selective.envs[i].set_is_done_at_step(l)


def assert_irregular_rollout_shape(rollout, expected_shape):
    for i, s in enumerate(expected_shape):
        assert len(rollout.of(i)) == s, rollout.of(i)


def test_irregular_worker_resets_environments_after_rollout(workers_irregular, multiple_envs_selective, policy):
    setup_env_lengths(multiple_envs_selective, lengths=[9, 2, 6, 4, 1])

    workers_irregular.rollout(5)

    assert_all_attributes_eq(multiple_envs_selective.envs, 'num_resets_received', 1, 2, 1, 2, 2)


def assert_all_attributes_eq(envs, attribute, *values):
    for i, (e, v) in enumerate(zip(envs, values)):
        assert getattr(e, attribute) == v, f"{e}[{i}].{attribute}"


def test_irregular_worker_sets_last_observations_only_for_non_terminated_envs(workers_irregular,
                                                                              multiple_envs_selective,
                                                                              policy):
    setup_envs(multiple_envs_selective, lengths=[9, 2, 6, 4, 5])

    rollout = workers_irregular.rollout(5)

    obs_shape = multiple_envs_selective.observation_space.shape
    assert_irregular_rollout_key(rollout, 'last_obs',
                                 {0: make_obs(obs_shape, rank_id=0), 2: make_obs(obs_shape, rank_id=2)})


def setup_envs(multiple_envs_selective, lengths):
    for rank_id, length in enumerate(lengths):
        setup_env(multiple_envs_selective, rank_id, length)


def setup_env(multi_env, rank_id, length):
    shape = multi_env.observation_space.shape
    multi_env.envs[rank_id].set_returns_obs(make_observations(shape, rank_id, length))
    multi_env.envs[rank_id].set_returns_rewards([rank_id] * length)
    multi_env.envs[rank_id].set_returns_infos([{'my_rank': rank_id}] * length)


def make_observations(shape, rank_id, length):
    return [make_obs(shape, rank_id)] * length


def make_obs(shape, rank_id):
    return np.ones(shape) * rank_id


def assert_irregular_rollout_key(rollout, key, data_per_rank):
    for rank_id in range(rollout.num_workers):
        if rank_id in data_per_rank:
            assert_tensors_equal(rollout.of(rank_id)[key], data_per_rank[rank_id])
        else:
            assert key not in rollout.of(rank_id)


def test_irregular_worker_collects_data_from_each_worker_correctly(workers_irregular, multiple_envs_selective, policy):
    setup_envs(multiple_envs_selective, lengths=[9, 2, 6, 4, 5])

    rollout = workers_irregular.rollout(5)

    assert_selective_rollout_of(rollout, rank_id=0, length=5, is_terminated=False)
    assert_selective_rollout_of(rollout, rank_id=1, length=2, is_terminated=True)
    assert_selective_rollout_of(rollout, rank_id=2, length=5, is_terminated=False)
    assert_selective_rollout_of(rollout, rank_id=3, length=4, is_terminated=True)
    assert_selective_rollout_of(rollout, rank_id=4, length=5, is_terminated=True)


def unwrap_env_data(batch):
    return batch['rewards'], batch['dones'], batch['infos']


def assert_selective_rollout_of(rollout, rank_id, length, is_terminated):
    rewards, dones, infos = unwrap_env_data(rollout.of(rank_id))
    assert_tensors_equal(rewards, [rank_id] * length)
    dones = [False] * length
    if is_terminated:
        dones[-1] = True
    assert_tensors_equal(dones, dones)


def test_irregular_worker_properly_adds_additional_info_from_policy(workers_irregular, multiple_envs_selective, policy):
    policy.set_additional_information({'more': 'info'})
    setup_envs(multiple_envs_selective, lengths=[9, 2, 6, 4, 5])

    rollout = workers_irregular.rollout(5)

    assert_irregular_rollout_key(rollout, 'more', {
        0: ['info'] * 5,
        1: ['info'] * 2,
        2: ['info'] * 5,
        3: ['info'] * 4,
        4: ['info'] * 5,
    })
