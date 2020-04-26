import gym
import numpy as np
import pytest

from amarl.envs import MemoryEnv, TellDoneEnv


@pytest.fixture
def mem_env():
    return MemoryEnv()


@pytest.fixture
def tell_env():
    return TellDoneEnv()


def test_mem_observation_and_action_space_is_set_correctly(mem_env):
    assert mem_env.observation_space.shape == (8,)
    assert mem_env.action_space.n == 2


def test_mem_initial_observation_in_default_shape(mem_env):
    obs = mem_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == mem_env.observation_space.shape


def test_mem_reward_is_zero_on_any_action_before_final_step(mem_env):
    mem_env.reset()
    assert unpack_reward(mem_env.step(0)) == 0
    assert unpack_reward(mem_env.step(1)) == 0


def unpack_reward(args):
    return args[1]


def test_mem_observations_before_final_step_are_empty(mem_env):
    mem_env.reset()
    assert_obs(unpack_obs(mem_env.step(1)), np.zeros(mem_env.observation_space.shape))
    assert_obs(unpack_obs(mem_env.step(2)), np.zeros(mem_env.observation_space.shape))


def unpack_obs(args):
    return args[0]


def assert_obs(actual, expected):
    np.testing.assert_equal(actual, expected)


def test_mem_final_step_is_done_after_n_steps(mem_env):
    for _ in range(2):
        mem_env.reset()
        assert not unpack_done(mem_env.step(0))
        assert not unpack_done(mem_env.step(0))
        assert unpack_done(mem_env.step(0))


def unpack_done(args):
    return args[2]


def test_mem_final_step_reward_is_selecting_the_right_action_predicated_on_the_amount_of_ones_of_initial_state(mem_env):
    for _ in range(10):
        init_obs = mem_env.reset()
        for _ in range(mem_env.episode_length - 1):
            mem_env.step(0)

        right_action = calc_right_action_from(init_obs)
        assert unpack_reward(mem_env.step(right_action)) == 1


def calc_right_action_from(init_obs):
    return 1 if np.sum(init_obs) > (init_obs.size * 0.5) else 0


def test_mem_final_step_reward_is_negative_when_the_wrong_action_is_selected(mem_env):
    for _ in range(10):
        init_obs = mem_env.reset()
        for _ in range(mem_env.episode_length - 1):
            mem_env.step(0)

        wrong_action = 0 if np.sum(init_obs) > (init_obs.size * 0.5) else 1
        assert unpack_reward(mem_env.step(wrong_action)) == -1


def test_mem_info_is_empty_dict(mem_env):
    mem_env.reset()
    assert mem_env.step(0)[-1] == {}


def test_mem_episode_length_is_configurable():
    env = MemoryEnv(episode_length=1)
    env.reset()
    assert unpack_done(env.step(0))
    init_obs = env.reset()
    assert unpack_reward(env.step(calc_right_action_from(init_obs))) == 1


def test_tell_observation_and_action_space_is_set_correctly(tell_env):
    assert tell_env.observation_space.shape == (tell_env.max_len,)
    assert tell_env.action_space.n == 2


def test_tell_initial_observation_in_default_shape(tell_env):
    obs = tell_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == tell_env.observation_space.shape


def test_tell_reward_is_zero_on_noop_before_final_step(tell_env):
    tell_env.seed(42)
    tell_env.reset()
    assert unpack_reward(tell_env.step(0)) == 0
    assert unpack_reward(tell_env.step(0)) == 0


def test_tell_reward_is_negative_proportional_to_final_step_on_premature_tell_action(tell_env):
    tell_env.seed(42)
    tell_env.reset()
    assert unpack_reward(tell_env.step(1)) == 1 / (tell_env.final_step + 1)

    tell_env.seed(42)
    tell_env.reset()
    tell_env.step(0)
    assert unpack_reward(tell_env.step(1)) == 1 / tell_env.final_step


def test_tell_env_is_done_once_tell_action_is_used(tell_env):
    tell_env.seed(42)
    tell_env.reset()
    assert unpack_done(tell_env.step(1))


def test_tell_observations_before_final_step_are_the_same_as_the_first(tell_env):
    tell_env.seed(42)
    first = tell_env.reset()
    assert_obs(unpack_obs(tell_env.step(0)), first)
    assert_obs(unpack_obs(tell_env.step(0)), first)


def test_tell_final_step_is_encoded_as_one_in_observation(tell_env):
    for _ in range(3):
        steps_till_done, = np.where(tell_env.reset() == 1)
        for _ in range(steps_till_done[0]):
            assert not unpack_done(tell_env.step(0))
        assert unpack_done(tell_env.step(0))


def test_tell_steps_until_done_is_different_on_each_reset(tell_env):
    tell_env.seed(42)
    assert_obs_nq(tell_env.reset(), tell_env.reset())


def assert_obs_nq(expected, actual):
    assert not np.array_equal(actual, expected)


def test_tell_reward_is_one_when_tell_done_action_is_selected_on_last_step(tell_env):
    steps_till_done, = np.where(tell_env.reset() == 1)
    for _ in range(steps_till_done[0]):
        tell_env.step(0)
    assert unpack_reward(tell_env.step(1)) == 1


def test_tell_info_is_empty_dict(tell_env):
    tell_env.reset()
    assert tell_env.step(0)[-1] == {}


def test_tell_raises_exception_when_it_has_not_been_properly_reset(tell_env):
    tell_env.seed(42)
    steps_till_done, = np.where(tell_env.reset() == 1)
    for _ in range(steps_till_done[0] + 1):
        tell_env.step(0)

    with pytest.raises(gym.error.ResetNeeded):
        tell_env.step(0)


def test_tell_episode_length_is_configurable():
    env = TellDoneEnv(max_len=8)
    assert env.observation_space.shape == (8,)
