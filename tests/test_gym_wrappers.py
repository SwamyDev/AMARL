import os
from collections import deque

import gym
import pytest
import numpy as np
import torch
from gym.spaces import Box

from amarl.wrappers import MultipleEnvs, active_gym, RenderedObservation, NoOpResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, \
    SignReward, TorchObservation, StackFrames


class EnvStub(gym.Env):
    class ALEStub:
        def __init__(self):
            self._lives = 3

        def lives(self):
            return self._lives

        def set_lives(self, lives):
            self._lives = lives

        def decrement_lives(self):
            self._lives -= 1

    def __init__(self):
        self.observation_space = Box(0, 1, shape=(1, 8))
        self.np_random = np.random.RandomState()
        self.ale = self.ALEStub()
        self._return_obs = None
        self._return_infos = None
        self._return_rewards = None

    def set_lives(self, lives):
        self.ale.set_lives(lives)

    def loose_life(self):
        self.ale.decrement_lives()

    def set_returns_obs(self, obs):
        self._return_obs = deque(obs)

    def set_returns_infos(self, infos):
        self._return_infos = deque(infos)

    def set_returns_rewards(self, rewards):
        self._return_rewards = deque(rewards)

    def get_action_meanings(self):
        return ['NOOP']

    def step(self, action):
        o = np.zeros(self.observation_space.shape) if self._return_obs is None else self._return_obs.popleft()
        d = False if self._return_obs is None else len(self._return_obs) == 0
        i = {'info': "dummy"} if self._return_infos is None else self._return_infos.popleft()
        r = 1 if self._return_rewards is None else self._return_rewards.popleft()
        return o, r, d, i

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class EnvSpy(EnvStub):
    def __init__(self, recorded_close_calls):
        super().__init__()
        self._recorded_close_calls = recorded_close_calls
        self.num_noop_actions_received = 0
        self.num_resets_received = 0
        self.last_action_received = None
        self.actions_received = []

    def step(self, action):
        self.num_noop_actions_received += (action == 0)
        self.actions_received.append(action)
        self.last_action_received = action
        return super().step(action)

    def reset(self):
        self.num_resets_received += 1
        return super().reset()

    def close(self):
        super().close()
        self._recorded_close_calls.append(self)


@pytest.fixture
def recorded_close_calls():
    return list()


@pytest.fixture
def env_spy(recorded_close_calls):
    return EnvSpy(recorded_close_calls)


@pytest.fixture
def multiple_envs(recorded_close_calls):
    return MultipleEnvs(lambda: EnvSpy(recorded_close_calls), 5)


@pytest.fixture
def rendered_env():
    return RenderedObservation(gym.make('CartPole-v0'))


@pytest.fixture
def cube_crash_env():
    return gym.make("CubeCrash-v0")


@pytest.fixture
def noop_env(env_spy):
    return NoOpResetEnv(env_spy)


@pytest.fixture
def skip_env(env_spy):
    return MaxAndSkipEnv(env_spy)


@pytest.fixture
def episodic_env(env_spy):
    return EpisodicLifeEnv(env_spy)


@pytest.fixture
def sign_env(env_spy):
    return SignReward(env_spy)


@pytest.fixture
def torch_env(cube_crash_env):
    return TorchObservation(cube_crash_env)


@pytest.fixture
def stack_size():
    return 3


@pytest.fixture
def stack_env(torch_env, stack_size):
    return StackFrames(torch_env, size=stack_size)


def test_multiple_envs_wrapper_closes_all_created_envs(recorded_close_calls, multiple_envs):
    multiple_envs.close()
    assert len(recorded_close_calls) == 5


def test_gym_context_manager_cleans_up_environment_even_when_error_is_raised(recorded_close_calls, env_spy):
    def interrupted_process():
        with active_gym(env_spy) as env:
            assert env == env_spy
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted_process()

    assert recorded_close_calls[0] == env_spy


def test_rendered_cart_pole_observation_has_correct_observation_space(rendered_env):
    assert rendered_env.observation_space.shape == (3, 40, 40)


def test_rendered_cart_pole_observation_returns_render_image_as_observation(rendered_env):
    assert rendered_env.reset().shape == (3, 40, 40)
    assert unwrap_obs(rendered_env.step(0)).shape == (3, 40, 40)


def unwrap_obs(t):
    return t[0]


def test_noop_reset_skips_random_number_of_frames_at_the_start_via_special_noop(env_spy, noop_env):
    noop_env.unwrapped.np_random.seed(42)
    noop_env.reset()
    assert env_spy.num_noop_actions_received == 7
    env_spy.num_noop_actions_receive = 0
    noop_env.reset()
    assert env_spy.num_noop_actions_received == 27


def test_noop_reset_passes_on_step_unmodified(env_spy, noop_env):
    noop_env.reset()
    noop_env.step(1)
    assert env_spy.last_action_received == 1


def test_max_and_skip_repeats_action_skip_times(env_spy, skip_env):
    skip_env.reset()
    skip_env.step(1)
    assert env_spy.actions_received == [1, 1, 1, 1]


def test_max_and_skip_sums_reward(env_spy, skip_env):
    skip_env.reset()
    assert unwrap_reward(skip_env.step(1)) == 4


def unwrap_reward(t):
    return t[1]


def test_max_and_skip_returns_maximum_of_last_two_frames(env_spy, skip_env):
    env_spy.set_returns_obs([np.zeros(env_spy.observation_space.shape), np.zeros(env_spy.observation_space.shape),
                             np.ones(env_spy.observation_space.shape), np.ones(env_spy.observation_space.shape) * 0.5])
    skip_env.reset()
    assert_obs_eq(unwrap_obs(skip_env.step(1)), np.ones(env_spy.observation_space.shape))


def assert_obs_eq(actual, expected):
    np.testing.assert_equal(actual, expected)


def test_max_and_skip_pass_on_last_info(env_spy, skip_env):
    env_spy.set_returns_infos([{}, {}, {}, {'info': "last"}])
    skip_env.reset()
    assert unwrap_info(skip_env.step(1))['info'] == "last"


def unwrap_info(t):
    return t[-1]


def test_max_and_skip_break_at_done(env_spy, skip_env):
    env_spy.set_returns_obs([np.ones(env_spy.observation_space.shape)])
    skip_env.reset()
    _, r, done, _ = skip_env.step(1)
    assert done and r == 1


def test_episodic_life_is_done_when_player_looses_one_life(env_spy, episodic_env):
    episodic_env.reset()
    episodic_env.step(0)
    env_spy.loose_life()
    assert unwrap_done(episodic_env.step(0))


def unwrap_done(t):
    return t[2]


def test_episodic_life_does_not_reset_when_there_are_lives_left(env_spy, episodic_env):
    episodic_env.reset()
    episodic_env.step(0)
    env_spy.loose_life()
    episodic_env.reset()
    assert env_spy.num_resets_received == 1


def test_episodic_life_performs_noop_on_reset_when_lives_are_left(env_spy, episodic_env):
    episodic_env.reset()
    episodic_env.step(1)
    env_spy.loose_life()
    episodic_env.reset()
    assert env_spy.num_noop_actions_received == 1


def test_episodic_life_resets_on_real_done(env_spy, episodic_env):
    env_spy.set_returns_obs([np.ones(env_spy.observation_space.shape)])
    episodic_env.reset()
    assert unwrap_done(episodic_env.step(1))
    episodic_env.reset()
    assert env_spy.num_resets_received == 2


# This behaviour is necessary because QBert stays at zero lives for a few frames before terminating
def test_episodic_life_waits_for_real_done_on_zero_lives_because(env_spy, episodic_env):
    env_spy.set_returns_obs([np.ones(env_spy.observation_space.shape)] * 3)
    env_spy.set_lives(1)
    episodic_env.reset()
    episodic_env.step(1)
    env_spy.loose_life()

    assert not unwrap_done(episodic_env.step(1))
    assert unwrap_done(episodic_env.step(1))


def test_sign_reward_env_transforms_reward_to_its_signum(env_spy, sign_env):
    env_spy.set_returns_rewards([42, -7])
    sign_env.reset()
    assert unwrap_reward(sign_env.step(0)) == 1
    assert unwrap_reward(sign_env.step(0)) == -1


def test_torch_observation_by_default_wraps_numpy_in_gray_scale_84x84_chw_torch_tensor(torch_env):
    assert torch_env.reset().shape == (1, 84, 84)
    obs = unwrap_obs(torch_env.step(0))
    assert isinstance(obs, torch.Tensor) and obs.shape == (1, 84, 84)


def test_torch_observation_image_size_can_be_configured(cube_crash_env):
    configured_env = TorchObservation(cube_crash_env, image_size=(32, 32))
    assert configured_env.reset().shape == (1, 32, 32)


def test_torch_observation_image_grayscale_can_be_configured(cube_crash_env):
    configured_env = TorchObservation(cube_crash_env, grayscale=False)
    assert configured_env.reset().shape == (3, 84, 84)


def test_torch_observation_updates_observation_space_accordingly(torch_env, cube_crash_env):
    assert torch_env.observation_space.shape == (1, 84, 84)
    altered_image_size = TorchObservation(cube_crash_env, image_size=(16, 16))
    assert altered_image_size.observation_space.shape == (1, 16, 16)
    colored = TorchObservation(cube_crash_env, grayscale=False)
    assert colored.observation_space.shape == (3, 84, 84)


def test_stack_frames_stacks_reset_frame_n_times_on_reset(stack_env, stack_size):
    obs = stack_env.reset()
    assert obs.shape == (stack_size, 84, 84)
    obs = obs.cpu().numpy()
    for i in range(1, stack_size):
        assert_obs_eq(obs[0], obs[i])


def test_stack_frames_appends_next_frame_to_stack_on_step(stack_env, stack_size):
    stack_env.reset()
    obs = unwrap_obs(stack_env.step(1))
    assert obs.shape == (stack_size, 84, 84)
    obs = obs.cpu().numpy()
    for i in range(1, stack_size - 1):
        assert_obs_eq(obs[0], obs[i])

    assert_obs_not_eq(obs[-1], obs[-2])


def assert_obs_not_eq(actual, expected):
    assert (actual != expected).any()
