import gym
import pytest
import numpy as np
import torch

from amarl.wrappers import MultipleEnvs, active_gym, RenderedObservation, NoOpResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, \
    SignReward, TorchObservation, StackFrames, OriginalReturnWrapper
from tests import assert_selective_obs_eq, assert_obs_eq
from tests.doubles.envs import EnvSpy


@pytest.fixture
def env_spy(obs_space, default_obs, recorded_close_calls, recorded_render_calls):
    return EnvSpy(obs_space, default_obs, recorded_close_calls, recorded_render_calls)


@pytest.fixture
def multiple_envs(obs_space, default_obs, recorded_close_calls, recorded_render_calls):
    return MultipleEnvs(lambda: EnvSpy(obs_space, default_obs, recorded_close_calls, recorded_render_calls), 5)


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


@pytest.fixture
def return_env(env_spy):
    return OriginalReturnWrapper(env_spy)


def test_multiple_envs_wrapper_closes_all_created_envs(recorded_close_calls, multiple_envs):
    multiple_envs.close()
    assert len(recorded_close_calls) == 5


def test_multiple_envs_wrapper_passes_on_render_call(recorded_render_calls, multiple_envs):
    multiple_envs.render()
    assert len(recorded_render_calls) == 5


def test_multiple_envs_provides_number_of_active_envs_and_indices_of_terminated_envs(multiple_envs_selective):
    multiple_envs_selective.reset()
    assert multiple_envs_selective.num_active_envs == 5
    multiple_envs_selective.envs[2].set_is_done_at_step(1)
    multiple_envs_selective.step({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    assert multiple_envs_selective.num_active_envs == 4
    assert multiple_envs_selective.terminated_env_ids == {2}


def test_multiple_envs_pass_on_actions_selectively(multiple_envs_selective):
    multiple_envs_selective.reset()
    multiple_envs_selective.step({0: 1, 2: 0, 3: 1})
    assert multiple_envs_selective.envs[0].actions_received == [1]
    assert multiple_envs_selective.envs[1].actions_received == []
    assert multiple_envs_selective.envs[2].actions_received == [0]
    assert multiple_envs_selective.envs[3].actions_received == [1]
    assert multiple_envs_selective.envs[4].actions_received == []


def test_multiple_envs_raise_assertion_error_when_trying_to_set_an_action_for_a_terminated_env(multiple_envs_selective):
    multiple_envs_selective.reset()
    multiple_envs_selective.envs[2].set_is_done_at_step(1)
    multiple_envs_selective.step({2: 0})
    with pytest.raises(MultipleEnvs.TerminatedEnvironmentError):
        multiple_envs_selective.step({2: 0})


def test_multiple_envs_return_data_selectively_if_selective_actions_are_passed(multiple_envs_selective, default_obs):
    multiple_envs_selective.reset()
    multiple_envs_selective.envs[2].set_is_done_at_step(1)

    obs, rs, dones, infos = multiple_envs_selective.step({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    assert_selective_obs_eq(obs, {0: default_obs, 1: default_obs, 2: default_obs, 3: default_obs, 4: default_obs})
    assert rs == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    assert dones == {0: False, 1: False, 2: True, 3: False, 4: False}
    assert infos == {0: {'info': "dummy"}, 1: {'info': "dummy"}, 2: {'info': "dummy"}, 3: {'info': "dummy"},
                     4: {'info': "dummy"}}

    obs, rs, dones, infos = multiple_envs_selective.step({1: 0, 4: 0})
    assert_selective_obs_eq(obs, {1: default_obs, 4: default_obs})
    assert rs == {1: 1, 4: 1}
    assert dones == {1: False, 4: False}
    assert infos == {1: {'info': "dummy"}, 4: {'info': "dummy"}}


def test_multiple_envs_reactivates_environment_on_reset(multiple_envs_selective, default_obs):
    multiple_envs_selective.reset()
    multiple_envs_selective.envs[2].set_is_done_at_step(1)

    multiple_envs_selective.step({2: 0})
    assert_obs_eq(multiple_envs_selective.reset_env(2), default_obs)
    assert multiple_envs_selective.num_active_envs == 5
    assert multiple_envs_selective.envs[2].num_resets_received == 2

    multiple_envs_selective.step({2: 0})
    assert_selective_obs_eq(multiple_envs_selective.reset(),
                            {0: default_obs, 1: default_obs, 2: default_obs, 3: default_obs, 4: default_obs})
    assert multiple_envs_selective.num_active_envs == 5
    assert multiple_envs_selective.envs[2].num_resets_received == 3


def test_gym_context_manager_cleans_up_environment_even_when_error_is_raised(recorded_close_calls, env_spy):
    def interrupted_process():
        with active_gym(env_spy) as env:
            assert env == env_spy
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted_process()

    assert recorded_close_calls[0] == env_spy


@pytest.mark.rendered
def test_rendered_cart_pole_observation_has_correct_observation_space(rendered_env):
    assert rendered_env.observation_space.shape == (3, 40, 40)


@pytest.mark.rendered
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


def test_noop_resets_again_when_environment_resets_during_noop_loop(env_spy, noop_env):
    noop_env.unwrapped.np_random.seed(42)
    env_spy.set_is_done_at_step(4)
    noop_env.reset()
    assert env_spy.num_resets_received == 2


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


def test_episodic_return_is_none_when_environment_is_not_done(return_env):
    return_env.reset()
    assert unwrap_info(return_env.step(0))['episodic_return'] is None


def test_episodic_return_is_accumulated_reward_when_environment_is_done(return_env, env_spy):
    rewards = [1, 2, -1]
    env_spy.set_returns_rewards(rewards)

    return_env.reset()
    for _ in range(len(rewards) - 1):
        return_env.step(0)

    assert unwrap_info(return_env.step(0))['episodic_return'] == 2


def test_episodic_return_is_accumulated_reward_is_reset_after_done(return_env, env_spy):
    rewards = [1, 2, -1]
    env_spy.set_returns_rewards(rewards)

    return_env.reset()
    for _ in range(len(rewards)):
        return_env.step(0)

    rewards = [1, 0, -2]
    env_spy.set_returns_rewards(rewards)
    for _ in range(len(rewards) - 1):
        return_env.step(0)

    assert unwrap_info(return_env.step(0))['episodic_return'] == -1


def test_episodic_return_passes_on_reset_call(return_env, env_spy):
    return_env.reset()
    assert env_spy.num_resets_received == 1
