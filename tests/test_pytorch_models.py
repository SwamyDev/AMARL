import pytest
import torch as th
import numpy as np
from gym.spaces import Box, Discrete

from amarl.models import ActorCriticBaseline, LSTM_SIZE


@pytest.fixture
def obs_space():
    return Box(0.0, 0.0, (15, 15, 3), np.float32)


@pytest.fixture
def num_outputs():
    return 16


@pytest.fixture
def baseline(obs_space, num_outputs):
    return ActorCriticBaseline(obs_space, Discrete(8), num_outputs, {}, "ActorCriticBaseline")


def test_actor_critic_baseline_produces_correct_initial_state(baseline):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    baseline.to(device=device)
    state = baseline.get_initial_state()
    th.testing.assert_allclose(state[0], th.zeros(LSTM_SIZE, dtype=th.float32, device=device))
    th.testing.assert_allclose(state[1], th.zeros(LSTM_SIZE, dtype=th.float32, device=device))


def test_actor_critic_baseline_produces_correct_action_logits_and_new_state(baseline, obs_space, num_outputs):
    batch_size = 4
    sequence_len = 16

    observations_batch = th.rand(batch_size, sequence_len, obs_space.shape[2], obs_space.shape[1], obs_space.shape[0])
    state = [th.zeros(batch_size, LSTM_SIZE), th.zeros(batch_size, LSTM_SIZE)]

    logits, new_state = baseline.forward_rnn(observations_batch, state, [sequence_len] * batch_size)

    assert logits.shape == (batch_size, sequence_len, num_outputs)
    assert state[0].shape == new_state[0].shape
    assert state[1].shape == new_state[1].shape


def test_actor_critic_baseline_produces_a_value_estimate(baseline, obs_space):
    batch_size = 8
    sequence_len = 4

    observations_batch = th.rand(batch_size, sequence_len, obs_space.shape[2], obs_space.shape[1], obs_space.shape[0])
    state = [th.zeros(batch_size, LSTM_SIZE), th.zeros(batch_size, LSTM_SIZE)]

    baseline.forward_rnn(observations_batch, state, [sequence_len] * batch_size)
    assert baseline.value_function().shape == (batch_size * sequence_len,)
