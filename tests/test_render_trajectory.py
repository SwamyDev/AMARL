import pytest
import numpy as np
import torch

from amarl.policies import Policy
from amarl.visualisation import render_trajectory


class EnvStub:
    def __init__(self, frames):
        self._frames = frames
        self._n_steps = 0

    def reset(self):
        return torch.ones(2)

    def step(self, a):
        assert a.shape[0] == 2
        self._n_steps += 1
        return torch.ones(2), 1, len(self._frames) == self._n_steps, dict(episodic_return=2)

    def render(self, mode):
        assert mode == 'rgb_array'
        return self._frames[self._n_steps - 1]

    def close(self):
        pass


class AgentDummy(Policy):
    def compute_actions(self, observations):
        assert observations.shape[0] == 1
        return np.zeros((1, 2)), None

    def learn_on_batch(self, batch):
        pass


@pytest.fixture
def frames():
    return np.random.random((16, 64, 32, 3))


@pytest.fixture
def env(frames):
    return EnvStub(frames)


@pytest.fixture
def agent():
    return AgentDummy()


def test_render_trajectory_video_frames(env, frames, agent):
    video, _ = render_trajectory(env, agent)
    np.testing.assert_array_equal(video, np.expand_dims(frames.transpose(0, 3, 1, 2), axis=0))


def test_render_trajectory_returns_reward_infos(env, frames, agent):
    _, reward_infos = render_trajectory(env, agent, reward_infos=['episodic_return'])
    assert reward_infos['total_reward'] == 16
    assert reward_infos['total_episodic_return'] == 32
