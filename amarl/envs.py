import gym
import numpy as np
from gym.spaces import Box, Discrete


class MemoryEnv(gym.Env):
    def  __init__(self, episode_length=3):
        self.observation_space = Box(0, 1, shape=(8,), dtype=np.float32)
        self.action_space = Discrete(2)
        self.episode_length = episode_length
        self._current_step = 0
        self._random_state = np.random.RandomState()
        self._right_final_action = None

    def seed(self, seed=None):
        self._random_state = np.random.RandomState(seed=seed)

    def step(self, action):
        self._current_step += 1
        done = self._current_step == self.episode_length
        if done:
            r = 1 if action == self._right_final_action else -1
        else:
            r = 0
        return np.zeros(self.observation_space.shape), r, done, {}

    def reset(self):
        self._current_step = 0
        init_obs = (self._random_state.uniform(0, 1, self.observation_space.shape) > 0.5).astype(np.float32)
        self._right_final_action = 1 if np.sum(init_obs) > 0.5 * init_obs.size else 0
        return init_obs

    def render(self, mode='human'):
        pass


class TellDoneEnv(gym.Env):
    def __init__(self, max_len=5):
        self.observation_space = Box(0, 1, shape=(max_len,), dtype=np.float32)
        self.action_space = Discrete(2)
        self._max_len = max_len
        self._steps = 0
        self._random_state = np.random.RandomState()
        self._final_step = None
        self._init_obs = None

    @property
    def max_len(self):
        return self._max_len

    @property
    def final_step(self):
        return self._final_step

    def seed(self, seed=None):
        self._random_state = np.random.RandomState(seed=seed)

    def step(self, action):
        diff = self._final_step - self._steps
        if diff < 0:
            raise gym.error.ResetNeeded()

        self._steps += 1
        done = diff == 0

        reward = 0
        if action == 1:
            reward = 1 / (diff + 1)
            done = True

        return self._init_obs, reward, done, {}

    def reset(self):
        self._final_step = self._random_state.randint(self.max_len)
        self._init_obs = np.zeros(self.observation_space.shape).astype(self.observation_space.dtype)
        self._init_obs[self._final_step] = 1.0
        self._steps = 0
        return self._init_obs

    def render(self, mode='human'):
        pass
