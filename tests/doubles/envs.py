from collections import deque

import gym
import numpy as np


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

    def __init__(self, obs_space, default_obs):
        self.observation_space = obs_space
        self._default_obs = default_obs
        self.np_random = np.random.RandomState()
        self.ale = self.ALEStub()
        self._return_obs = None
        self._return_infos = None
        self._return_rewards = None
        self._is_done_at_step = None
        self._steps = 0

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

    def set_is_done_at_step(self, step):
        self._is_done_at_step = step

    def get_action_meanings(self):
        return ['NOOP']

    def step(self, action):
        self._steps += 1
        o = np.zeros(self.observation_space.shape) if self._return_obs is None else self._return_obs.popleft()
        i = {'info': "dummy"} if self._return_infos is None else self._return_infos.popleft()
        r = 1 if self._return_rewards is None else self._return_rewards.popleft()
        d = False if self._return_obs is None else len(self._return_obs) == 0
        d = d if self._return_rewards is None else len(self._return_rewards) == 0
        d = d if self._is_done_at_step is None else self._steps == self._is_done_at_step
        return o, r, d, i

    def reset(self):
        self._steps = 0
        return self._default_obs

    def render(self, mode='human'):
        pass


class EnvSpy(EnvStub):
    def __init__(self, obs_space, default_obs, recorded_close_calls, recorded_render_calls):
        super().__init__(obs_space, default_obs)
        self._recorded_close_calls = recorded_close_calls
        self._recorded_render_calls = recorded_render_calls
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

    def render(self, mode='human'):
        self._recorded_render_calls.append(mode)
        return super().render(mode)

    def close(self):
        super().close()
        self._recorded_close_calls.append(self)