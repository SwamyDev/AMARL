import gym
import numpy as np

from amarl.processing import get_cart_pos_normalized, proc_screen


class MultipleEnvs(gym.Env):
    def __init__(self, env_factory, num_envs):
        self._num_envs = num_envs
        self._envs = [env_factory() for _ in range(self._num_envs)]
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        self.reward_range = self._envs[0].reward_range
        self.metadata = self._envs[0].metadata

    @property
    def envs(self):
        return self._envs

    def step(self, action: np.array):
        obs = np.empty((len(self), *self.observation_space.shape), dtype=self.observation_space.dtype)
        rewards = np.empty((len(self),), dtype=np.float)
        dones = np.empty((len(self),), dtype=np.bool)
        infos = np.empty((len(self),), dtype=np.object)
        for idx in range(len(self)):
            o, r, d, i = self._envs[idx].step(action[idx])
            obs[idx] = o
            rewards[idx] = r
            dones[idx] = d
            infos[idx] = i

        return obs, rewards, dones, infos

    def reset(self):
        obs = np.empty((len(self), *self.observation_space.shape), dtype=self.observation_space.dtype)
        for i in range(len(self)):
            obs[i] = self._envs[i].reset()

        return obs

    def render(self, mode='human'):
        res = list()
        for e in self._envs:
            res.append(e.render(mode))
        return res

    def __len__(self):
        return self._num_envs


class RenderedObservation(gym.ObservationWrapper):
    def observation(self, observation):
        screen = self.env.render(mode='rgb_array')
        cart_pos = get_cart_pos_normalized(self.env, screen.shape[1])
        return proc_screen(screen, cart_pos)
