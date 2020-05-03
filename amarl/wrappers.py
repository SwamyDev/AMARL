from collections import deque

import gym
import numpy as np
import torch
import torchvision.transforms as T

from contextlib import contextmanager
from gym.spaces import Box
from PIL import Image
from amarl.processing import get_cart_pos_normalized, proc_screen


class MultipleEnvs(gym.Env):
    class TerminatedEnvironmentError(gym.error.Error):
        pass

    def __init__(self, env_factory, num_envs):
        self._num_envs = num_envs
        self._envs = [env_factory() for _ in range(self._num_envs)]
        seed = np.random.randint(int(1e9))
        for i, e in enumerate(self._envs):
            e.seed(seed + i)
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        self.reward_range = self._envs[0].reward_range
        self.metadata = self._envs[0].metadata
        self._terminated_envs = set()

    @property
    def envs(self):
        return self._envs

    def step(self, action):
        is_selective = type(action) is dict
        dones, infos, obs, rewards = self._make_return_objects(is_selective)
        rank_ids = action if is_selective else range(len(self))
        for idx in rank_ids:
            if idx in self.terminated_env_ids:
                raise self.TerminatedEnvironmentError(f"Passing an action to terminated environment: {idx}")
            o, r, d, i = self._envs[idx].step(action[idx])
            obs[idx] = o
            rewards[idx] = r
            dones[idx] = d
            infos[idx] = i
            if d:
                self._terminated_envs.add(idx)

        return obs, rewards, dones, infos

    def _make_return_objects(self, is_selective):
        if is_selective:
            obs, rewards, dones, infos = dict(), dict(), dict(), dict()
        else:
            obs = np.empty((len(self), *self.observation_space.shape), dtype=self.observation_space.dtype)
            rewards = np.empty((len(self),), dtype=np.float)
            dones = np.empty((len(self),), dtype=np.bool)
            infos = np.empty((len(self),), dtype=np.object)
        return dones, infos, obs, rewards

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

    def close(self):
        for env in self._envs:
            env.close()

        super().close()

    @property
    def terminated_env_ids(self):
        return self._terminated_envs

    @property
    def num_active_envs(self):
        return len(self) - len(self.terminated_env_ids)

    def __len__(self):
        return self._num_envs


class RenderedObservation(gym.Wrapper):
    def __init__(self, env):
        super(RenderedObservation, self).__init__(env)
        self.observation_space = Box(-np.inf, np.inf, shape=(3, 40, 40))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        screen = self.env.render(mode='rgb_array')
        cart_pos = get_cart_pos_normalized(self.env, screen.shape[1])
        return proc_screen(screen, cart_pos)


class NoOpResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoOpResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        info = None
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class SignReward(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class TorchObservation(gym.ObservationWrapper):
    def __init__(self, env, image_size=None, grayscale=True):
        super(TorchObservation, self).__init__(env)
        image_size = image_size or (84, 84)
        channels = 1 if grayscale else 3
        self.observation_space = Box(0, 1, shape=(channels, *image_size))

        self.resize = T.Compose(self._make_image_pipe_operations(grayscale, image_size))
        self._last_frame = None

    @staticmethod
    def _make_image_pipe_operations(grayscale, image_size):
        operations = list()
        operations.append(T.ToPILImage())
        if grayscale:
            operations.append(T.Grayscale())
        operations.append(T.Resize(image_size, interpolation=Image.CUBIC))
        operations.append(T.ToTensor())
        return operations

    def observation(self, observation):
        return self.resize(observation)


class StackFrames(gym.Wrapper):
    def __init__(self, env, size=4):
        super(StackFrames, self).__init__(env)
        self._size = size
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=1, shape=(shp[0] * self._size, shp[1], shp[2]),
                                     dtype=env.observation_space.dtype)
        self._frames = deque([], maxlen=self._size)
        self._cur_size = 0

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._size):
            self._enqueue(obs)
        return self._get_obs()

    def _enqueue(self, obs):
        self._frames.append(obs)
        self._cur_size = min(self._cur_size + 1, self._size)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._enqueue(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert self._cur_size == self._size
        return torch.cat(list(self._frames), dim=0)


@contextmanager
def active_gym(gym):
    try:
        yield gym
    finally:
        gym.close()
