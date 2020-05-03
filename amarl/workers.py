import time
import logging
from pathlib import Path

import numpy as np

from amarl.messenger import broadcast, Message
from amarl.metrics import PerformanceMeasure

logger = logging.getLogger(__name__)


class RolloutBatch:
    def __init__(self):
        self._data = dict()
        self._length = 0

    def __getitem__(self, item):
        return self._data[item]

    def append(self, new_elements):
        for k in new_elements:
            if k not in self._data:
                self._data[k] = list()
            self._data[k].append(new_elements[k])

        self._length += 1

    def set_element(self, key, new_element):
        self._data[key] = new_element

    def __len__(self):
        return self._length


class RolloutWorker:
    def __init__(self, env, policy, stop_on_done=False):
        self._env = env
        self._policy = policy
        self._stop_on_done = stop_on_done
        self._last_obs = self._env.reset()
        self._total_steps = 0

    @property
    def total_steps(self):
        return self._total_steps

    def rollout(self, horizon):
        rollout = RolloutBatch()
        for _ in range(horizon):
            actions, additional = self._policy.compute_actions(self._last_obs)
            obs, rewards, dones, infos = self._env.step(actions.cpu().numpy())
            broadcast(Message.TRAINING, infos=infos)
            self._last_obs = obs

            elements = dict(actions=actions, rewards=rewards, dones=dones, infos=infos)
            elements.update(additional or {})
            rollout.append(elements)

            self._total_steps += len(self._env)
            if any(dones):
                self._reset_terminated_envs(dones)
                if self._stop_on_done:
                    break

        rollout.set_element('last_obs', self._last_obs)
        return rollout

    def _reset_terminated_envs(self, dones):
        index_dones = np.where(dones)[0]
        for idx_done in index_dones:
            self._last_obs[idx_done] = self._env.reset_env(idx_done)
        broadcast(Message.ENV_TERMINATED, index_dones=index_dones)
