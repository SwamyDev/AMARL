import time
import logging
from pathlib import Path

import numpy as np

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
    def __init__(self, env, policy):
        self._env = env
        self._policy = policy
        self._last_obs = self._env.reset()

        self._final_rewards = list()
        self._total_steps = 0
        self._measurement = PerformanceMeasure()
        self._measurement.start()

    def rollout(self, horizon):
        rollout = RolloutBatch()
        for _ in range(horizon):
            actions, additional = self._policy.compute_actions(self._last_obs)
            obs, rewards, dones, infos = self._env.step(actions.cpu().numpy())
            self._print_progress(infos)
            self._last_obs = obs

            elements = dict(actions=actions, rewards=rewards, dones=dones, infos=infos)
            elements.update(additional or {})
            rollout.append(elements)

            self._reset_terminated_envs(dones)

        self._total_steps += horizon * len(self._env)
        rollout.set_element('last_obs', self._last_obs)

        self._print_performance()
        return rollout

    def _reset_terminated_envs(self, dones):
        if any(dones):
            for done_idx in np.where(dones)[0]:
                self._env.envs[done_idx].reset()

    def _print_performance(self):
        if self._total_steps % 10000 == 0:
            self._measurement.stop()
            logger.info(f"steps: {self._total_steps}, performance: {10000 / self._measurement.elapsed}steps/s")
            self._measurement.start()

    def _print_progress(self, infos):
        for i in infos:
            r = i.get('episodic_return', None)
            if r is not None:
                self._final_rewards.append(r)
                if len(self._final_rewards) % 100 == 0:
                    logger.info(f"steps: {self._total_steps}, last reward: {sum(self._final_rewards[-100:]) / 100}")
