import numpy as np


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

    def rollout(self, horizon):
        rollout = RolloutBatch()
        for _ in range(horizon):
            actions, additional = self._policy.compute_actions(self._last_obs)
            obs, rewards, dones, infos = self._env.step(actions)
            self._last_obs = obs

            elements = dict(actions=actions, rewards=rewards, dones=dones, infos=infos)
            elements.update(additional or {})
            rollout.append(elements)

            self._reset_terminated_envs(dones)

        rollout.set_element('last_obs', self._last_obs)
        return rollout

    def _reset_terminated_envs(self, dones):
        if any(dones):
            for done_idx in np.where(dones)[0]:
                self._env.envs[done_idx].reset()
