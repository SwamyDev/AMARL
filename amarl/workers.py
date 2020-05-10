import logging
from collections import defaultdict

from amarl.messenger import broadcast, Message

logger = logging.getLogger(__name__)


class Rollout:
    def __init__(self):
        self._data = dict()
        self._length = 0

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data

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

    def __repr__(self):
        return f"Rollout({self._data}, length={self._length})"  # f"Rollout({self._data}, length={self._length})"


class IrregularRollout:
    def __init__(self):
        self._data = defaultdict(lambda: Rollout())

    @property
    def num_workers(self):
        return len(self._data)

    def of(self, rank_id):
        return self._data[rank_id]

    def append(self, new_elements):
        update = defaultdict(lambda: dict())
        for key, data_per_rank in new_elements.items():
            for rank_id, data in data_per_rank.items():
                update[rank_id][key] = data

        for rank_id, data_per_rank in update.items():
            self._data[rank_id].append(data_per_rank)

    def set_element(self, key, new_elements):
        for rank_id in new_elements:
            self._data[rank_id].set_element(key, new_elements[rank_id])

    def __repr__(self):
        rpr = "IrregularRollout({\n"
        for rank_id in self._data:
            rpr += f"\t{rank_id}: {repr(self._data[rank_id])}\n"
        rpr += "})"
        return rpr


class RolloutWorker:
    def __init__(self, env, policy, regularized=True):
        self._env = env
        self._policy = policy
        self._regularized = regularized
        self._last_obs = self._env.reset()
        self._total_steps = 0

    @property
    def total_steps(self):
        return self._total_steps

    def rollout(self, horizon):
        rollout = Rollout() if self._regularized else IrregularRollout()
        for _ in range(horizon):
            self._append_to_rollout_batch(rollout, *self._do_step())
            self._total_steps += len(self._env)
            self._handle_terminated_envs()

        rollout.set_element('last_obs', self._last_obs)
        if not self._regularized:
            self._reset_terminated_envs()
        return rollout

    def _do_step(self):
        actions, additional = self._policy.compute_actions(self._last_obs)
        obs, rewards, dones, infos = self._env.step(actions)
        broadcast(Message.TRAINING, infos=infos)
        self._last_obs = obs
        return actions, additional, dones, infos, rewards

    @staticmethod
    def _append_to_rollout_batch(rollout, actions, additional, dones, infos, rewards):
        elements = dict(actions=actions, rewards=rewards, dones=dones, infos=infos)
        elements.update(additional or {})
        rollout.append(elements)

    def _handle_terminated_envs(self):
        if self._env.num_active_envs < len(self._env):
            if self._regularized:
                self._reset_terminated_envs()
            else:
                self._pop_terminated_obs()

    def _reset_terminated_envs(self):
        index_dones = list(self._env.terminated_env_ids)
        for idx_done in index_dones:
            self._last_obs[idx_done] = self._env.reset_env(idx_done)
        broadcast(Message.ENV_TERMINATED, index_dones=index_dones)

    def _pop_terminated_obs(self):
        for i in self._env.terminated_env_ids:
            if i in self._last_obs:
                del self._last_obs[i]
