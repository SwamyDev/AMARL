import numpy as np


class BatchRecorder:
    def __init__(self, batch_size, observation_shape, num_envs):
        self._batch = dict(actions=np.empty((batch_size, num_envs), dtype=np.int),
                           obs=np.empty((batch_size, num_envs, *observation_shape), dtype=np.float),
                           rewards=np.empty((batch_size, num_envs), dtype=np.float),
                           new_obs=np.empty((batch_size, num_envs, *observation_shape), dtype=np.float),
                           dones=np.empty((batch_size, num_envs), dtype=np.bool),
                           infos=np.empty((batch_size, num_envs), dtype=np.object))
        self._idx = 0

    def get_dict(self):
        return self._batch

    def record_transition(self, action, obs, reward, new_obs, done, info):
        self._batch['actions'][self._idx] = action
        self._batch['obs'][self._idx] = obs
        self._batch['rewards'][self._idx] = reward
        self._batch['new_obs'][self._idx] = new_obs
        self._batch['dones'][self._idx] = done
        self._batch['infos'][self._idx] = info
        self._idx += 1


class RolloutWorker:
    def __init__(self, env, policy):
        self._env = env
        self._policy = policy
        self._last_obs = self._env.reset()

    def rollout(self, horizon):
        batch = BatchRecorder(horizon, self._env.observation_space.shape, len(self._env))

        for _ in range(horizon):
            actions = self._policy.compute_actions(self._last_obs)
            obs, reward, dones, index = self._env.step(actions)
            batch.record_transition(actions, self._last_obs, reward, obs, dones, index)
            self._last_obs = obs
            if any(dones):
                for done_idx in np.where(dones)[0]:
                    self._env.envs[done_idx].reset()

        return batch.get_dict()
