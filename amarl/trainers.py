from amarl.policies import CommunicationPolicy, A2CPolicy
from amarl.workers import RolloutWorker


class A2CTrainer:
    def __init__(self, env, config):
        self._rollout_length = config['rollout_length']
        self._policy = A2CPolicy()
        self._worker = RolloutWorker(env, self._policy)

    def train(self):
        batch = self._worker.rollout(self._rollout_length)
        self._policy.learn_on_batch(batch)


class CommunicationTrainer:
    def __init__(self, env_factory, config):
        self._policy = CommunicationPolicy()
        self._workers = RolloutWorker(env_factory, config['num_workers'])
        self._roll_out_length = config['roll_out_length']
        self._policy_selector = lambda agent_id: self._policy

    def train(self):
        batch = self._workers.rollout(self._policy, self._roll_out_length)
        self._policy.learn_on_batch(batch)
