from amarl.policies import CommunicationPolicy
from amarl.workers import RolloutWorker


class CommunicationTrainer:
    def __init__(self, env_factory, config):
        self._policy = CommunicationPolicy()
        self._workers = RolloutWorker(env_factory, config['num_workers'])
        self._roll_out_length = config['roll_out_length']
        self._policy_selector = lambda agent_id: self._policy

    def train(self):
        batch = self._workers.rollout(self._policy, self._roll_out_length)
        self._policy.learn_on_batch(batch)
