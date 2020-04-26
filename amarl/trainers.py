from amarl.policies import CommunicationPolicy, A2CPolicy, A2CLSTMPolicy
from amarl.workers import RolloutWorker


class PolicyOptimizationTrainer:
    def __init__(self, worker, policy, rollout_horizon):
        self._worker = worker
        self._policy = policy
        self._rollout_horizon = rollout_horizon

    @property
    def steps_trained(self):
        return self._worker.total_steps

    def train(self):
        batch = self._worker.rollout(self._rollout_horizon)
        self._policy.learn_on_batch(batch)


class A2CTrainer(PolicyOptimizationTrainer):
    def __init__(self, env, config):
        policy = A2CPolicy(env.observation_space, env.action_space, optimizer=config.get('optimizer'),
                           gradient_clip=config.get('gradient_clip', 5), device=config.get('device', 'cpu'))
        worker = RolloutWorker(env, policy)
        super().__init__(worker, policy, config['rollout_horizon'])


class A2CLSTMTrainer(PolicyOptimizationTrainer):
    def __init__(self, env, config):
        policy = A2CLSTMPolicy(env.observation_space, env.action_space, optimizer=config.get('optimizer'),
                               gradient_clip=config.get('gradient_clip', 5), device=config.get('device', 'cpu'))
        worker = RolloutWorker(env, policy)
        super().__init__(worker, policy, config['rollout_horizon'])


class CommunicationTrainer:
    def __init__(self, env_factory, config):
        self._policy = CommunicationPolicy()
        self._workers = RolloutWorker(env_factory, config['num_workers'])
        self._roll_out_length = config['roll_out_length']
        self._policy_selector = lambda agent_id: self._policy

    def train(self):
        batch = self._workers.rollout(self._policy, self._roll_out_length)
        self._policy.learn_on_batch(batch)
