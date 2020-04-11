import pytest

from amarl.trainers import PolicyOptimizationTrainer


class RolloutDummy:
    def __init__(self, size):
        self.size = size

    def __eq__(self, other):
        return self.size == other.size

    def __repr__(self):
        return f"RolloutDummy(size={self.size})"


class WorkerStub:
    def rollout(self, horizon):
        return RolloutDummy(horizon)


class WorkerSpy(WorkerStub):
    def __init__(self):
        self.returned_rollout = None
        self._total_steps = 0

    def set_total_steps(self, steps):
        self._total_steps = steps

    def rollout(self, horizon):
        self.returned_rollout = super().rollout(horizon)
        return self.returned_rollout

    @property
    def total_steps(self):
        return self._total_steps


class PolicyStub:
    pass


class PolicySpy(PolicyStub):
    def __init__(self):
        self.received_rollout = None

    def learn_on_batch(self, batch):
        self.received_rollout = batch


@pytest.fixture
def worker():
    return WorkerSpy()


@pytest.fixture
def policy():
    return PolicySpy()


@pytest.fixture
def po_trainer(worker, policy):
    return PolicyOptimizationTrainer(worker, policy, rollout_horizon=5)


def test_policy_optimisation_trainer_passes_rollout_of_worker_to_policy_to_train_on(po_trainer, worker, policy):
    po_trainer.train()
    assert worker.returned_rollout is not None and worker.returned_rollout == policy.received_rollout


def test_trainer_returns_steps_performed_by_worker(po_trainer, worker):
    worker.set_total_steps(42)
    assert po_trainer.steps_trained == 42
