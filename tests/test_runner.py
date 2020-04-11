import amarl

import pytest


class TrainableStub:
    def __init__(self):
        self.steps_per_train = 1
        self._steps = 0

    @property
    def steps_trained(self):
        return self._steps

    def train(self):
        self._steps += self.steps_per_train


class TrainableSpy(TrainableStub):
    def __init__(self):
        super().__init__()
        self.num_train_calls = 0

    def train(self):
        self.num_train_calls += 1
        super().train()


@pytest.fixture
def trainable():
    return TrainableSpy()


def test_runner_trains_trainable_for_specified_steps(trainable):
    trainable.steps_per_train = 10
    amarl.run(trainable, num_steps=100)
    assert trainable.num_train_calls == 10
