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


def make_callable():
    class _CallableSpy:
        def __init__(self):
            self.received_arg = None
            self.num_called = 0

        def __call__(self, *args, **kwargs):
            self.received_arg = args[0]
            self.num_called += 1

    return _CallableSpy()


def test_execute_callable_at_step_frequency(trainable):
    every_10_steps = make_callable()
    every_50_steps = make_callable()

    trainable.steps_per_train = 8
    amarl.run(trainable, num_steps=100, step_frequency_fns={
        10: every_10_steps,
        50: every_50_steps
    })

    assert every_10_steps.received_arg == trainable
    assert every_10_steps.num_called == 10
    assert every_50_steps.num_called == 2
