from amarl._version import __version__
from amarl.main import cli

name = "amarl"


def run(trainable, num_steps, step_frequency_fns=None):
    step_frequency_fns = step_frequency_fns or {}
    last_calls = {f: 0 for f in step_frequency_fns}

    while trainable.steps_trained < num_steps:
        trainable.train()
        for f in step_frequency_fns:
            if (trainable.steps_trained - last_calls[f]) >= f:
                step_frequency_fns[f](trainable)
                last_calls[f] += f
