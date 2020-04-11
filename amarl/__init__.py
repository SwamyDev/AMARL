from amarl._version import __version__
from amarl.main import cli

name = "amarl"


def run(trainable, num_steps):
    while trainable.steps_trained < num_steps:
        trainable.train()
