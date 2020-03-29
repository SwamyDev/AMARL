from amarl._version import __version__
from amarl.main import cli

name = "amarl"


def run(trainable, num_episodes):
    for _ in range(num_episodes):
        trainable.train()
