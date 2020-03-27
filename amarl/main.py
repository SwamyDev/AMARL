import abc
from collections import deque, defaultdict

import click
import logging
import matplotlib.pyplot as plt
from social_dilemmas.envs.cleanup import CleanupEnv

from social_dilemmas.envs.harvest import HarvestEnv

from amarl.maps import MINI_HARVEST_MAP, FIRING_CLEANUP_MAP

logger = logging.getLogger(__name__)


class EnvironmentFactory(abc.ABC):
    def __init__(self, num_agents):
        self._num_agents = num_agents

    @abc.abstractmethod
    def __call__(self):
        return CleanupEnv()


class HarvestEnvFactory(EnvironmentFactory):
    def __call__(self):
        return HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=self._num_agents)


class CleanupEnvFactory(EnvironmentFactory):
    def __call__(self):
        return CleanupEnv(ascii_map=FIRING_CLEANUP_MAP, num_agents=self._num_agents)


_ENVIRONMENT_FACTORIES = {
    'harvest': HarvestEnvFactory,
    'cleanup': CleanupEnvFactory,
}

_ENVIRONMENT_DEFAULT = tuple(_ENVIRONMENT_FACTORIES.keys())[0]


@click.group()
@click.option('-e', '--environment', default=_ENVIRONMENT_DEFAULT, type=click.STRING,
              help=f"choose an environment from {tuple(_ENVIRONMENT_FACTORIES.keys())}. default: {_ENVIRONMENT_DEFAULT}")
@click.option('-n', '--num-agents', default=2, type=click.INT, help="number of agents. default: 2")
@click.option('--log-level', default="INFO", type=click.STRING, help="set the logging level (default: INFO)")
@click.pass_context
def cli(ctx, environment, num_agents, log_level):   # pragma: no cover
    """
    CLI to explore environments and train agents on them
    """
    numeric_level = getattr(logging, log_level.upper())
    logging.basicConfig(level=numeric_level)
    logging.root.setLevel(numeric_level)

    ctx.obj = dict(
        env_selected=_ENVIRONMENT_FACTORIES[environment.lower()](num_agents)
    )


@cli.command()
@click.pass_context
def explore(ctx):   # pragma: no cover
    env = ctx.obj['env_selected']()
    logger.info(f"Observations space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Number of agents: {env.num_agents}")

    for agent_id in env.agents:
        logger.info(f"{agent_id}: action space: {env.agents[agent_id].action_space}")

    plt.ion()
    done = False
    env.reset()
    last_rewards = deque(maxlen=100)

    while not done:
        plt.pause(0.001)
        env.render()
        actions = {agent_id: env.agents[agent_id].action_space.sample() for agent_id in env.agents}
        _, rewards, dones, _ = env.step(actions)

        last_rewards.append(rewards)
        avg = calc_average(last_rewards)
        avg_pretty = {k: f"{avg[k]:.2f}" for k in avg}

        print(f"\rrunning average rewards: {avg_pretty}", end="")
        logger.debug(f"actions: {actions}")
        logger.debug(f"rewards: {rewards}")
        logger.debug(f"dones: {dones}")
        done = dones['__all__']
    print("\n")


def calc_average(last_rewards):
    avg = defaultdict(lambda: 0)
    for r in last_rewards:
        for i in r:
            avg[i] += r[i] / len(last_rewards)
    return avg


if __name__ == '__main__':
    cli()
