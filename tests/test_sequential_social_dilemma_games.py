import numpy as np

from gym.spaces import Box, Discrete
from social_dilemmas.envs.harvest import HarvestEnv

from amarl.maps import MINI_HARVEST_MAP


def test_harvest_map_actions():
    env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)
    env.reset()
    agents = list(env.agents.values())
    action_dim = agents[0].action_space.n
    for i in range(action_dim):
        env.step({'agent-0': i})


def test_harvest_map_spaces():
    env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)
    assert env.observation_space == Box(0.0, 0.0, (15, 15, 3), np.float32)
    assert env.action_space == Discrete(8)
