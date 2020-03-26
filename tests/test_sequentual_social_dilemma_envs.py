from social_dilemmas.envs.harvest import HarvestEnv

# Maps for Harvest
MINI_HARVEST_MAP = [
    '@@@@@@',
    '@ P  @',
    '@  AA@',
    '@  AA@',
    '@  AP@',
    '@@@@@@',
]


def test_harvest_map():
    env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)
    env.reset()
    agents = list(env.agents.values())
    action_dim = agents[0].action_space.n
    for i in range(action_dim):
        env.step({'agent-0': i})
