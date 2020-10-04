import numpy as np


def render_trajectory(env, agent, reward_infos=None):
    reward_infos = reward_infos or []
    totals = dict(total_reward=0)
    for i in reward_infos:
        totals[f"total_{i}"] = 0

    obs = env.reset()
    done = False
    screens = []
    while not done:
        actions, _ = agent.compute_actions(obs.unsqueeze(dim=0))
        obs, reward, done, infos = env.step(actions[0])
        screen = env.render(mode='rgb_array')
        screens.append(screen.transpose(2, 0, 1))
        totals['total_reward'] += reward
        for i in reward_infos:
            if i in infos and infos[i]:
                totals[f"total_{i}"] += infos[i]
    env.close()

    return np.array([screens]), totals
