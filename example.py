import logging

import gym
from torch.utils.tensorboard import SummaryWriter

import amarl
from amarl.messenger import monitor, LogMonitor, CombinedMonitor, TensorboardMonitor
from amarl.trainers import A2CTrainer
from amarl.visualisation import render_trajectory
from amarl.wrappers import MultipleEnvs, active_gym, OriginalReturnWrapper, SignReward, TorchObservation, StackFrames, \
    NoOpResetEnv, MaxAndSkipEnv, EpisodicLifeEnv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s:%(name)s: %(message)s')
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def make_env():
    e = gym.make('BreakoutNoFrameskip-v4')
    e = NoOpResetEnv(e)
    e = MaxAndSkipEnv(e)
    e = EpisodicLifeEnv(e)
    e = OriginalReturnWrapper(e)
    e = SignReward(e)
    e = TorchObservation(e)
    e = StackFrames(e, size=4)
    return e


writer = SummaryWriter()


def evaluate_agent(trainable):
    e = make_env()
    video, rewards = render_trajectory(e, trainable.policy, reward_infos=['episodic_return'])
    writer.add_video("trajectory", video, trainable.steps_trained, fps=40)
    writer.add_scalar("rewards/total", rewards['total_reward'], trainable.steps_trained)
    writer.add_scalar("returns/total", rewards['total_episodic_return'], trainable.steps_trained)


log_monitor = LogMonitor(logger, progress_averaging=100, performance_sample_size=10000)
tb_monitor = TensorboardMonitor(writer, 10, scalars=dict(episodic_return='returns/episodic'))
env = MultipleEnvs(make_env, num_envs=16)
with active_gym(env) as env, monitor(CombinedMonitor([log_monitor, tb_monitor])) as m:
    com = A2CTrainer(env, config={'rollout_horizon': 5, 'device': 'cpu'})
    try:
        amarl.run(com, num_steps=int(1e5), step_frequency_fns={int(2e4): evaluate_agent})
    finally:
        pass
