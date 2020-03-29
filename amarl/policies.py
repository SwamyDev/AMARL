import abc


class Policy(abc.ABC):
    @abc.abstractmethod
    def compute_actions(self, observations):
        pass


class RandomPolicy(Policy):
    def __init__(self, action_space):
        self._action_space = action_space

    def compute_actions(self, observations):
        return [self._action_space.sample() for _ in range(len(observations))]


class CommunicationPolicy(Policy):
    pass
