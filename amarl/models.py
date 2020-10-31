import numpy as np
import torch as th
import torch.nn as nn
from gym.spaces import Box
from ray.rllib.models import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from typing import List

from ray.rllib.utils.framework import TensorType

LSTM_SIZE = 128


class ActorCriticBaseline(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        input_space = get_preprocessor(obs_space)(obs_space).observation_space
        self._cnn_shape = input_space.shape
        self._conv = nn.Sequential(
            nn.Conv2d(self._cnn_shape[0], 6, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self._conv(th.as_tensor(input_space.sample()[None]).float()).shape[1]

        self._linear = nn.Sequential(
            nn.Linear(n_flatten, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self._lstm = nn.LSTM(32, LSTM_SIZE, batch_first=True)

        self._action_branch = nn.Linear(LSTM_SIZE, num_outputs)
        self._value_branch = nn.Linear(LSTM_SIZE, 1)

        self._features = None

        for i in range(2):
            self.inference_view_requirements[f"state_in_{i}"] = \
                ViewRequirement(f"state_out_{i}", shift=-1, space=Box(-1.0, 1.0, shape=(LSTM_SIZE,)))

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        return [self._action_branch.weight.new(LSTM_SIZE).zero_(),
                self._action_branch.weight.new(LSTM_SIZE).zero_()]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        x = th.reshape(inputs, (-1,) + self._cnn_shape)
        x = self._conv(x)
        x = self._linear(x)
        x_time_ranked = th.reshape(x, (inputs.shape[0], inputs.shape[1], x.shape[-1]))
        self._features, (h, c) = self._lstm(x_time_ranked, [th.unsqueeze(state[0], 0), th.unsqueeze(state[1], 0)])
        logits = self._action_branch(self._features)
        return logits, [th.squeeze(h, 0), th.squeeze(c, 0)]

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return th.reshape(self._value_branch(self._features), [-1])

