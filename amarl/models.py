import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions import Categorical


def cv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride + 1


class CommunicationNet(nn.Module):
    def __init__(self, view_dims, num_actions, num_agents, vocabulary_size, com_embedding):
        super().__init__()
        self._num_actions = num_actions
        self._num_agents = num_agents
        self._vocabulary_size = vocabulary_size

        self.cv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)

        w = cv2d_size_out(view_dims[0], 3, 1)
        h = cv2d_size_out(view_dims[1], 3, 1)
        self.fc1 = nn.Linear(w * h * 6, 32)
        self.fc2 = nn.Linear(32, 32)

        self.lstm = nn.LSTM(32 + (num_agents - 1) * vocabulary_size, 128)
        self.hidden = self.init_hidden(128)

        self.action_pi = nn.Linear(128, num_actions)
        self.action_v = nn.Linear(128, 1)

        self.message_embedding = nn.Linear(128, com_embedding)
        self.message_pi = nn.Linear(com_embedding, vocabulary_size)
        self.message_v = nn.Linear(com_embedding, 1)

    @staticmethod
    def init_hidden(hidden_dim):
        return (autograd.Variable(torch.zeros(1, 1, hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, hidden_dim)))

    def forward(self, obs, msg):
        obs = F.relu(self.cv1(obs))
        obs = obs.view(obs.size(0), -1)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))

        lstm_in = torch.cat((obs, msg), dim=1)
        lstm_out, self.hidden = self.lstm(lstm_in.view(lstm_in.size(0), 1, -1), self.hidden)
        lstm_out = lstm_out.view(lstm_out.size(0), -1)

        act_logits = self.action_pi(lstm_out)
        act_dists = Categorical(logits=act_logits)
        act_v = self.action_v(lstm_out)

        msg_enc = self.message_embedding(lstm_out)
        msg_dists = Categorical(logits=self.message_pi(msg_enc))
        msg_v = self.message_v(msg_enc)

        return act_dists, act_v, msg_dists, msg_v


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class A2CNet(nn.Module):
    def __init__(self, view_dims, num_actions):
        super().__init__()
        self.cv1 = layer_init(nn.Conv2d(view_dims[0], 32, kernel_size=8, stride=4))
        self.cv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.cv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        h = cv2d_size_out(cv2d_size_out(cv2d_size_out(view_dims[1], 8, 4), 4, 2), 3, 1)
        w = cv2d_size_out(cv2d_size_out(cv2d_size_out(view_dims[2], 8, 4), 4, 2), 3, 1)
        self.fc1 = layer_init(nn.Linear(h * w * 64, 512))

        self.pi = layer_init(nn.Linear(512, num_actions), 1e-3)
        self.v = layer_init(nn.Linear(512, 1), 1e-3)

    def forward(self, obs):
        obs = F.relu(self.cv1(obs))
        obs = F.relu(self.cv2(obs))
        obs = F.relu(self.cv3(obs))
        obs = obs.view(obs.size(0), -1)
        obs = F.relu(self.fc1(obs))

        logits = self.pi(obs)
        dists = Categorical(logits=logits)
        vs = self.v(obs)
        return dists, vs


class A2CLinearNet(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.pi = nn.Linear(64, num_actions)
        self.v = nn.Linear(64, 1)

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        logits = self.pi(obs)
        dists = Categorical(logits=logits)
        vs = self.v(obs)
        return dists, vs
