import torch

from amarl.models import CommunicationNet, A2CNet, A2CLinearNet, A2CSocialInfluenceNet, A2CLinearLSTMNet


def test_communication_net_infers_all_parameters_correctly():
    model = CommunicationNet(view_dims=(15, 15), num_actions=8, num_agents=5, vocabulary_size=7, com_embedding=16)
    observations = torch.randn(4, 3, 15, 15)
    messages = torch.randn(4, 4 * 7)

    action_dists, action_values, message_dists, message_values = model(observations, messages)

    assert action_dists.param_shape == (4, 8) and action_values.shape == (4, 1)
    assert message_dists.param_shape == (4, 7) and message_values.shape == (4, 1)


def test_a2c_net_infers_all_parameters_correctly():
    model = A2CNet(view_dims=(3, 128, 128), num_actions=2)
    observations = torch.randn(4, 3, 128, 128)

    action_dists, action_values = model(observations)

    assert action_dists.param_shape == (4, 2) and action_values.shape == (4, 1)


def test_a2c_linear_net_infers_all_parameters_correctly():
    model = A2CLinearNet(observation_size=8, num_actions=2)
    observations = torch.randn(4, 8)

    action_dists, action_values = model(observations)

    assert action_dists.param_shape == (4, 2) and action_values.shape == (4, 1)


def test_a2c_social_influence_infers_all_parameters_correctly():
    model = A2CSocialInfluenceNet(view_dims=(3, 84, 84), num_actions=2)
    observations = torch.randn(1, 3, 84, 84)
    hidden = model.get_initial_state()

    action_dists, action_values, hidden = model(observations, hidden)

    assert action_dists.param_shape == (1, 2) and action_values.shape == (1, 1)
    hx, cx = hidden
    assert hx.shape == (1, 512) and cx.shape == (1, 512)


def test_a2c_lstm_linear_infers_all_parameters_correctly():
    model = A2CLinearLSTMNet(observation_size=8, num_actions=2)
    observations = torch.randn(1, 8)
    hidden = model.get_initial_state()

    action_dists, action_values, hidden = model(observations, hidden)

    assert action_dists.param_shape == (1, 2) and action_values.shape == (1, 1)
    hx, cx = hidden
    assert hx.shape == (1, 64) and cx.shape == (1, 64)

