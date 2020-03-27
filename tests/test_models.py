import torch

from amarl.models import CommunicationNet


def test_communication_net_infers_all_parameters_correctly():
    model = CommunicationNet(view_dims=(15, 15), num_actions=8, num_agents=5, vocabulary_size=7, com_embedding=16)
    observations = torch.randn(4, 3, 15, 15)
    messages = torch.randn(4, 4 * 7)

    action_dists, action_values, message_dists, message_values = model(observations, messages)

    assert action_dists.param_shape == (4, 8) and action_values.shape == (4, 1)
    assert message_dists.param_shape == (4, 7) and message_values.shape == (4, 1)
