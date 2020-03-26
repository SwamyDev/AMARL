import torch

from amarl.models import Net


def test_feed_forward_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    model = Net().to(device)
    assert model(torch.randn(16, 8).to(device)).cpu().detach().numpy().shape == (16, 2)
