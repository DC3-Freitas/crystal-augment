import torch
import losses


def test_vicreg_variance_loss():
    # Constant embeddings -> loss is gamma
    z = torch.ones((1000, 10))
    gamma = 1.0
    loss = losses.vicreg_variance_loss(z, gamma=gamma)
    assert torch.isclose(
        loss, torch.tensor(gamma), atol=0.1
    ), f"Expected loss {gamma}, got {loss.item()}"
    # Embeddings with gaussian (std > gamma) -> loss is 0
    z = torch.randn((1000, 10))
    loss = losses.vicreg_variance_loss(z, gamma=gamma)
    assert loss < 0.05, f"Expected loss 0, got {loss.item()}"


def test_vicreg_covariance_loss():
    # Perfectly correlated -> loss is high
    z1 = torch.randn((1000, 5))
    z = torch.cat([z1, z1], dim=1)
    loss = losses.vicreg_covariance_loss(z)
    assert loss > 0.1, f"Expected high covariance loss, got {loss.item()}"
    # Uncorrelated -> loss is low
    z = torch.randn((1000, 10))
    loss = losses.vicreg_covariance_loss(z)
    assert loss < 0.01, f"Expected low covariance loss, got {loss.item()}"


def test_ranking_loss():
    # Correct order -> loss is 0
    s = torch.tensor([0.3, 0.2, 0.1])
    pairs = [(0, 1), (1, 2)]
    loss = losses.ranking_loss(s, pairs, margin=0.1)
    assert torch.isclose(
        loss, torch.tensor(0.0), atol=1e-4
    ), f"Expected loss 0, got {loss.item()}"
    # Incorrect order -> loss is positive
    s = torch.tensor([0.1, 0.2, 0.3])
    loss = losses.ranking_loss(s, pairs, margin=0.1)
    assert loss > 0.05, f"Expected positive loss, got {loss.item()}"
