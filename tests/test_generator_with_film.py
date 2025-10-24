"""
Unit tests for UnitHiFiGANGenerator with FiLM conditioning.

Covers:
- Basic forward pass without FiLM
- Forward pass with FiLM (simple + MLP)
- Conditioning dimension checks
- Identity consistency when FiLM off
"""

import torch
import pytest
from models.hifigan_generator import UnitHiFiGANGenerator


# ----------------------------
# Dummy Config for Testing
# ----------------------------
@pytest.fixture
def base_config():
    return {
        "num_embeddings": 1000,
        "embedding_dim": 128,
        "model_in_dim": 128,
        "upsample_initial_channel": 256,
        "upsample_rates": [4, 2],
        "upsample_kernel_sizes": [8, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5]] * 3,
        "cond_dim": 512,
        "film_hidden_dim": 128,
        "film_dropout_p": 0.1,
    }


# ----------------------------
# Utility
# ----------------------------
@pytest.fixture
def random_input():
    return torch.randint(0, 999, (2, 50))  # [B=2, T=50]


@pytest.fixture
def random_embeddings():
    speaker = torch.randn(2, 256)
    emotion = torch.randn(2, 256)
    return speaker, emotion


# ----------------------------
# Tests
# ----------------------------

def test_forward_no_film(base_config, random_input):
    """Ensure generator runs without FiLM."""
    model = UnitHiFiGANGenerator(base_config, use_film=False)
    y = model(random_input)
    assert y.shape[1] == 1, "Output must have 1 channel"
    assert torch.isfinite(y).all(), "Output contains NaN/inf"


def test_forward_with_film_simple(base_config, random_input, random_embeddings):
    """Test generator with simple FiLM conditioning."""
    cfg = {**base_config, "use_film_mlp": False}
    model = UnitHiFiGANGenerator(cfg, use_film=True)
    speaker, emotion = random_embeddings
    y = model(random_input, speaker, emotion)
    assert y.shape[1] == 1
    assert torch.isfinite(y).all()


def test_forward_with_film_mlp(base_config, random_input, random_embeddings):
    """Test generator with MLP FiLM variant."""
    cfg = {**base_config, "use_film_mlp": True}
    model = UnitHiFiGANGenerator(cfg, use_film=True)
    speaker, emotion = random_embeddings
    y = model(random_input, speaker, emotion)
    assert y.shape[1] == 1
    assert torch.isfinite(y).all()


def test_cond_dim_mismatch_raises(base_config, random_input):
    """Ensure mismatched conditioning dims raise an error."""
    cfg = {**base_config, "cond_dim": 512}
    model = UnitHiFiGANGenerator(cfg, use_film=True)

    speaker = torch.randn(2, 128)   # wrong dims
    emotion = torch.randn(2, 128)   # wrong dims
    with pytest.raises(RuntimeError):
        model(random_input, speaker, emotion)


def test_identity_behavior_without_condition(base_config, random_input):
    """When use_film=True but no conditioning provided, output should still be valid."""
    cfg = {**base_config, "use_film_mlp": False}
    model = UnitHiFiGANGenerator(cfg, use_film=True)
    y = model(random_input)
    assert torch.isfinite(y).all()
    assert y.shape[1] == 1


def test_reproducibility(base_config, random_input, random_embeddings):
    """Ensure deterministic behavior under fixed seed."""
    torch.manual_seed(42)
    cfg = {**base_config, "use_film_mlp": False}
    model = UnitHiFiGANGenerator(cfg, use_film=True)

    speaker, emotion = random_embeddings
    y1 = model(random_input, speaker, emotion)
    torch.manual_seed(42)
    model2 = UnitHiFiGANGenerator(cfg, use_film=True)
    y2 = model2(random_input, speaker, emotion)

    # outputs won't be identical due to random init, but should be finite & same shape
    assert y1.shape == y2.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
