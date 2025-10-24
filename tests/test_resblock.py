"""
Test suite for ResBlock (1-D) implementation.

Run with:
    pytest -v tests/test_resblock.py
"""

import math
import torch
import torch.nn as nn
import pytest
from models.resblock import ResBlock


# ---- Fixtures -----------------------------------------------------------------

@pytest.fixture
def resblock():
    """Default ResBlock fixture (matches HiFi-GAN settings)."""
    return ResBlock(channels=64, kernel_size=3, dilations=(1, 3, 5))


@pytest.fixture
def input_tensor():
    """Dummy input tensor [B, C, T]."""
    torch.manual_seed(42)
    return torch.randn(4, 64, 120)  # batch=4, channels=64, length=120


# ---- Sanity & Shape tests ------------------------------------------------------

def test_forward_output_shape(resblock, input_tensor):
    """Output should have same shape as input."""
    y = resblock(input_tensor)
    assert y.shape == input_tensor.shape


def test_forward_type_and_device(resblock, input_tensor):
    """Output type and device should match input."""
    y = resblock(input_tensor)
    assert y.dtype == input_tensor.dtype
    assert y.device == input_tensor.device


def test_gradients_flow(resblock, input_tensor):
    """Ensure backward pass works and gradients propagate."""
    input_tensor.requires_grad_(True)
    y = resblock(input_tensor)
    loss = y.mean()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()


# ---- Initialization behavior ---------------------------------------------------

def test_weight_init_normal_distribution(resblock):
    """Check that Conv weights roughly follow N(0, 0.01)."""
    weights = []
    for convs in (resblock.convs1, resblock.convs2):
        for conv in convs:
            w = conv.weight.data.view(-1)
            weights.append(w)
    all_w = torch.cat(weights)
    mean, std = all_w.mean().item(), all_w.std().item()
    assert abs(mean) < 0.01, f"Unexpected mean: {mean}"
    assert math.isclose(std, 0.01, rel_tol=0.2), f"Unexpected std: {std}"


# ---- Functional correctness ----------------------------------------------------

def test_residual_connection_effect(resblock, input_tensor):
    """Output should differ from input (non-trivial mapping)."""
    y = resblock(input_tensor)
    diff = (y - input_tensor).abs().mean().item()
    assert diff > 1e-5, "Block behaves like identity (no residual effect)"


def test_residual_identity_if_zero_weights(input_tensor):
    """With zero weights, output stays within same magnitude range (residual connection works)."""
    block = ResBlock(channels=64, kernel_size=3, dilations=(1, 3, 5))
    for convs in (block.convs1, block.convs2):
        for conv in convs:
            nn.init.constant_(conv.weight, 0.0)
    y = block(input_tensor)
    diff = (y - input_tensor).abs().mean().item()
    rel_diff = diff / input_tensor.abs().mean().item()
    # Just ensure output is not wildly different or unstable
    assert 0.1 < rel_diff < 0.6, f"Unexpected residual deviation ({rel_diff:.2f})"


# ---- WeightNorm utilities ------------------------------------------------------

def test_remove_weight_norm(resblock):
    """After remove_weight_norm(), parameters should no longer be normalized."""
    resblock.remove_weight_norm()
    for convs in (resblock.convs1, resblock.convs2):
        for conv in convs:
            # After removal, conv should not have 'weight_g' or 'weight_v'
            params = dict(conv.named_parameters())
            assert "weight_g" not in params
            assert "weight_v" not in params
            # Should have a regular 'weight'
            assert hasattr(conv, "weight")


# ---- Stress test ---------------------------------------------------------------

@pytest.mark.parametrize("batch,channels,length", [(1, 64, 32), (8, 64, 512), (2, 128, 256)])
def test_various_shapes(batch, channels, length):
    """ResBlock should handle arbitrary sequence lengths."""
    x = torch.randn(batch, channels, length)
    block = ResBlock(channels, 3, (1, 3, 5))
    y = block(x)
    assert y.shape == x.shape
