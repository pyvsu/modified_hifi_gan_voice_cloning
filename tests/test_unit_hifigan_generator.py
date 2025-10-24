"""
Comprehensive test suite for UnitHiFiGANGenerator.

Covers:
1. Model construction & structure
2. Forward pass shape correctness
3. Numerical stability (no NaN/Inf)
4. Weight norm removal consistency
"""

import pytest
import torch
from models.hifigan_generator import UnitHiFiGANGenerator


@pytest.fixture(scope="module")
def config():
    """Return a shared configuration for all tests."""
    return {
        "num_embeddings": 1000,
        "embedding_dim": 128,
        "model_in_dim": 128,
        "upsample_initial_channel": 512,
        "upsample_rates": [5, 4, 4, 2, 2],
        "upsample_kernel_sizes": [11, 8, 8, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
    }


@pytest.fixture(scope="module")
def model(config):
    """Return a generator model in eval mode."""
    G = UnitHiFiGANGenerator(config)
    G.eval()
    return G


def log_tensor_stats(tensor, name):
    """Helper function for debug logging."""
    mean = tensor.mean().item()
    std = tensor.std().item()
    print(f"\n[{name}] mean={mean:.4f}, std={std:.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")


# ------------------------------------------------------------------------------
# 1. Structure Tests
# ------------------------------------------------------------------------------

def test_structure_counts(model, config):
    """Ensure model builds correct number of layers and MRF groups."""
    print("\n[TEST] Verifying generator structure...")

    assert len(model.ups) == len(config["upsample_rates"]), "Mismatch in number of upsampling layers"
    assert len(model.resblocks) == len(config["upsample_rates"]), "Mismatch in number of ResBlock groups"

    for i, stage_blocks in enumerate(model.resblocks):
        print(f"  Stage {i}: {len(stage_blocks)} ResBlocks")
        assert len(stage_blocks) == len(config["resblock_kernel_sizes"])


# ------------------------------------------------------------------------------
# 2. Forward Pass Test
# ------------------------------------------------------------------------------

def test_forward_pass_shape(model, config):
    """Forward pass should produce correct waveform shape."""
    print("\n[TEST] Running forward pass shape check...")

    B, T = 2, 100
    units = torch.randint(0, config["num_embeddings"], (B, T))
    y = model(units)
    expected_length = T
    for r in config["upsample_rates"]:
        expected_length *= r

    log_tensor_stats(y, "Output waveform")

    assert y.shape == (B, 1, expected_length), f"Expected (B,1,{expected_length}), got {tuple(y.shape)}"


# ------------------------------------------------------------------------------
# 3. Numerical Stability
# ------------------------------------------------------------------------------

def test_no_nan_or_inf(model, config):
    """Output should contain finite values only."""
    print("\n[TEST] Checking for NaN/Inf...")

    units = torch.randint(0, config["num_embeddings"], (1, 50))
    with torch.no_grad():
        y = model(units)

    log_tensor_stats(y, "Stability check")

    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    assert abs(y.mean().item()) < 0.5, "Output mean too large, may indicate unstable behavior"


# ------------------------------------------------------------------------------
# 4. Weight Norm Removal
# ------------------------------------------------------------------------------

def test_remove_weight_norm_consistency(model, config):
    """Removing weight norms should not drastically alter output."""
    print("\n[TEST] Checking weight norm removal consistency...")

    units = torch.randint(0, config["num_embeddings"], (1, 10))
    with torch.no_grad():
        before = model(units)

    model.remove_weight_norm()

    with torch.no_grad():
        after = model(units)

    diff = (before - after).abs().mean().item()
    print(f"  Mean absolute diff after removing weight norm: {diff:.6f}")

    assert diff < 1e-3, f"Output changed significantly after removing weight norm ({diff:.6f})"


# ------------------------------------------------------------------------------
# 5. Gradient Flow (Optional)
# ------------------------------------------------------------------------------

def test_backward_pass(model, config):
    """Ensure gradients can propagate without NaNs."""
    print("\n[TEST] Checking backward pass...")

    model.train()
    units = torch.randint(0, config["num_embeddings"], (1, 10))
    y = model(units)
    loss = y.abs().mean()
    loss.backward()

    total_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad += grad_norm
    print(f"  Total gradient norm: {total_grad:.4f}")

    assert total_grad > 0, "No gradients propagated!"
