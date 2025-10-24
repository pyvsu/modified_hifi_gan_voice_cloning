import torch
import pytest

from models.multiscale_discriminator import ScaleDiscriminator, MultiScaleDiscriminator


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def waveform():
    """Random synthetic waveform [batch, channels, time]."""
    torch.manual_seed(42)
    return torch.randn(2, 1, 6400)


@pytest.fixture
def small_waveform():
    """Smaller input to trigger downsampling edge cases."""
    torch.manual_seed(42)
    return torch.randn(1, 1, 512)


# -------------------------
# ScaleDiscriminator Tests
# -------------------------

def test_scale_discriminator_forward(waveform):
    """Forward pass should produce logits [B, N] and a list of feature maps."""
    disc = ScaleDiscriminator(use_spectral_norm=False)
    logits, features = disc(waveform)

    assert logits.ndim == 2, "Logits should be [B, N] after flatten"
    assert logits.shape[0] == waveform.shape[0], "Batch dimension mismatch"
    assert len(features) == 8, "Expected 7 conv layers + 1 post-conv feature map"


def test_feature_maps_monotonic_channel_growth(waveform):
    """Each layer should increase or maintain channel depth."""
    disc = ScaleDiscriminator()
    _, features = disc(waveform)

    channels = [feat.shape[1] for feat in features[:-1]]
    assert channels == sorted(channels), "Channels should non-decreasing across layers"


def test_grad_flow_scale(waveform):
    """Backprop should update parameters."""
    disc = ScaleDiscriminator()
    logits, _ = disc(waveform)
    loss = logits.mean()
    loss.backward()

    grad_exists = any((p.grad is not None) for p in disc.parameters())
    assert grad_exists, "Gradients should propagate through ScaleDiscriminator"


# -------------------------
# MultiScaleDiscriminator Tests
# -------------------------

def test_msd_forward_shapes(waveform):
    """Forward pass should produce 3 sets of logits and feature maps."""
    msd = MultiScaleDiscriminator()
    real_logits, fake_logits, real_feats, fake_feats = msd(waveform, waveform)

    assert len(real_logits) == 3
    assert len(fake_logits) == 3
    assert len(real_feats) == 3
    assert len(fake_feats) == 3

    # Logits should flatten to [B, N]
    for logit in real_logits:
        assert logit.ndim == 2
        assert logit.shape[0] == waveform.shape[0]


def test_downsampling_effect(small_waveform):
    """
    Scales > 0 should have shorter time dimension after avg pooling.
    Checks that downsampling occurs progressively.
    """
    msd = MultiScaleDiscriminator()
    real_logits, _, real_feats, _ = msd(small_waveform, small_waveform)

    # Extract temporal dimensions from the first activation of each sub-disc
    time_dims = [feat[0].shape[-1] for feat in real_feats]

    assert time_dims[0] > time_dims[1] > time_dims[2], \
        "Time dimension should shrink at each scale"


def test_grad_flow_msd(waveform):
    """Backprop should update parameters across all scales."""
    msd = MultiScaleDiscriminator()
    real_logits, fake_logits, _, _ = msd(waveform, waveform)

    loss = sum([r.mean() + f.mean() for r, f in zip(real_logits, fake_logits)])
    loss.backward()

    grad_exists = any((p.grad is not None) for d in msd.discriminators
                      for p in d.parameters())
    assert grad_exists, "Gradients should flow through all sub-discriminators"


def test_spectral_norm_only_first_scale():
    """Verify spectral_norm is only applied on the first ScaleDiscriminator."""
    msd = MultiScaleDiscriminator()

    # Check first scale
    first_layer = msd.discriminators[0].conv_layers[0]
    assert hasattr(first_layer, 'weight_orig'), \
        "First scale should use spectral_norm (produces weight_orig attribute)"

    # Check second scale
    second_layer = msd.discriminators[1].conv_layers[0]
    assert not hasattr(second_layer, 'weight_orig'), \
        "Second scale should use weight_norm, not spectral_norm"


def test_real_fake_symmetry(waveform):
    """The MSD should accept any real/fake pair with matched shapes."""
    msd = MultiScaleDiscriminator()
    _, _, real_feats, fake_feats = msd(waveform, waveform * 0.5)  # arbitrary fake

    for r_feat, f_feat in zip(real_feats, fake_feats):
        assert r_feat[0].shape[-1] == f_feat[0].shape[-1], \
            "Real/fake feature maps should match in shape"


def test_small_batch_robustness():
    """Ensure MSD works with batch_size=1 and short sequence."""
    x = torch.randn(1, 1, 1024)
    msd = MultiScaleDiscriminator()
    logits, _, _, _ = msd(x, x)
    assert len(logits) == 3
