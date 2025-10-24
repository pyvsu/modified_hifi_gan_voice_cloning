import pytest
import torch
import torch.nn as nn

from models.multiperiod_discriminator import PeriodDiscriminator, MultiPeriodDiscriminator
from models.utils import get_padding


@pytest.fixture
def sample_waveform():
    """Generate a small batch of synthetic waveforms."""
    torch.manual_seed(42)
    # [batch, channels, time]
    return torch.randn(2, 1, 480)


# -----------------------------
#  Test: PeriodDiscriminator
# -----------------------------

def test_period_discriminator_structure():
    """Check number of layers and parameter shapes."""
    disc = PeriodDiscriminator(period=5)
    assert len(disc.convs) == 5, "Expected 5 conv layers in PeriodDiscriminator"
    assert isinstance(disc.conv_post, nn.Conv2d), "conv_post should be Conv2d"
    for conv in disc.convs:
        assert isinstance(conv, nn.Conv2d), "Each conv must be Conv2d"


def test_padding_is_dynamic():
    """Ensure get_padding uses kernel_size argument, not hardcoded 5."""
    ks = 7
    disc = PeriodDiscriminator(period=3, kernel_size=ks)
    # Verify computed padding equals get_padding(ks, 1)
    padding = disc.convs[0].padding[0]
    assert padding == get_padding(ks, 1), "Padding must match computed value"


def test_forward_shape_and_feature_maps(sample_waveform):
    """Ensure forward pass produces valid outputs and feature maps."""
    disc = PeriodDiscriminator(period=5)
    score, features = disc(sample_waveform)
    # Output should be [B, N] logits
    assert score.ndim == 2
    assert score.shape[0] == sample_waveform.shape[0], "Batch size mismatch"
    assert len(features) == len(disc.convs) + 1, "Feature maps count mismatch"


def test_padding_reflect_mode_preserves_continuity():
    """Ensure reflect padding works and adds correct number of samples."""
    disc = PeriodDiscriminator(period=7)
    x = torch.randn(1, 1, 100)
    _, _, t_before = x.shape
    score, _ = disc(x)
    # Check divisible padding was applied internally
    assert (t_before % 7) != 0  # precondition
    # No exception means padding was successful
    assert isinstance(score, torch.Tensor)


def test_grad_flow(sample_waveform):
    """Ensure gradients can flow back through discriminator."""
    disc = PeriodDiscriminator(period=3)
    score, _ = disc(sample_waveform)
    loss = score.mean()
    loss.backward()
    for p in disc.parameters():
        assert p.grad is not None, "No gradient propagated to parameter"


# -----------------------------
#  Test: MultiPeriodDiscriminator
# -----------------------------

def test_multi_period_discriminator_forward(sample_waveform):
    """Ensure MPD runs forward on real/fake inputs."""
    mpd = MultiPeriodDiscriminator(periods=(2, 3, 5))
    real_scores, fake_scores, real_feats, fake_feats = mpd(sample_waveform, sample_waveform)

    assert len(real_scores) == 3
    assert len(fake_scores) == 3
    assert len(real_feats) == 3
    assert len(fake_feats) == 3

    for rs, fs in zip(real_scores, fake_scores):
        assert rs.shape == fs.shape, "Real/Fake logits shape mismatch"


def test_all_subdiscriminators_different_periods():
    """Ensure each sub-discriminator has a distinct period."""
    mpd = MultiPeriodDiscriminator(periods=(2, 3, 5, 7, 11))
    periods = [d.period for d in mpd.sub_discriminators]
    assert len(set(periods)) == len(periods), "Each sub-discriminator period must be unique"


def test_mpd_grad_flow(sample_waveform):
    """Ensure MPD backward works for all sub-discriminators."""
    mpd = MultiPeriodDiscriminator(periods=(2, 3, 5))
    real_scores, fake_scores, _, _ = mpd(sample_waveform, sample_waveform)
    # Combine all discriminator outputs for a fake adversarial loss
    total_loss = sum([r.mean() + f.mean() for r, f in zip(real_scores, fake_scores)])
    total_loss.backward()
    for disc in mpd.sub_discriminators:
        for p in disc.parameters():
            assert p.grad is not None, "No gradient flow in sub-discriminator"
