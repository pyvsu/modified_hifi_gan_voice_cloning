from typing import List, Tuple

import torch
from torchaudio.transforms import MelSpectrogram


def feature_matching_loss(
    real_features: List[List[torch.Tensor]],
    fake_features: List[List[torch.Tensor]],
    weight: float = 2.0
) -> torch.Tensor:
    """
    Encourages the generator to match discriminator internal activations
    to improve perceptual quality.
    """

    loss = torch.zeros(1, device=real_features[0][0].device)

    # First nesting corresponds to each sub-discriminator
    for subdisc_real_feats, subdisc_fake_feats in zip(real_features, fake_features):

        # Second nesting corresponds to each layer for the current sub-discriminator
        for real_layer, fake_layer in zip(subdisc_real_feats, subdisc_fake_feats):

            # Compute L1 (absolute) distance
            loss += torch.mean(torch.abs(real_layer - fake_layer))

    return loss * weight


# GAN Loss
def discriminator_adversarial_loss(
    real_logits: List[torch.Tensor],
    fake_logits: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """
    LS-GAN discriminator loss.

    real -> should be 1
    fake -> should be 0
    """

    total_loss = torch.zeros(1, device=real_logits[0].device)
    real_losses = []
    fake_losses = []

    for real_logit, fake_logit in zip(real_logits, fake_logits):
        real_loss = torch.mean((1.0 - real_logit) ** 2)
        fake_loss = torch.mean(fake_logit ** 2)

        total_loss += (real_loss + fake_loss)
        real_losses.append(real_loss.item())
        fake_losses.append(fake_loss.item())

    return total_loss, real_losses, fake_losses


def generator_adversarial_loss(fake_logits: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float]]:
    """
    LS-GAN generator loss.

    Push fake outputs to be classified as 1.
    """

    total_loss = torch.zeros(1, device=fake_logits[0].device)
    losses = []

    for fake_logit in fake_logits:
        loss = torch.mean((1.0 - fake_logit) ** 2)
        total_loss += loss
        losses.append(loss.item())

    return total_loss, losses


def mel_spectrogram_loss(
    real_audio: torch.Tensor,
    fake_audio: torch.Tensor,
    mel_transform: MelSpectrogram,
    weight: float = 45.0
) -> torch.Tensor:
    """
    Computes mel-spectrogram L1 loss between real and fake audio.
    """

    real_mel_spectrogram = mel_transform(real_audio)
    fake_mel_spectrogram = mel_transform(fake_audio)

    loss = torch.mean(torch.abs(real_mel_spectrogram - fake_mel_spectrogram))

    return loss * weight
