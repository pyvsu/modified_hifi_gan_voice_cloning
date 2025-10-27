#!/usr/bin/env python3
import os
import json
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset.libritts_r import LibriTTSRDataset
from dataset.collate import collate_fn

from models.hifigan_generator import UnitHiFiGANGenerator
from models.multiperiod_discriminator import MultiPeriodDiscriminator
from models.multiscale_discriminator import MultiScaleDiscriminator

from training.ema import update_ema
from training.checkpoint import save_checkpoint, maybe_restore_checkpoint
from training.mel import MelSpectrogram
from training.losses import (
    feature_matching_loss,
    mel_spectrogram_loss,
    generator_adversarial_loss,
    discriminator_adversarial_loss
)


UNIT_HOP_SAMPLES = 320  # mHuBERT units at 50 Hz for 16 kHz audio

DEFAULT_HYPERPARAMS = dict(
    lr_generator=2e-4,
    lr_discriminator=2e-4,
    betas=(0.8, 0.99),
    weight_decay=0.0,
    weight_mel=45.0,
    weight_feature_matching=2.0,
    weight_gan=1.0,
    amp=True,
    ema_decay=0.999,
    checkpoint_interval=10000,
)

def seed_everything(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser("Unit-Conditioned HiFi-GAN with FiLM Training")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--segment-size", type=int, default=8960)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--outdir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lr-generator", type=float, default=DEFAULT_HYPERPARAMS["lr_generator"])
    parser.add_argument("--lr-discriminator", type=float, default=DEFAULT_HYPERPARAMS["lr_discriminator"])
    parser.add_argument("--amp", action="store_true", default=DEFAULT_HYPERPARAMS["amp"])
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_HYPERPARAMS["ema_decay"])
    parser.add_argument("--weight-mel", type=float, default=DEFAULT_HYPERPARAMS["weight_mel"])
    parser.add_argument("--weight-feature-matching", type=float, default=DEFAULT_HYPERPARAMS["weight_feature_matching"])
    parser.add_argument("--weight-gan", type=float, default=DEFAULT_HYPERPARAMS["weight_gan"])
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_HYPERPARAMS["checkpoint_interval"])

    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def crop_aligned_segments(waveform, units, segment_size, unit_hop=UNIT_HOP_SAMPLES):
    """
    Random crop that aligns waveform and unit frames.

    waveform: [B, 1, T]
    units:    [B, Tu]
    """
    batch_size, _, total_audio_length = waveform.shape
    segment_units = segment_size // unit_hop

    cropped_audio_list = []
    cropped_unit_list = []

    for idx in range(batch_size):
        unit_sequence = units[idx]
        unit_length = unit_sequence.shape[0]

        # Pad units if too short
        if unit_length < segment_units:
            pad_amount = segment_units - unit_length
            unit_sequence = F.pad(unit_sequence, (0, pad_amount), value=0)
            unit_length = unit_sequence.shape[0]

        max_unit_start = max(0, unit_length - segment_units)
        unit_start = random.randint(0, max_unit_start)
        unit_end = unit_start + segment_units
        unit_crop = unit_sequence[unit_start:unit_end]

        audio_start = unit_start * unit_hop
        audio_end = audio_start + segment_size

        if audio_end > total_audio_length:
            audio_start = max(0, total_audio_length - segment_size)
            audio_end = audio_start + segment_size

        audio_crop = waveform[idx:idx+1, :, audio_start:audio_end]

        cropped_audio_list.append(audio_crop)
        cropped_unit_list.append(unit_crop)

    waveform_out = torch.cat(cropped_audio_list, dim=0)
    units_out = torch.stack(cropped_unit_list, dim=0).long()
    return waveform_out, units_out


def run_discriminators_all(mpd, msd, audio):
    logits = []
    features = []

    # Multi-Period Discriminator
    mpd_scores, mpd_feats = mpd(audio)
    for score, feats in zip(mpd_scores, mpd_feats):
        logits.append(score)
        features.append(feats)

    # Multi-Scale Discriminator
    msd_scores, msd_feats = msd(audio)
    for score, feats in zip(msd_scores, msd_feats):
        logits.append(score)
        features.append(feats)

    return logits, features


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    # Datasets
    train_dataset = LibriTTSRDataset(
        root_dir=args.data_root,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Models
    generator = UnitHiFiGANGenerator(config, use_film=True).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # Exponential Moving Average (EMA)
    # Not directly needed during training, used primarily during inference
    # Smooths generator weight updates over time, improving inference stability and reducing noise artifacts
    ema = UnitHiFiGANGenerator(config, use_film=True).to(device)
    ema.load_state_dict(generator.state_dict())
    ema.eval()

    # Optimizers
    discriminator_optimizer = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=args.lr_discriminator,
        betas=(0.8, 0.99),
        weight_decay=0.0,
    )

    # Split FiLM (cond_proj) into slower LR group
    # To avoid FiLM updates being overly aggressive and destabilizing the generator
    # Previous approach updated both FiLM and the generator using the same LR caused instability and exploding gradients
    # Current approach will now induce FiLM to learn slower so that it avoids overpowering the generator
    film_params = []
    base_params = []

    for name, param in generator.named_parameters():
        if any(key in name for key in ["film_layers", "cond_proj"]):
            film_params.append(param)
        else:
            base_params.append(param)

    generator_optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr_generator},               # backbone
            {"params": film_params, "lr": args.lr_generator * 0.1},         # 10x slower
        ],
        betas=(0.8, 0.99),
        weight_decay=0.0,
    )

    grad_scaler = GradScaler(enabled=args.amp)

    # Mel Spectrogram
    mel_extractor = MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        fmin=0,
        fmax=8000,
        power=1.0,
        center=True
    ).to(device)

    # Resume Training
    global_step, start_epoch = 0, 0

    if args.resume:
        global_step, start_epoch = maybe_restore_checkpoint(
            args.resume,
            generator,
            mpd,
            msd,
            ema,
            generator_optimizer,
            discriminator_optimizer,
            grad_scaler,
            device,
        )

    # Train
    generator.train()
    mpd.train()
    msd.train()

    print("[*] Starting training...")

    for epoch in range(start_epoch, args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for batch in progress_bar:

            batch_waveform      = batch["wav"].unsqueeze(1).to(device)     # [B, 1, T]
            batch_units         = batch["units"].to(device)                # [B, U]
            batch_speaker_emb   = batch["speaker_emb"].to(device)          # [B, D_s]
            batch_emotion_emb   = batch["emotion_emb"].to(device)          # [B, D_e]

            # Aligned Crop
            cropped_waveform, cropped_units = crop_aligned_segments(
                batch_waveform,
                batch_units,
                segment_size=args.segment_size,
                unit_hop=UNIT_HOP_SAMPLES
            )


            # Update Discriminators
            with autocast(enabled=args.amp, device_type=args.device):
                fake_audio = generator(
                    cropped_units,
                    batch_speaker_emb,
                    batch_emotion_emb,
                    global_step=global_step
                )

                real_logits, real_features = run_discriminators_all(mpd, msd, cropped_waveform)
                fake_logits, fake_features = run_discriminators_all(mpd, msd, fake_audio.detach())

                discriminator_loss, _, _ = discriminator_adversarial_loss(real_logits, fake_logits)

                # NaN guard
                if torch.isnan(discriminator_loss) or torch.isinf(discriminator_loss):
                    print(f"[WARN] NaN/Inf detected in D at step {global_step}. Skipping D update.")
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    continue

            discriminator_optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(discriminator_loss).backward()
            grad_scaler.step(discriminator_optimizer)


            # Update Generator
            with autocast(enabled=args.amp, device_type=args.device):
                fake_audio = generator(
                    cropped_units,
                    batch_speaker_emb,
                    batch_emotion_emb,
                    global_step=global_step
                )

                fake_logits, fake_features = run_discriminators_all(mpd, msd, fake_audio)

                with torch.no_grad():
                    _, real_features = run_discriminators_all(mpd, msd, cropped_waveform)

                mel_loss = mel_spectrogram_loss(
                    real_audio=cropped_waveform.squeeze(1),
                    fake_audio=fake_audio.squeeze(1),
                    mel_transform=mel_extractor,
                )

                fm_loss = feature_matching_loss(real_features, fake_features)
                adversarial_loss_value, _ = generator_adversarial_loss(fake_logits)

                generator_loss = (
                    args.weight_mel * mel_loss +
                    args.weight_feature_matching * fm_loss +
                    args.weight_gan * adversarial_loss_value
                )

                # NaN guard: skip update if loss invalid
                if torch.isnan(generator_loss) or torch.isinf(generator_loss):
                    print(f"[WARN] NaN/Inf detected at step {global_step}. Skipping update.")
                    generator_optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

            # Backpropagation
            generator_optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(generator_loss).backward()

            # Added gradient clipping to prevent exploding gradients, which can lead to 'inf' weights
            # This consequently led to losses being NaN during training
            grad_scaler.unscale_(generator_optimizer)
            clip_grad_norm_(generator.parameters(), max_norm=5.0)

            # Update params
            grad_scaler.step(generator_optimizer)
            
            # NaN guard
            if grad_scaler.get_scale() > 2**16:
                print(f"[WARN] GradScaler scale too large. Resetting.")
                grad_scaler.update(new_scale=2.0)

            grad_scaler.update()

            # Update EMA only if losses are stable to avoid corrupting the checkpoint
            with torch.no_grad():
                if torch.isfinite(generator_loss):
                    update_ema(ema, generator, decay=args.ema_decay)
                else:
                    print(f"[WARN] Skipping EMA update at step {global_step} due to unstable loss.")

            # Increment global step
            global_step += 1

            progress_bar.set_postfix({
                "D": f"{discriminator_loss.item():.3f}",
                "G": f"{generator_loss.item():.3f}",
                "mel": f"{mel_loss.item():.3f}",
                "fm": f"{fm_loss.item():.3f}",
                "gan": f"{adversarial_loss_value.item():.3f}",
            })


            # Logging
            if global_step % 50 == 0:
                print(
                    f"[Epoch {epoch} | Step {global_step}] "
                    f"D: {discriminator_loss.item():.4f} | "
                    f"G: {generator_loss.item():.4f} "
                    f"(mel {mel_loss.item():.4f}, fm {fm_loss.item():.4f}, gan {adversarial_loss_value.item():.4f})"

                )

            # Periodic checkpoint
            if global_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    outdir=args.outdir,
                    tag=f"step_{global_step}",
                    generator=generator,
                    discriminator_period=mpd,
                    discriminator_scale=msd,
                    ema_generator=ema,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    scaler=grad_scaler,
                    step=global_step,
                    epoch=epoch,
                )

        # End of epoch checkpoint
        save_checkpoint(
            outdir=args.outdir,
            tag=f"epoch_{epoch}",
            generator=generator,
            discriminator_period=mpd,
            discriminator_scale=msd,
            ema_generator=ema,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            scaler=grad_scaler,
            step=global_step,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
