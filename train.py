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
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.libritts_r import LibriTTSRDataset
from dataset.length_bucket_sampler import LengthBucketSampler
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

    # Progressive unfreezing
    parser.add_argument(
        "--unfreeze-steps",
        type=str,
        default="5000,10000",
        help="Comma-separated step milestones to progressively unfreeze generator: e.g., '5000,10000'. "
            "Phase 1: FiLM only; after 1st milestone: +late blocks; after 2nd: +all."
    )
    parser.add_argument(
        "--phase1-include-last",
        action="store_true",
        help="If set, include the very last generator block (e.g., conv_post / last upsample) in Phase 1 with FiLM."
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--segment-size", type=int, default=8960)
    parser.add_argument("--max-steps", type=int, default=20000, help="Total number of optimization steps to run before stopping training.")

    # Avoid using this
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--val-batch-size", type=int, default=None, help="Defaults to train batch-size if None")
    parser.add_argument("--val-max-batches", type=int, default=12, help="How many mini-batches to evaluate per validation run")
    parser.add_argument("--val-intervals-per-epoch", type=int, default=4, help="Number of validation runs per epoch: 1 = only end, 2 = midpoint + end, 4 = 4 times evenly spaced, etc.")

    parser.add_argument("--outdir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained-weights", type=str, default="", help="Path to pretrained generator weights (.pt/.pth). Ignored if --resume is used.")
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


def _parse_unfreeze_steps_args(steps_str: str):
    try:
        steps = [int(s.strip()) for s in steps_str.split(",") if s.strip()]
        steps = sorted(list({s for s in steps if s > 0}))
        return steps
    except Exception:
        return []


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pretrained_model(generator, path, device):
    print(f"[*] Loading pretrained generator from: {path}")
    state = torch.load(path, map_location=device)

    # Handle checkpoints where weights are nested (e.g. {'generator': {...}})
    if "generator" in state:
        state = state["generator"]

    missing, unexpected = generator.load_state_dict(state, strict=False)

    # ---- Detailed reporting ----
    print(f"    Missing keys: {len(missing)}")
    if len(missing) > 0:
        print("      (parameters present in current model but NOT in checkpoint)")
        for k in missing:
            print(f"        - {k}")

    print(f"    Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print("      (parameters present in checkpoint but NOT in current model)")
        for k in unexpected:
            print(f"        - {k}")

    print("[*] Pretrained load complete.\n")


def _split_generator_param_names(generator):
    """
    Returns dict with three non-overlapping name sets for progressive unfreezing:
    - film_names: FiLM & cond projection layers (always trainable)
    - late_names: last-stage layers (upsample/resblocks/conv_post)
    - base_rest: everything else (mid/early)
    """
    film_keys = ("film_layers", "cond_proj")
    late_keys = ("conv_post", "ups", "upsample", "resblocks", "mrf", "mrd", "post", "tail")

    all_names = [n for (n, _) in generator.named_parameters()]

    film_names = {n for n in all_names if any(k in n for k in film_keys)}
    base_names = [n for n in all_names if n not in film_names]

    # Late candidates
    late_candidates = [n for n in base_names if any(k in n.lower() for k in late_keys)]

    if len(late_candidates) == 0:
        # Fallback: take last 25% of params if no match
        k = max(1, len(base_names) // 4)
        late_candidates = base_names[-k:]

    late_names = set(late_candidates)
    base_rest = {n for n in base_names if n not in late_names}

    return {
        "film_names": film_names,
        "late_names": late_names,
        "base_rest": base_rest,
    }


def _params_by_names(module, name_set):
    name_to_param = dict(module.named_parameters())
    return [name_to_param[n] for n in name_set if n in name_to_param]


def _set_requires_grad_by_names(module, name_set, value: bool):
    for n, p in module.named_parameters():
        if n in name_set:
            p.requires_grad = value


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


@torch.no_grad()
def validate_loop(
    gen_eval, mpd, msd, valid_loader, mel_extractor, device, 
    amp=True, segment_size=8960, unit_hop=UNIT_HOP_SAMPLES, max_batches=12
):
    """Lightweight validation: Mel L1 + Feature Matching over a limited number of mini-batches."""
    gen_eval.eval()
    mpd.eval()
    msd.eval()

    mel_losses = []
    fm_losses = []

    iteration = 0
    for batch in valid_loader:
        if iteration >= max_batches:
            break
        iteration += 1

        wav          = batch["wav"].unsqueeze(1).to(device)    # [B,1,T]
        units        = batch["units"].to(device)               # [B,U]
        speaker_emb  = batch["speaker_emb"].to(device)         # [B,Ds]
        emotion_emb  = batch["emotion_emb"].to(device)         # [B,De]

        # Use same crop policy as training for fair comparison
        wav_crop, units_crop = crop_aligned_segments(wav, units, segment_size=segment_size, unit_hop=unit_hop)

        with autocast(enabled=amp, device_type="cuda" if device.type == "cuda" else "cpu"):
            fake = gen_eval(units_crop, speaker_emb, emotion_emb)

            # Discriminator features in eval (no grads)
            fake_logits, fake_feats = run_discriminators_all(mpd, msd, fake)
            _, real_feats           = run_discriminators_all(mpd, msd, wav_crop)

            mel_loss = mel_spectrogram_loss(
                real_audio=wav_crop.squeeze(1),
                fake_audio=fake.squeeze(1),
                mel_transform=mel_extractor,
            )
            fm_loss  = feature_matching_loss(real_feats, fake_feats)

        mel_losses.append(mel_loss.item())
        fm_losses.append(fm_loss.item())

    results = {
        "mel": float(sum(mel_losses) / max(1, len(mel_losses))),
        "fm": float(sum(fm_losses)  / max(1, len(fm_losses))),
    }
    return results

def main():
    args = parse_arguments()

    if args.resume and args.pretrained_weights:
        print("[WARN] --resume overrides --pretrained-weights. Only the checkpoint will be loaded.")

    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    # Setup Datasets
    train_dataset = LibriTTSRDataset(root_dir=args.data_root, split="train")
    valid_dataset = LibriTTSRDataset(root_dir=args.data_root, split="valid")

    # Setup LengthBucketSamplers
    train_lengths = [
        length if length is not None else 0
        for length in getattr(train_dataset, "unit_lengths", [])
    ]

    valid_lengths = [
        length if length is not None else 0
        for length in getattr(valid_dataset, "unit_lengths", [])
    ]

    train_sampler = LengthBucketSampler(
        lengths=train_lengths,
        batch_size=args.batch_size,
        bucket_size=200,
        shuffle=True,
    )

    valid_sampler = LengthBucketSampler(
        lengths=valid_lengths,
        batch_size=args.val_batch_size or args.batch_size,
        bucket_size=200,
        shuffle=False,  # keep validation deterministic
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Determine validation frequency (convert intervals-per-epoch to step interval)
    steps_per_epoch = len(train_loader)
    # total_steps = args.epochs * steps_per_epoch

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * len(train_loader)
    
    print(f"[*] Total training steps: {total_steps}")

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

    # =============================
    # Two-Group Differential LRs
    # =============================
    # Group 1 (FAST): New layers that must learn from scratch
    #   - FiLM layers
    #   - cond_proj
    #   - dict (unit embedding table)
    #
    # Group 2 (SLOW): Pretrained HiFi-GAN backbone
    #   - conv_pre, ups, resblocks, conv_post

    groups = _split_generator_param_names(generator)
    film_names = groups["film_names"]

    # Explicitly include embedding ('dict') in the fast group
    all_param_names = [n for (n, _) in generator.named_parameters()]
    embed_names = {n for n in all_param_names if n.startswith("dict")}

    fast_names = set(film_names).union(embed_names)
    slow_names = {n for n in all_param_names if n not in fast_names}

    fast_params = _params_by_names(generator, fast_names)
    slow_params = _params_by_names(generator, slow_names)

    assert len(fast_params) > 0, "FAST group is empty (expected FiLM/cond_proj/dict)."
    assert len(slow_params) > 0, "SLOW group is empty (expected pretrained backbone)."

    # Safer fine-tuning multipliers:
    generator_param_groups = [
        {"params": fast_params, "lr": args.lr_generator},
        {"params": slow_params, "lr": args.lr_generator * 0.1},
    ]

    generator_optimizer = torch.optim.AdamW(
        generator_param_groups,
        betas=(0.8, 0.99),
        weight_decay=0.0,
    )

    discriminator_optimizer = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=args.lr_discriminator,
        betas=(0.8, 0.99),
        weight_decay=0.0,
    )

    generator_scheduler = CosineAnnealingLR(
        generator_optimizer, T_max=total_steps, eta_min=1e-6
    )
    discriminator_scheduler = CosineAnnealingLR(
        discriminator_optimizer, T_max=total_steps, eta_min=1e-6
    )

    print(f"[*] Using Two-Group DLR — FAST(new)=1.0x, SLOW(pre)=0.1x | CosineAnnealingLR (T_max={total_steps})")
    print(f"[*] FAST(new) params: {len(fast_names)} | SLOW(pre) params: {len(slow_names)}")
    for n in sorted(list(embed_names))[:3]:
        print(f"    [fast] {n}")


    # Progressive unfreezing setup
    # unfreeze_milestones = _parse_unfreeze_steps_args(args.unfreeze_steps)
    # groups = _split_generator_param_names(generator)

    # film_names = groups["film_names"]
    # late_names = groups["late_names"]
    # base_rest  = groups["base_rest"]

    # # Phase 1: only FiLM trainable (optionally include last block)
    # phase1_names = set(film_names)
    # if args.phase1_include_last:
    #     phase1_names = phase1_names.union(late_names)

    # # Freeze everything first
    # _set_requires_grad_by_names(generator, set(film_names).union(late_names).union(base_rest), False)
    # # Then unfreeze Phase-1 names
    # _set_requires_grad_by_names(generator, phase1_names, True)

    # # Keep state for milestones
    # current_phase = 1  # 1: FiLM-only(±last), 2: +late, 3: all

    # # Optimizers
    # # Phase-1 optimizer: FiLM (+ optional last block) only
    # film_params = _params_by_names(generator, film_names)
    # phase1_extra = _params_by_names(generator, (late_names if args.phase1_include_last else set()))
    # phase1_param_groups = []

    # if len(phase1_extra) > 0:
    #     # Last block at the same LR as the base generator
    #     phase1_param_groups.append({"params": phase1_extra, "lr": args.lr_generator})
    
    # # FiLM faster
    # phase1_param_groups.append({"params": film_params, "lr": args.lr_generator * 2.0})

    # generator_optimizer = torch.optim.AdamW(
    #     phase1_param_groups,
    #     betas=(0.8, 0.99),
    #     weight_decay=0.0,
    # )

    # discriminator_optimizer = torch.optim.AdamW(
    #     list(mpd.parameters()) + list(msd.parameters()),
    #     lr=args.lr_discriminator,
    #     betas=(0.8, 0.99),
    #     weight_decay=0.0,
    # )

    # # LR Schedulers
    # generator_scheduler = CosineAnnealingLR(
    #     generator_optimizer,
    #     T_max=total_steps,
    #     eta_min=1e-6
    # )

    # discriminator_scheduler = CosineAnnealingLR(
    #     discriminator_optimizer,
    #     T_max=total_steps,
    #     eta_min=1e-6
    # )

    # print(f"[*] Using CosineAnnealingLR (T_max={total_steps}, eta_min=1e-6)")

    # def advance_unfreeze_phase(phase:int):
    #     """
    #     Phase 1 -> Phase 2: add late block params
    #     Phase 2 -> Phase 3: add remaining base params
    #     """
    #     nonlocal generator_optimizer

    #     if phase == 1:
    #         # Move to Phase 2: unfreeze late block (if not already included)
    #         newly_trainable = late_names if not args.phase1_include_last else set()
    #         if len(newly_trainable) > 0:
    #             _set_requires_grad_by_names(generator, newly_trainable, True)
    #             new_params = _params_by_names(generator, newly_trainable)
    #             if len(new_params) > 0:
    #                 # Late block at base LR
    #                 generator_optimizer.add_param_group({"params": new_params, "lr": args.lr_generator})
    #         print("[*] Unfreeze Phase 2: training FiLM + late block(s).")
    #         return 2

    #     elif phase == 2:
    #         # Move to Phase 3: unfreeze the rest
    #         _set_requires_grad_by_names(generator, base_rest, True)
    #         new_params = _params_by_names(generator, base_rest)
    #         if len(new_params) > 0:
    #             generator_optimizer.add_param_group({"params": new_params, "lr": args.lr_generator})
    #         print("[*] Unfreeze Phase 3: training FiLM + ALL generator layers.")
    #         return 3

    #     return phase  # no change

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
    global_step = 0
    start_epoch = 0

    if args.resume:
        global_step, start_epoch = maybe_restore_checkpoint(
            args.resume,
            generator,
            mpd,
            msd,
            ema,
            generator_optimizer,
            discriminator_optimizer,
            generator_scheduler,
            discriminator_scheduler,
            grad_scaler,
            device,
        )
    elif args.pretrained_weights:
        # Only load pretrained if we are NOT resuming
        load_pretrained_model(generator, args.pretrained_weights, device)

        # EMA should start from the same weights
        ema.load_state_dict(generator.state_dict())


    if args.val_intervals_per_epoch < 1:
        args.val_intervals_per_epoch = 1  # safety clamp
    
    val_interval_steps = max(1, total_steps // max(1, args.val_intervals_per_epoch))
    print(f"[*] Validation will run {args.val_intervals_per_epoch}x total (~every {val_interval_steps} steps).")
    
    # val_interval_steps = total_steps // (args.val_intervals_per_epoch * args.epochs)
    # print(f"[*] Validation will run {args.val_intervals_per_epoch}x per epoch (~every {val_interval_steps} steps).")

    # Train
    generator.train()
    mpd.train()
    msd.train()

    # Will be used to determine the best model to be saved
    best_mel = float("inf")

    print("[*] Starting training...")

    # for epoch in range(start_epoch, args.epochs):
    epoch = start_epoch
    while global_step < total_steps:
        progress_bar = tqdm(train_loader, desc=f"Step {global_step}/{total_steps} | Epoch {epoch}", leave=True)
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for batch in progress_bar:
            if global_step >= total_steps:
                break

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
            with autocast(enabled=args.amp, device_type="cuda" if device.type == "cuda" else "cpu"):
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
            with autocast(enabled=args.amp, device_type="cuda" if device.type == "cuda" else "cpu"):
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

            # ---- LR Scheduler step ----
            generator_scheduler.step()
            discriminator_scheduler.step()

            # Update EMA only if losses are stable to avoid corrupting the checkpoint
            with torch.no_grad():
                if torch.isfinite(generator_loss):
                    update_ema(ema, generator, decay=args.ema_decay)
                else:
                    print(f"[WARN] Skipping EMA update at step {global_step} due to unstable loss.")

            # Increment global step
            global_step += 1

            # ---- Progressive unfreezing milestones ----
            # if len(unfreeze_milestones) > 0:
            #     # If we've crossed the next milestone, advance the phase
            #     while len(unfreeze_milestones) > 0 and global_step >= unfreeze_milestones[0]:
            #         next_step = unfreeze_milestones.pop(0)
            #         new_phase = advance_unfreeze_phase(current_phase)
            #         if new_phase != current_phase:
            #             current_phase = new_phase
            #             print(f"[*] Advanced to Phase {current_phase} at step {global_step} (milestone {next_step}).")


            progress_bar.set_postfix({
                "D": f"{discriminator_loss.item():.3f}",
                "G": f"{generator_loss.item():.3f}",
                "mel": f"{mel_loss.item():.3f}",
                "fm": f"{fm_loss.item():.3f}",
                "gan": f"{adversarial_loss_value.item():.3f}",
            })

            # Mid-epoch validation
            if global_step % val_interval_steps == 0:
                gen_eval = ema if ema is not None else generator

                val_out = validate_loop(
                    gen_eval, mpd, msd, valid_loader, mel_extractor, device,
                    amp=args.amp, segment_size=args.segment_size,
                    unit_hop=UNIT_HOP_SAMPLES, max_batches=args.val_max_batches
                )

                percent = (global_step % steps_per_epoch) / steps_per_epoch * 100
                print(f"[VAL | Step {global_step} | {percent:.0f}% of epoch] mel={val_out['mel']:.4f} fm={val_out['fm']:.4f}")

                if val_out["mel"] < best_mel:
                    best_mel = val_out["mel"]
                    save_checkpoint(
                        outdir=args.outdir,
                        tag="best_model",
                        generator=generator,
                        discriminator_period=mpd,
                        discriminator_scale=msd,
                        ema_generator=ema,
                        generator_optimizer=generator_optimizer,
                        discriminator_optimizer=discriminator_optimizer,
                        generator_scheduler=generator_scheduler,
                        discriminator_scheduler=discriminator_scheduler,
                        scaler=grad_scaler,
                        step=global_step,
                        epoch=epoch,
                    )
                    print(f"[✔] New best model saved (mel={best_mel:.4f})")

                    # Save best model metrics for traceability
                    with open(os.path.join(args.outdir, "best_model_metrics.json"), "w") as f:
                        json.dump({
                            "mel_loss": best_mel,
                            "epoch": epoch,
                            "step": global_step
                        }, f, indent=2)


            # Logging
            if global_step % 100 == 0:
                print(
                    f"[Epoch {epoch} | Step {global_step}] "
                    f"D: {discriminator_loss.item():.4f} | "
                    f"G: {generator_loss.item():.4f} "
                    f"(mel {mel_loss.item():.4f}, fm {fm_loss.item():.4f}, gan {adversarial_loss_value.item():.4f})"
                )

                current_lr = generator_scheduler.get_last_lr()[0]
                print(f"[LR] Step {global_step}: Generator LR={current_lr:.6f}")
            
        # End-of-epoch validation (EMA for eval if available)
        gen_eval = ema if ema is not None else generator

        val_out = validate_loop(
            gen_eval, mpd, msd, valid_loader, mel_extractor, device,
            amp=args.amp, segment_size=args.segment_size,unit_hop=UNIT_HOP_SAMPLES, max_batches=args.val_max_batches
        )

        print(f"[VAL | Epoch {epoch}] mel={val_out['mel']:.4f} fm={val_out['fm']:.4f}")

        # Track best and save a "best_mel" checkpoint
        if val_out["mel"] < best_mel:
            best_mel = val_out["mel"]
            save_checkpoint(
                outdir=args.outdir,
                tag="best_model",
                generator=generator,
                discriminator_period=mpd,
                discriminator_scale=msd,
                ema_generator=ema,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
                scaler=grad_scaler,
                step=global_step,
                epoch=epoch,
            )
            print(f"[✔] New best model saved (mel={best_mel:.4f})")

            # Save best model metrics for traceability
            with open(os.path.join(args.outdir, "best_model_metrics.json"), "w") as f:
                json.dump({
                    "mel_loss": best_mel,
                    "epoch": epoch,
                    "step": global_step
                }, f, indent=2)

        epoch += 1

        # Return models to train mode for next epoch
        generator.train()
        mpd.train()
        msd.train()


if __name__ == "__main__":
    main()
