#!/usr/bin/env python3
"""
Batch resample WAV files (e.g., LibriTTS-R 24 kHz -> 22.05 kHz) using torchaudio.
- Preserves folder structure
- Multiprocessing for speed
- Safe to resume (skips existing files unless --overwrite)
- Validates sample rate and channels
- Forces mono if input is unexpectedly stereo
- Minimal memory footprint (stream one file at a time)

Usage:
    python 01_resample_audio.py \
        --in-root /path/to/LibriTTS-R \
        --out-root /path/to/normalized_audio \
        --orig-sr 24000 \
        --new-sr 22050 \
        --num-workers 8
"""

import argparse
import warnings
from typing import Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torchaudio
from tqdm import tqdm


# Suppress torchaudio deprecation spam
warnings.filterwarnings(
    "ignore",
    message=".*TorchAudio.*TorchCodec.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
)


def parse_args():
    ap = argparse.ArgumentParser(description="Batch resample WAVs with torchaudio (preserve directories).")
    ap.add_argument("--in-root", type=Path, required=True, help="Input root directory (e.g., LibriTTS-R)")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root directory (e.g., normalized_audio)")
    ap.add_argument("--orig-sr", type=int, default=24000, help="Expected original sample rate (default: 24000)")
    ap.add_argument("--new-sr", type=int, default=22050, help="Target sample rate (default: 22050)")
    ap.add_argument("--pattern", type=str, default="*.wav", help="Glob pattern to match audio files")
    ap.add_argument("--num-workers", type=int, default=6, help="Parallel workers (processes)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist")
    ap.add_argument("--dry-run", action="store_true", help="List files that would be processed and exit")
    return ap.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_out_path(in_path: Path, in_root: Path, out_root: Path) -> Path:
    return out_root / in_path.relative_to(in_root)


def resample_waveform(waveform: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out)
    return resampler(waveform)


def process_one(
    in_path: Path,
    out_path: Path,
    expected_sr: int,
    target_sr: int,
    overwrite: bool=False
) -> Tuple[Path, str]:
    """
    Returns: (out_path, status)
    status in {"skipped", "ok", "warn", "error:<message>"}
    """

    try:
        if out_path.exists() and not overwrite:
            return (out_path, "skipped")

        ensure_parent(out_path)
        waveform, sr = torchaudio.load(in_path)  # [C, T], float32

        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            return (out_path, "error:EmptyAudio:zero-length waveform")

        # Force mono if stereo is encountered
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Safety: contiguous for Resample
        waveform = waveform.to(torch.float32).contiguous()

        if sr != expected_sr:
            status_note = f"warn:detected_sr={sr}!=expected={expected_sr}"
        else:
            status_note = "ok"

        # Resample
        waveform_rs = resample_waveform(waveform, sr, target_sr)

        # Safety: clamp amplitude range [-1,1]
        waveform_rs = torch.clamp(waveform_rs, -1.0, 1.0)

        # Save
        torchaudio.save(
            out_path.as_posix(),
            waveform_rs,
            sample_rate=target_sr,
        )
        return (out_path, status_note)

    except Exception as e:
        return (out_path, f"error:{type(e).__name__}:{e}")


def main():
    args = parse_args()

    in_root = args.in_root.resolve()
    out_root = args.out_root.resolve()

    if not in_root.exists():
        raise SystemExit(f"Input root does not exist: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect files
    files = list(in_root.rglob(args.pattern))
    if len(files) == 0:
        raise SystemExit(f"No files matched pattern {args.pattern} under {in_root}")

    if args.dry_run:
        print(f"[DRY RUN] Found {len(files)} files under {in_root}")
        for p in files[:20]:
            print(" -", p)
        if len(files) > 20:
            print(f"... and {len(files)-20} more")
        return

    total = len(files)
    print(f"Resampling {total} files {args.orig_sr} Hz → {args.new_sr} Hz")
    print(f"Writing to: {out_root}")
    print(f"Workers: {args.num_workers} | Overwrite: {args.overwrite}")

    futures = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for in_path in files:
            out_path = build_out_path(in_path, in_root, out_root)
            futures.append(
                ex.submit(process_one, in_path, out_path, args.orig_sr, args.new_sr, args.overwrite)
            )

        skipped = 0
        errors = 0
        warnings = 0

        # tqdm MAIN-PROCESS SAFETY
        with tqdm(total=total, desc="Resampling", unit="file") as pbar:
            for fut in as_completed(futures):
                out_path, status = fut.result()

                if status.startswith("error"):
                    errors += 1
                    pbar.write(f"ERROR -> {out_path} :: {status}")
                elif status.startswith("warn"):
                    warnings += 1
                    pbar.write(f"WARN  -> {out_path} :: {status}")
                elif status == "skipped":
                    skipped += 1

                pbar.update(1)

    print("\nDONE.")
    print(f"Processed: {total - skipped}")
    print(f"Skipped: {skipped}")
    print(f"Warnings: {warnings}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()
