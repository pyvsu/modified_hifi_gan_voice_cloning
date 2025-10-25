#!/usr/bin/env python3
"""
03_normalize_loudness.py
Normalize loudness of trimmed audio to a target LUFS level.

- Uses pyloudnorm (ITU-R BS.1770) perceptual loudness
- Preserves directory hierarchy
- Peak-safe with backoff margin
- Multiprocessing for speed
"""

import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import torch
import torchaudio
from tqdm import tqdm
import pyloudnorm as pyln

warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")


def parse_args():
    ap = argparse.ArgumentParser(description="LUFS normalize trimmed WAVs.")
    ap.add_argument("--in-root", type=Path, required=True, help="Input root (trimmed_audio).")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root (normalized_audio).")
    ap.add_argument("--pattern", type=str, default="*.wav", help="Glob pattern for matching wav files.")
    ap.add_argument("--num-workers", type=int, default=6, help="Multiprocessing workers.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    ap.add_argument("--dry-run", action="store_true", help="Preview files and exit.")
    ap.add_argument("--target-lufs", type=float, default=-23.0, help="Target integrated LUFS.")
    ap.add_argument("--peak-margin-db", type=float, default=0.1, help="Peak margin relative to 0 dBFS.")
    ap.add_argument("--max-gain-db", type=float, default=20.0, help="Max allowed gain (dB).")
    ap.add_argument("--min-gain-db", type=float, default=-20.0, help="Min allowed gain (dB).")
    return ap.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_out_path(in_path, in_root, out_root):
    return out_root / in_path.relative_to(in_root)


def db_to_lin(db):
    return 10.0 ** (db / 20.0)


def process_one(in_path: Path,
                out_path: Path,
                target_lufs: float,
                peak_margin_db: float,
                max_gain_db: float,
                min_gain_db: float,
                overwrite: bool = False) -> Tuple[Path, str]:

    try:
        if out_path.exists() and not overwrite:
            return (out_path, "skipped")

        ensure_parent(out_path)
        waveform, sr = torchaudio.load(in_path)  # [C, T]

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.numel() == 0:
            return (out_path, "error:EmptyAudio")

        audio = waveform.squeeze(0).cpu().numpy()

        meter = pyln.Meter(sr)  # ITU BS.1770-4
        try:
            measured_lufs = meter.integrated_loudness(audio)
        except Exception:
            return (out_path, "error:LUFSFail")

        # Compute gain needed
        gain_db = target_lufs - measured_lufs

        # Clip gain to user limits
        if gain_db > max_gain_db:
            gain_db = max_gain_db
            status = "warn:gain_capped_max"
        elif gain_db < min_gain_db:
            gain_db = min_gain_db
            status = "warn:gain_capped_min"
        else:
            status = "ok"

        # Apply gain
        audio_norm = audio * db_to_lin(gain_db)

        # Peak check
        peak = max(abs(audio_norm.max()), abs(audio_norm.min()))
        peak_limit = db_to_lin(-peak_margin_db)

        if peak > peak_limit:
            backoff_db = 20.0 * torch.log10(torch.tensor(peak_limit / peak)).item()
            gain_db += backoff_db
            audio_norm = audio * db_to_lin(gain_db)
            status = "warn:peak_limited"

        # Save
        audio_norm_t = torch.tensor(audio_norm).unsqueeze(0)
        audio_norm_t = torch.clamp(audio_norm_t, -1.0, 1.0)

        torchaudio.save(
            out_path.as_posix(),
            audio_norm_t,
            sample_rate=sr,
        )

        return (out_path, status)

    except Exception as e:
        return (out_path, f"error:{type(e).__name__}:{e}")


def main():
    args = parse_args()
    in_root = args.in_root.resolve()
    out_root = args.out_root.resolve()

    if not in_root.exists():
        raise SystemExit(f"Input root does not exist: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    files = list(in_root.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No WAV files found under {in_root}")

    if args.dry_run:
        print(f"[DRY RUN] Found {len(files)} files:")
        for p in files[:20]:
            print(" -", p)
        return

    print(f"Normalizing LUFS for {len(files)} files → {args.target_lufs} LUFS")
    print(f"Peak margin: {args.peak_margin_db} dB")
    print(f"Workers: {args.num_workers} | Overwrite: {args.overwrite}")

    futures = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for in_path in files:
            out_path = build_out_path(in_path, in_root, out_root)
            futures.append(ex.submit(
                process_one,
                in_path,
                out_path,
                args.target_lufs,
                args.peak_margin_db,
                args.max_gain_db,
                args.min_gain_db,
                args.overwrite
            ))

        skipped = 0
        errors = 0
        warnings_count = 0

        with tqdm(total=len(files), desc="LUFS Normalize", unit="file") as pbar:
            for fut in as_completed(futures):
                out_path, status = fut.result()

                if status.startswith("error"):
                    errors += 1
                    pbar.write(f"ERROR -> {out_path} :: {status}")
                elif status.startswith("warn"):
                    warnings_count += 1
                    pbar.write(f"WARN  -> {out_path} :: {status}")
                elif status == "skipped":
                    skipped += 1

                pbar.update(1)

    print("\nDONE.")
    print(f"Processed : {len(files) - skipped}")
    print(f"Skipped   : {skipped}")
    print(f"Warnings  : {warnings_count}")
    print(f"Errors    : {errors}")


if __name__ == "__main__":
    main()
