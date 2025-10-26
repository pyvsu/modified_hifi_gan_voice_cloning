#!/usr/bin/env python3
"""
Build flat JSONL metadata for unit-based emotional TTS.

- Single CLI arg: --root (dataset parent directory)
- Auto-detect partitions from subfolders under normalized_audio/
- Derive sibling modality paths by mirroring subpaths/filenames
- Missing-file safeguards (skip and warn)
- tqdm progress bar
- Relative paths (relative to --root)
- Flat JSONL per partition: train.jsonl, valid.jsonl, test.jsonl
- No "split" field inside entries
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

# Optional deps (best-effort)
try:
    import soundfile as sf  # Preferred for duration
except Exception:
    sf = None

try:
    import torchaudio  # Fallback for duration
except Exception:
    torchaudio = None

try:
    import torch  # For loading .pt unit ids / embeddings
except Exception as e:
    raise SystemExit("PyTorch is required to load .pt files. Please install torch.") from e


def detect_partition_from_path(path: Path) -> Optional[str]:
    """
    Map any partitioned subpath to one of: 'train', 'valid', 'test'.
    Heuristic: look for keywords in the path parts.
    """
    parts = [p.lower() for p in path.parts]
    # Check in order of specificity
    if any("train" in p for p in parts):
        return "train"
    if any("dev" in p or "valid" in p for p in parts):
        return "valid"
    if any("test" in p for p in parts):
        return "test"
    return None


def get_audio_info(wav_path: Path) -> Tuple[float, int]:
    """
    Return (duration_sec, sample_rate) using the most available backend.
    Preference: soundfile -> torchaudio -> wave (stdlib).
    """
    wav_path_str = str(wav_path)

    # 1) soundfile
    if sf is not None:
        try:
            f = sf.SoundFile(wav_path_str)
            duration = len(f) / float(f.samplerate)
            return duration, int(f.samplerate)
        except Exception:
            pass

    # 2) torchaudio
    if torchaudio is not None:
        try:
            info = torchaudio.info(wav_path_str)
            # torchaudio.info can return either AudioMetaData or a tuple in older versions
            sr = int(getattr(info, "sample_rate", getattr(info, "sample_rate", 0)))
            num_frames = int(getattr(info, "num_frames", getattr(info, "num_frames", 0)))
            if sr > 0 and num_frames >= 0:
                return (num_frames / float(sr), sr)
        except Exception:
            pass

    # 3) wave (stdlib) fallback
    try:
        import wave
        with wave.open(wav_path_str, "rb") as w:
            sr = w.getframerate()
            frames = w.getnframes()
            duration = frames / float(sr) if sr else 0.0
            return duration, int(sr)
    except Exception:
        # Last resort
        return 0.0, 0


def len_units(units_obj) -> int:
    """
    Try to compute number of tokens/frames in the units object loaded from .pt
    Handles common shapes: 1D (T,), 2D (T, D), list/np array, dict with 'codes' key, etc.
    """
    # If it's a dict with a common key
    if isinstance(units_obj, dict):
        for key in ("units", "codes", "ids", "tokens"):
            if key in units_obj:
                return len_units(units_obj[key])

    # PyTorch tensor
    if torch.is_tensor(units_obj):
        if units_obj.ndim == 0:
            return int(units_obj.item())
        return int(units_obj.shape[0])

    # Numpy array
    try:
        import numpy as np
        if isinstance(units_obj, np.ndarray):
            return int(units_obj.shape[0])
    except Exception:
        pass

    # Python list/sequence
    if hasattr(units_obj, "__len__"):
        return int(len(units_obj))

    # Fallback
    return 0


def derive_modal_paths(root: Path, rel_wav_path: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Given a relative wav path like:
      normalized_audio/train-clean-100/84/121550/84_121550_000014_000002.wav
    return absolute full paths for:
      wav, units, speaker_emb, emotion_emb
    by mirroring subpath under modality roots.
    """
    parts = rel_wav_path.parts
    # Replace top folder 'normalized_audio' with each modality root
    if parts[0] != "normalized_audio":
        raise ValueError(f"Expected relative path to start with 'normalized_audio', got: {rel_wav_path}")

    # Rebuild sibling paths
    units_rel = Path("unit_embeddings", *parts[1:-1], Path(parts[-1]).with_suffix(".pt"))
    spk_rel = Path("speaker_embeddings", *parts[1:-1], Path(parts[-1]).with_suffix(".pt"))
    emo_rel = Path("emotion_embeddings", *parts[1:-1], Path(parts[-1]).with_suffix(".pt"))

    wav_abs = root / rel_wav_path
    units_abs = root / units_rel
    spk_abs = root / spk_rel
    emo_abs = root / emo_rel

    return wav_abs, units_abs, spk_abs, emo_abs


def build_metadata_entry(root: Path, rel_wav_path: Path) -> Optional[dict]:
    """
    Build a single flat metadata entry for one utterance.
    Returns dict or None if files missing or invalid.
    """
    wav_abs, units_abs, spk_abs, emo_abs = derive_modal_paths(root, rel_wav_path)

    # Enforce existence
    for p in (wav_abs, units_abs, spk_abs, emo_abs):
        if not p.exists():
            logging.warning(f"Missing modality for {rel_wav_path}: {p.name} not found at {p}")
            return None

    # Parse IDs
    utt = rel_wav_path.stem
    speaker_id = utt.split("_", 1)[0] if "_" in utt else rel_wav_path.parent.name

    # Audio info
    duration_sec, sample_rate = get_audio_info(wav_abs)

    # Units length
    try:
        units_obj = torch.load(units_abs, map_location="cpu")
        num_units = len_units(units_obj)
    except Exception as e:
        logging.warning(f"Failed to load units for {rel_wav_path}: {e}")
        return None

    # Relative paths (portable)
    wav_rel_out = os.path.relpath(wav_abs, root)
    units_rel_out = os.path.relpath(units_abs, root)
    spk_rel_out = os.path.relpath(spk_abs, root)
    emo_rel_out = os.path.relpath(emo_abs, root)

    entry = {
        "utterance_id": utt,
        "speaker_id": speaker_id,
        "wav": wav_rel_out.replace("\\", "/"),
        "units": units_rel_out.replace("\\", "/"),
        "speaker_emb": spk_rel_out.replace("\\", "/"),
        "emotion_emb": emo_rel_out.replace("\\", "/"),
        "duration_sec": round(float(duration_sec), 5),
        "num_units": int(num_units),
        "sample_rate": int(sample_rate),
    }
    return entry


def main():
    parser = argparse.ArgumentParser(description="Generate flat JSONL metadata for unit-based emotional TTS.")
    parser.add_argument("--root", type=str, required=True, help="Dataset parent directory.")
    parser.add_argument("--wav-ext", type=str, default=".wav", help="Audio file extension to scan under normalized_audio (default: .wav)")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Skip utterances shorter than this (seconds). Default: 0 (no filter)")
    parser.add_argument("--min-units", type=int, default=0, help="Skip utterances with fewer than this many units. Default: 0 (no filter)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    root = Path(args.root).resolve()
    norm_root = root / "normalized_audio"
    if not norm_root.exists():
        raise SystemExit(f"'normalized_audio' not found under {root}")

    # Prepare output files at root
    out_map = {
        "train": root / "train.jsonl",
        "valid": root / "valid.jsonl",
        "test": root / "test.jsonl",
    }
    # Open all three (overwrite by default, empty if no entries)
    writers = {k: open(v, "w", encoding="utf-8") for k, v in out_map.items()}

    # Collect all wav files (one pass) for a single tqdm
    wav_files = list(norm_root.rglob(f"*{args.wav_ext}"))
    total = len(wav_files)
    written = {"train": 0, "valid": 0, "test": 0}
    skipped = 0

    with tqdm(total=total, desc="Building metadata", unit="utt") as pbar:
        for wav_abs in wav_files:
            # Compute path relative to root and detect partition from it
            rel_wav = wav_abs.relative_to(root)
            split = detect_partition_from_path(rel_wav)
            if split is None:
                logging.warning(f"Could not detect partition for: {rel_wav}. Skipping.")
                skipped += 1
                pbar.update(1)
                continue

            entry = build_metadata_entry(root, rel_wav)
            if entry is None:
                skipped += 1
                pbar.update(1)
                continue

            # Filters
            if entry["duration_sec"] < args.min_duration:
                skipped += 1
                pbar.update(1)
                continue
            if entry["num_units"] < args.min_units:
                skipped += 1
                pbar.update(1)
                continue

            # Write to the correct file (no "split" field in entry)
            writers[split].write(json.dumps(entry, ensure_ascii=False) + "\n")
            written[split] += 1
            pbar.update(1)

    # Close writers
    for f in writers.values():
        f.close()

    logging.info(f"Done. Wrote: train={written['train']}, valid={written['valid']}, test={written['test']}. Skipped={skipped}.")
    for k, path in out_map.items():
        logging.info(f"{k}.jsonl -> {path} ({os.path.getsize(path)} bytes)")

if __name__ == "__main__":
    main()
