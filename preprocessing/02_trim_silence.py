#!/usr/bin/env python3
"""
Trim leading/trailing silence using Silero VAD, preserving LibriTTS hierarchy.

Design:
- Input is 22.05 kHz mono WAVs (from 01_resample_audio.py).
- For VAD only, audio is resampled to 16 kHz (Silero's native rate).
- We detect all speech segments, then keep a single span:
    [ first_speech_start - pad, last_speech_end + pad ]
  (We preserve inner pauses for natural prosody.)
- If no speech is found, we WARN and copy the file as-is.

Usage:
  python 02_trim_silence.py \
    --in-root /path/to/normalized_audio/train-clean-100 \
    --out-root /path/to/trimmed_audio/train-clean-100 \
    --num-workers 6 \
    --pad-ms 50 \
    --threshold 0.5

Install:
  pip install torch torchaudio tqdm
  # Silero VAD weights are fetched via torch.hub (cached under ~/.cache/torch/hub)
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Optional

import torch
import torchaudio
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")

# ---------- Globals initialized per worker ----------
VAD_MODEL = None
GET_SPEECH_TIMESTAMPS = None

def _init_worker(vad_repo: str, vad_model_name: str, trust_repo: bool):
    """
    Initialize Silero VAD model in each worker process.
    """
    global VAD_MODEL, GET_SPEECH_TIMESTAMPS
    # Torch Hub will cache after first download; subsequent loads are fast.
    VAD_MODEL, utils = torch.hub.load(
        vad_repo, vad_model_name, trust_repo=trust_repo
    )
    # utils is a tuple: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
    GET_SPEECH_TIMESTAMPS = utils[0]


# ---------- Utility functions ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Trim leading/trailing silence with Silero VAD.")
    ap.add_argument("--in-root", type=Path, required=True, help="Input root (22.05k WAVs).")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root for trimmed WAVs.")
    ap.add_argument("--pattern", type=str, default="*.wav", help="Glob to match audio files.")
    ap.add_argument("--num-workers", type=int, default=6, help="Parallel workers.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    # VAD / trimming params
    ap.add_argument("--pad-ms", type=int, default=50, help="Padding around detected speech boundaries (ms).")
    ap.add_argument("--threshold", type=float, default=0.5, help="Silero speech prob threshold (0..1).")
    ap.add_argument("--min-speech-ms", type=int, default=200, help="Min speech segment length (ms) for VAD.")
    ap.add_argument("--min-duration-s", type=float, default=0.5, help="Warn if trimmed audio < this (seconds).")
    ap.add_argument("--max-trim-pct-warn", type=float, default=0.7,
                    help="Warn if > this fraction (0..1) of samples were trimmed.")
    # Advanced (rarely need to change)
    ap.add_argument("--vad-sr", type=int, default=16000, help="VAD working sample rate.")
    ap.add_argument("--vad-repo", type=str, default="snakers4/silero-vad", help="TorchHub repo for Silero VAD.")
    ap.add_argument("--vad-model", type=str, default="silero_vad", help="TorchHub model name.")
    ap.add_argument("--trust-repo", action="store_true", help="Pass trust_repo=True to torch.hub.load.")
    ap.add_argument("--dry-run", action="store_true", help="List files that would be processed and exit.")
    return ap.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_out_path(in_path: Path, in_root: Path, out_root: Path) -> Path:
    return out_root / in_path.relative_to(in_root)


def _merge_to_single_span(ts_list: List[dict]) -> Optional[Tuple[int, int]]:
    """
    Given Silero timestamps [{'start': s0, 'end': e0}, ...] in *samples at vad_sr*,
    return a single (start, end) covering all speech (leading/trailing trim only).
    """
    if not ts_list:
        return None
    s0 = min(t['start'] for t in ts_list)
    eN = max(t['end'] for t in ts_list)
    return (s0, eN)


def _time_convert(idx_samples: int, sr_from: int, sr_to: int) -> int:
    return int(round(idx_samples * (sr_to / sr_from)))


def _trim_with_vad(
    wav_22k: torch.Tensor,
    sr_22k: int,
    vad_sr: int,
    threshold: float,
    min_speech_ms: int,
    pad_ms: int,
) -> Tuple[torch.Tensor, str]:
    """
    Run VAD at vad_sr, map timestamps back to 22k, trim with padding.
    Returns (trimmed_wav, status_note).
    status_note in {"ok", "warn:nospeech_copied", "warn:very_short_after_trim", "ok:heavy_trim(...)"}
    """
    assert wav_22k.dim() == 2 and wav_22k.shape[0] == 1, "Expect mono [1, T]"
    T22 = wav_22k.shape[-1]

    # Resample to vad_sr for VAD only
    if sr_22k != vad_sr:
        rs_to_16k = torchaudio.transforms.Resample(orig_freq=sr_22k, new_freq=vad_sr)
        wav_16k = rs_to_16k(wav_22k)
    else:
        wav_16k = wav_22k

    # Silero expects 1D tensor on CPU
    audio_16k = wav_16k.squeeze(0).cpu()

    # Run VAD
    ts_list = GET_SPEECH_TIMESTAMPS(
        audio_16k,
        VAD_MODEL,
        sampling_rate=vad_sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        return_seconds=False,
    )

    if len(ts_list) == 0:
        # No speech detected -> copy as-is, warn
        return wav_22k, "warn:nospeech_copied"

    # Merge to single [start, end] to preserve inner pauses
    span_16k = _merge_to_single_span(ts_list)
    assert span_16k is not None
    start_16k, end_16k = span_16k

    # Apply padding in 22k space
    pad_22k = int(round(pad_ms / 1000.0 * sr_22k))

    # Map timestamps from 16k to 22k
    start_22k = _time_convert(start_16k, sr_from=vad_sr, sr_to=sr_22k) - pad_22k
    end_22k   = _time_convert(end_16k,   sr_from=vad_sr, sr_to=sr_22k) + pad_22k

    # Clamp
    start_22k = max(0, start_22k)
    end_22k = min(T22, end_22k)
    if end_22k <= start_22k:
        # Fallback: if padding/rounding made window invalid, copy as-is
        return wav_22k, "warn:invalid_span_copied"

    trimmed = wav_22k[:, start_22k:end_22k].contiguous()
    return trimmed, "ok"


def process_one(
    in_path: Path,
    out_path: Path,
    vad_sr: int,
    threshold: float,
    min_speech_ms: int,
    pad_ms: int,
    min_duration_s: float,
    max_trim_pct_warn: float,
    overwrite: bool = False,
) -> Tuple[Path, str]:
    """
    Returns (out_path, status).
    status can be:
      - "skipped"
      - "ok"
      - "ok:heavy_trim(xxx)"
      - "warn:nospeech_copied"
      - "warn:very_short_after_trim"
      - "warn:invalid_span_copied"
      - "error:..."
    """
    try:
        if out_path.exists() and not overwrite:
            return (out_path, "skipped")

        # Load 22.05k wav
        wav, sr = torchaudio.load(in_path)  # [C, T]
        if wav.numel() == 0 or wav.shape[-1] == 0:
            return (out_path, "error:EmptyAudio:zero-length waveform")

        # Force mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Ensure float32 and contiguous
        wav = wav.to(torch.float32).contiguous()

        # If sample rate is not 22.05k (edge case), we still proceed and map against that sr
        sr_in = sr
        original_len = wav.shape[-1]

        trimmed, status = _trim_with_vad(
            wav_22k=wav,
            sr_22k=sr_in,
            vad_sr=vad_sr,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            pad_ms=pad_ms,
        )

        kept = trimmed.shape[-1]
        trim_pct = 1.0 - float(kept) / float(original_len)

        # Warnings based on duration and trim ratio
        duration_s = kept / float(sr_in)
        if duration_s < min_duration_s and status == "ok":
            status = "warn:very_short_after_trim"
        elif status == "ok" and trim_pct >= max_trim_pct_warn:
            status = f"ok:heavy_trim({trim_pct:.2f})"

        # Save
        ensure_parent(out_path)
        torchaudio.save(out_path.as_posix(), trimmed, sample_rate=sr_in)
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

    files = sorted(in_root.rglob(args.pattern))
    if len(files) == 0:
        raise SystemExit(f"No files matched pattern {args.pattern} under {in_root}")

    if args.dry_run:
        print(f"[DRY RUN] Found {len(files)} files under {in_root}")
        for p in files[:30]:
            print(" -", p)
        if len(files) > 30:
            print(f"... and {len(files)-30} more")
        return

    print(f"Trimming leading/trailing silence with Silero VAD")
    print(f"Input : {in_root}")
    print(f"Output: {out_root}")
    print(f"Workers: {args.num_workers} | Overwrite: {args.overwrite}")
    print(f"Params: vad_sr={args.vad_sr}, threshold={args.threshold}, pad_ms={args.pad_ms}, min_speech_ms={args.min_speech_ms}")

    # Build task list
    tasks = [(p, build_out_path(p, in_root, out_root)) for p in files]

    # Multiprocessing with per-process VAD init
    skipped = 0
    errors = 0
    warns = 0
    heavies = 0
    nospeech = 0

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker,
        initargs=(args.vad_repo, args.vad_model, args.trust_repo),
    ) as ex:
        futures = [
            ex.submit(
                process_one,
                in_path,
                out_path,
                args.vad_sr,
                args.threshold,
                args.min_speech_ms,
                args.pad_ms,
                args.min_duration_s,
                args.max_trim_pct_warn,
                args.overwrite,
            )
            for in_path, out_path in tasks
        ]

        with tqdm(total=len(futures), desc="Trimming", unit="file") as pbar:
            for fut in as_completed(futures):
                out_path, status = fut.result()

                if status.startswith("error"):
                    errors += 1
                    pbar.write(f"ERROR -> {out_path} :: {status}")
                elif status == "skipped":
                    skipped += 1
                elif status.startswith("warn:nospeech"):
                    nospeech += 1
                    pbar.write(f"WARN  -> {out_path} :: {status}")
                elif status.startswith("warn"):
                    warns += 1
                    pbar.write(f"WARN  -> {out_path} :: {status}")
                elif status.startswith("ok:heavy_trim"):
                    heavies += 1
                    pbar.write(f"NOTE  -> {out_path} :: {status}")

                pbar.update(1)

    print("\nDONE.")
    print(f"Total     : {len(files)}")
    print(f"Processed : {len(files) - skipped}")
    print(f"Skipped   : {skipped}")
    print(f"Errors    : {errors}")
    print(f"Warnings  : {warns} (incl. very_short_after_trim)")
    print(f"No-speech : {nospeech} (copied as-is)")
    print(f"Heavy trim: {heavies} (>{int(args.max_trim_pct_warn*100)}% removed)")

if __name__ == "__main__":
    main()
