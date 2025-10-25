#!/usr/bin/env python3

import argparse
import os
from glob import glob
import torch
from tqdm import tqdm

from models.ecapa import ECAPA  # Your wrapper :contentReference[oaicite:0]{index=0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=str, required=True,
                        help="Root directory containing wav files.")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output root for embeddings.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if file exists.")
    args = parser.parse_args()

    wav_dir = args.wav_dir.rstrip("/")
    out_dir = args.out_dir.rstrip("/")

    # Recursively find ALL wav files
    wav_paths = sorted(glob(os.path.join(wav_dir, "**/*.wav"), recursive=True))
    if len(wav_paths) == 0:
        print(f"❌ No wav files found under {wav_dir}")
        return

    ecapa = ECAPA(device=args.device)

    for wav_path in tqdm(wav_paths, desc="Speaker embeddings"):

        # IMPORTANT: build mirrored relative path
        rel = os.path.relpath(wav_path, wav_dir)          # "19/227/utt.wav"
        rel_noext = os.path.splitext(rel)[0]              # "19/227/utt"
        out_path = os.path.join(out_dir, rel_noext + ".pt")

        # Create directories if missing
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Skip unless overwrite
        if os.path.exists(out_path) and not args.overwrite:
            continue

        emb = ecapa.extract_speaker_embeddings(wav_path)
        emb = emb.squeeze(0).squeeze(0).cpu()
        torch.save(emb, out_path)

    print(f"\n✅ Speaker embeddings saved under: {out_dir}")


if __name__ == "__main__":
    main()
