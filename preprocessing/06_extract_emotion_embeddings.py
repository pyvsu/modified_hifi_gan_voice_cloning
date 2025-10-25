#!/usr/bin/env python3
"""
Batch Emotion2Vec embedding extractor.
Preserves original folder hierarchy.
"""

import argparse
import os
from glob import glob
import torch
from tqdm import tqdm

from models.emotion2vec import Emotion2Vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav-dir",
        type=str,
        required=True,
        help="Root directory containing wav files."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output root directory for emotion embeddings."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="iic/emotion2vec_plus_base",
        help="Emotion2Vec model variant."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force regenerate embeddings."
    )

    args = parser.parse_args()

    wav_dir = args.wav_dir.rstrip("/")
    out_dir = args.out_dir.rstrip("/")

    # recursively find all wavs
    wav_paths = sorted(glob(os.path.join(wav_dir, "**/*.wav"), recursive=True))
    if len(wav_paths) == 0:
        print(f"❌ No wav files found under {wav_dir}")
        return

    emo = Emotion2Vec(model_id=args.model_id)

    for wav_path in tqdm(wav_paths, desc="Emotion2Vec embeddings"):

        # Preserve relative structure:
        #   19/227/19_227_000008.wav
        rel = os.path.relpath(wav_path, wav_dir)
        rel_noext = os.path.splitext(rel)[0]
        out_path = os.path.join(out_dir, rel_noext + ".pt")

        # Ensure nested directories exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # skip if exists
        if os.path.exists(out_path) and not args.overwrite:
            continue

        # Extract as numpy, convert to torch
        emb = emo.extract_emotion_embeddings(wav_path)
        emb = torch.tensor(emb).float().cpu()

        # Save to disk
        torch.save(emb, out_path)

    print(f"\n✅ Emotion embeddings saved under: {out_dir}")


if __name__ == "__main__":
    main()
