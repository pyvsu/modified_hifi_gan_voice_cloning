import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class LibriTTSRDataset(Dataset):
    """
    PyTorch dataset for unit-based LibriTTS-R.

    This dataset assumes that the audio has already been preprocessed
    (resampled, trimmed, normalized) and that the following modality files
    have been extracted and aligned:

    - Discrete HuBERT + KMeans unit sequences (``.pt``)
    - ECAPA-TDNN speaker embeddings (``.pt``)
    - Emotion2Vec emotion embeddings (``.pt``)
    - Waveform audio (``.wav``)

    The dataset reads samples from a partition-specific ``.jsonl`` metadata file
    (``train.jsonl``, ``valid.jsonl``, or ``test.jsonl``) located in ``root_dir``.
    Each line describes file paths and metadata for a single utterance.

    All utterances are returned with variable-length waveforms and unit sequences.
    Padding is handled externally within the custom ``collate_fn`` for batching.

    Parameters
    ----------
    root_dir : str or Path
        Parent directory containing:
            - ``normalized_audio/``
            - ``unit_embeddings/``
            - ``speaker_embeddings/``
            - ``emotion_embeddings/``
            - ``train.jsonl``, ``valid.jsonl``, ``test.jsonl``

    split : {"train", "valid", "test"}
        Which metadata partition to load.

    Raises
    ------
    ValueError
        If ``split`` is not one of the supported strings.

    FileNotFoundError
        If the expected ``<split>.jsonl`` file does not exist.
    """

    def __init__(self, root_dir, split):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()

        if split.lower() not in ("train", "valid", "test"):
            raise ValueError("split must be one of: train, valid, test")

        self.jsonl_path = self.root_dir / f"{self.split}.jsonl"

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.jsonl_path}")

        # Load metadata lines
        with open(self.jsonl_path, "r", encoding="utf-8") as file:
            self.metadata = [json.loads(line) for line in file]

        # Will be used by LengthBucketSampler for efficient bucketing
        self.unit_lengths = [record.get("num_units", None) for record in self.metadata]


    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        sample = self.metadata[idx]

        # Build absolute paths
        wav_path = self.root_dir / sample["wav"]
        units_path = self.root_dir / sample["units"]
        speaker_emb_path = self.root_dir / sample["speaker_emb"]
        emotion_emb_path = self.root_dir / sample["emotion_emb"]

        # Load modalities
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.squeeze(0)  # [T]

        units = torch.load(units_path)
        speaker_emb = torch.load(speaker_emb_path)
        emotion_emb = torch.load(emotion_emb_path)

        # Ensure speaker/emotion embeddings are 1D
        speaker_emb = speaker_emb.view(-1)
        emotion_emb = emotion_emb.view(-1)

        return {
            "wav": wav,
            "units": units,
            "speaker_emb": speaker_emb,
            "emotion_emb": emotion_emb,
        }
