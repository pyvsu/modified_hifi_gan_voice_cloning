"""
Custom collate function for batching variable-length waveforms
and discrete unit sequences.

Pads sequences to the max length within the batch. Useful for
HiFi-GAN training or fine-tuning where each sample may vary in duration.
"""

import torch

def collate_fn(batch):
    """
    Collate function to combine variable-length samples into a batch.

    Parameters
    ----------
    batch : list of dict
        Each element is a sample from LibriTTSRDataset with keys:
        - "wav"         : Tensor [T]
        - "units"       : Tensor [U]
        - "speaker_emb" : Tensor [E]
        - "emotion_emb" : Tensor [E]

    Returns
    -------
    dict
        A dictionary containing:
        - "wav"         : Tensor [B, T_max]
        - "wav_lengths" : Tensor [B]
        - "units"       : Tensor [B, U_max]
        - "unit_lengths": Tensor [B]
        - "speaker_emb" : Tensor [B, E]
        - "emotion_emb" : Tensor [B, E]
    """

    # Sort longest → shortest by unit length (optional but improves packing)
    sorted_batch = sorted(batch, key=lambda x: x["units"].shape[0], reverse=True)

    # Extract lists
    wavs  = [batch["wav"] for batch in sorted_batch]
    units = [batch["units"] for batch in sorted_batch]
    speaker_emb   = torch.stack([batch["speaker_emb"] for batch in sorted_batch])
    emotion_emb   = torch.stack([batch["emotion_emb"] for batch in sorted_batch])

    # Length tensors
    wav_lengths  = torch.tensor([len(wav) for wav in wavs], dtype=torch.long)
    unit_lengths = torch.tensor([len(unit) for unit in units], dtype=torch.long)

    # Pad waveforms
    max_wav = int(wav_lengths.max())
    padded_wav = torch.zeros(len(batch), max_wav)
    for i, wav in enumerate(wavs):
        padded_wav[i, : len(wav)] = wav

    # Pad units (token IDs, typically)
    max_units = int(unit_lengths.max())
    padded_units = torch.zeros(len(batch), max_units, dtype=units[0].dtype)
    for i, unit in enumerate(units):
        padded_units[i, : len(unit)] = unit

    return {
        "wav": padded_wav,                 # [B, T]
        "wav_lengths": wav_lengths,        # [B]
        "units": padded_units,             # [B, U]
        "unit_lengths": unit_lengths,      # [B]
        "speaker_emb": speaker_emb,        # [B, E]
        "emotion_emb": emotion_emb,        # [B, E]
    }
