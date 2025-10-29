import torchaudio


def save_audio(path, waveform, sample_rate=16000):
    """
    Saves a single waveform tensor [T] or batch [B,1,T]
    """
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0).squeeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    torchaudio.save(path, waveform.unsqueeze(0).cpu(), sample_rate)
    print(f"[audio] Saved → {path}")
