import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogram(nn.Module):
    """
    Log-mel extractor compatible with HiFi-GAN settings.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        fmin=0,
        fmax=8000,
        power=1.0,
        center=True,
        eps=1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.power = power
        self.center = center
        self.eps = eps

        mel: torch.Tensor = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
        self.register_buffer("mel_filter", mel)
        self.mel_filter: torch.Tensor

    def forward(self, waveform):  # waveform: [B, T]
        window = torch.hann_window(self.win_length, device=waveform.device)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=self.center,
        )

        spectrogram = (stft.real**2 + stft.imag**2) ** (self.power / 2.0)

        mel_filter = self.mel_filter.to(waveform.device)
        mel_spec = torch.matmul(mel_filter, spectrogram)
        mel_spec = torch.log(mel_spec.clamp_min(self.eps))
        return mel_spec


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """
    Builds Slaney-style mel filterbank like librosa.
    """
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freqs = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    filterbank = np.zeros((n_mels, len(freqs)), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left, center, right = hz_points[i - 1 : i + 2]
        l_slope = (freqs - left) / (center - left)
        r_slope = (right - freqs) / (right - center)
        filterbank[i - 1] = np.maximum(0.0, np.minimum(l_slope, r_slope))

    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]
    return torch.tensor(filterbank)
