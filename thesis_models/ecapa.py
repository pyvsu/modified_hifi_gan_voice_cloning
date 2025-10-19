import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition


class ECAPA:

    def __init__(self, device="cuda") -> None:
        self.ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={ "device": device }
        )

    @torch.no_grad()
    def extract_speaker_embeddings(self, wav_path):
        signal, sample_rate = torchaudio.load(wav_path)
        embeddings = self.ecapa.encode_batch(signal)
        return embeddings
