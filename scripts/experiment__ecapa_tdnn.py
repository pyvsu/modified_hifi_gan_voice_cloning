import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


DATA_PATH = "/mnt/d/streamspeech_datasets/cvss/cvss-t/es_en/test"
AUDIO_SAMPLE = "common_voice_es_18307009.mp3.wav"

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

signal, fs = torchaudio.load(f"{DATA_PATH}/{AUDIO_SAMPLE}")

embeddings = classifier.encode_batch(signal)

print(embeddings)
print(embeddings.shape)