from funasr import AutoModel

# DATA_PATH = "/mnt/d/streamspeech_datasets/cvss/cvss-t/es_en/test"
# AUDIO_SAMPLE = "common_voice_es_18307009.mp3.wav"
# AUDIO_PATH = f"{DATA_PATH}/{AUDIO_SAMPLE}"

AUDIO_PATH = "audio_samples/eng_man.wav"

# model="iic/emotion2vec_base"
# model="iic/emotion2vec_base_finetuned"
# model="iic/emotion2vec_plus_seed"
# model="iic/emotion2vec_plus_base"
model_id = "iic/emotion2vec_plus_base"

model = AutoModel(
    model=model_id,
    hub="hf",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

rec_result = model.generate(
    AUDIO_PATH,
    output_dir="./emotion2vec_outputs",
    granularity="utterance",
    extract_embedding=True
)

sample_result = rec_result[0]
predicted_label = max(sample_result["labels"])
predicted_score = max(sample_result["scores"])
embeddings = sample_result["feats"]

print("Emotion Classification")
print(f"Predicted Label: {predicted_label}")
print(f"Predicted Score: {predicted_score}\n")

print("Embeddings:")
print(embeddings.shape)
print(embeddings)


# print(f"Labels:", rec_result["labels"])
# print(f"Scores:", rec_result["scores"])
# print(f"Embeddings:", rec_result["feat"])

