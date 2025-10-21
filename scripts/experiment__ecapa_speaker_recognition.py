from speechbrain.inference.speaker import SpeakerRecognition

spanish_audio = "audio_samples/spanish.wav"
english_audio = "audio_samples/eng_man.wav"

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device":"cuda"}
)

# The prediction is 1 if the two signals in input are from the same speaker and 0 otherwise.
score, prediction = verification.verify_files(spanish_audio, english_audio)

print(score)
print(prediction)