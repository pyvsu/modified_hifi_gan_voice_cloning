#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

echo "Executing LibriTTS-R Preprocesing Pipeline"

root="/mnt/d/LibriTTS-R"
new_sample_rate=16000
output_dir="${root}/LibriTTS-R-${new_sample_rate}"
num_workers=8
km_file="/home/cynthia/modified-hifi-gan-voice-cloning/pretrained_models/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"

splits=("dev-clean" "test-clean")

for split in "${splits[@]}"; do
    echo ">>> Starting split: $split"

    echo ">>> Task 1: Resample audio to $new_sample_rate Hz"
    python preprocessing/01_resample_audio.py \
        --in-root "${root}/LibriTTS_R/${split}" \
        --out-root "${output_dir}/resampled_audio/${split}" \
        --new-sr $new_sample_rate \
        --num-workers $num_workers
    
    echo ">>> Task 2: Silence trimming using Silero VAD"
    python preprocessing/02_trim_silence.py \
        --in-root "${output_dir}/resampled_audio/${split}" \
        --out-root "${output_dir}/trimmed_audio/${split}" \
        --trust-repo

    echo ">>> Task 3: Normalizing audio to -23 LUFS"
    python preprocessing/03_normalize_loudness.py \
        --in-root "${output_dir}/trimmed_audio/${split}" \
        --out-root "${output_dir}/normalized_audio/${split}" \
        --num-workers $num_workers

    echo ">>> Task 4: Extracting discrete speech units using mHuBERT"
    python preprocessing/04_extract_discrete_speech_units.py \
        --in-root "${output_dir}/normalized_audio/${split}" \
        --out-root "${output_dir}/unit_embeddings/${split}" \
        --km-file "$km_file"

    echo ">>> Task 5: Extracting speaker embeddings using ECAPA-TDNN"
    python -m preprocessing.05_extract_speaker_embeddings \
        --wav-dir "${output_dir}/normalized_audio/${split}" \
        --out-dir "${output_dir}/speaker_embeddings/${split}"

    echo ">>> Task 6: Extracting emotion embeddings using emotion2vec"
    python -m preprocessing.06_extract_emotion_embeddings \
        --wav-dir "${output_dir}/normalized_audio/${split}" \
        --out-dir "${output_dir}/emotion_embeddings/${split}"
done

echo ">>> Task 7: Building metadata files for all splits"
python preprocessing/07_build_metadata.py --root "$output_dir"
