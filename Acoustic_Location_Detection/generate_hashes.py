import os
import json

from audio_preprocessing import preprocess_audio, list_audio_files
from feature_extraction import extract_features
from quantize_and_hash import quantize_vector, compute_acoustic_hash

DATASET_DIR = "Acoustic_Location_Detection/dataset"
GLOBAL_STATS_FILE = "Acoustic_Location_Detection/global_stats.json"
OUTPUT_FILE = "Acoustic_Location_Detection/hashes_output.json"

# CHANGE: dropped the old DEVICE_IDS dict -- it mapped fictional shop names
# (koramangala_shop, pharmacy_shop, textile_shop) that don't match this
# project's actual locations (cafeteria/nature/office/traffic/metro/home),
# so it was silently falling through to this exact naming convention anyway
# via the old .get() fallback. One fake device per location is a stand-in
# for not having real hardware yet -- in production, device_id comes from
# the soundbox's own hardware serial, not the location.
def device_id_for(location_name):
    return f"device_{location_name}_001"


def load_global_stats():
    with open(GLOBAL_STATS_FILE, "r") as f:
        data = json.load(f)
    return data["mean"], data["std"]


def main():
    """Hashes every clip already in dataset/ (the training corpus) as a
    bulk/offline demonstration that Stage 6 (quantize + SHA-256) runs
    correctly across the dataset. This is NOT the live path -- real incoming
    captures are hashed as they're scored, in predict_incoming.py, so a hash
    and its similarity result stay attached to the same capture."""
    global_mean, global_std = load_global_stats()
    results = {}

    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        device_id = device_id_for(location_name)
        wav_files = list_audio_files(location_dir)

        for wav_path in wav_files:
            print(f"Processing: {wav_path}")

            # Stage 1: preprocess
            audio, sr = preprocess_audio(wav_path)

            # Stages 2-5: extract 104-dim feature vector
            features = extract_features(audio, sr)

            # Stage 6: quantize + hash
            E_quant = quantize_vector(features, global_mean, global_std)
            acoustic_hash, date_bucket = compute_acoustic_hash(E_quant, device_id)

            file_key = os.path.basename(wav_path)
            results[file_key] = {
                "location": location_name,
                "device_id": device_id,
                "date_bucket": date_bucket,
                "acoustic_hash": acoustic_hash,
                "quantized_vector": E_quant.tolist(),
            }

            print(f"  -> hash: {acoustic_hash}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Saved all hashes to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
