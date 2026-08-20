import os
import glob
import json

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features
from quantize_and_hash import quantize_vector, compute_acoustic_hash

DATASET_DIR = "Acoustic_Location_Detection/dataset"
GLOBAL_STATS_FILE = "Acoustic_Location_Detection/global_stats.json"
OUTPUT_FILE = "Acoustic_Location_Detection/hashes_output.json"

# Simple stand-in device IDs since we don't have real hardware yet.
# Map each location folder name to a fake device serial.
DEVICE_IDS = {
    "koramangala_shop": "device_koramangala_001",
    "pharmacy_shop": "device_pharmacy_001",
    "textile_shop": "device_textile_001",
}


def load_global_stats():
    with open(GLOBAL_STATS_FILE, "r") as f:
        data = json.load(f)
    return data["mean"], data["std"]


def main():
    global_mean, global_std = load_global_stats()
    results = {}

    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        device_id = DEVICE_IDS.get(location_name, f"device_{location_name}_001")
        wav_files = glob.glob(os.path.join(location_dir, "*.wav"))

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