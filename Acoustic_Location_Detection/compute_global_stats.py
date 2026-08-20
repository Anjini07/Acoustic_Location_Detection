import os
import glob
import json
import numpy as np

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features

DATASET_DIR = "Acoustic_Location_Detection/dataset"
OUTPUT_FILE = "Acoustic_Location_Detection/global_stats.json"


def main():
    all_vectors = []
    file_count = 0

    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        wav_files = glob.glob(os.path.join(location_dir, "*.wav"))
        for wav_path in wav_files:
            print(f"Processing: {wav_path}")
            audio, sr = preprocess_audio(wav_path)
            features = extract_features(audio, sr)
            all_vectors.append(features)
            file_count += 1

    all_vectors = np.array(all_vectors)  # shape: (num_files, 104)

    global_mean = all_vectors.mean(axis=0)
    global_std = all_vectors.std(axis=0) + 1e-8  # avoid divide-by-zero

    output = {
        "mean": global_mean.tolist(),
        "std": global_std.tolist(),
        "num_files_used": file_count,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Used {file_count} files to compute global stats.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()