import os
import pandas as pd

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features

# --------------------------------------------------
# Dataset Path
# --------------------------------------------------

DATASET_PATH = "dataset"

dataset = []

# --------------------------------------------------
# Read every location
# --------------------------------------------------

for location in os.listdir(DATASET_PATH):

    location_path = os.path.join(DATASET_PATH, location)

    if not os.path.isdir(location_path):
        continue

    print(f"\nProcessing {location}")

    # Read every recording folder (part1, part2, ...)
    for recording in os.listdir(location_path):

        recording_path = os.path.join(location_path, recording)

        if not os.path.isdir(recording_path):
            continue

        print(f"  {recording}")

        # Read every clip
        for filename in os.listdir(recording_path):

            if filename.lower().endswith(".wav"):

                filepath = os.path.join(recording_path, filename)

                try:

                    # ----------------------------
                    # Preprocess Audio
                    # ----------------------------
                    audio, sr = preprocess_audio(filepath)

                    # ----------------------------
                    # Extract Features
                    # ----------------------------
                    features = extract_features(audio, sr)

                    # ----------------------------
                    # Add Labels
                    # ----------------------------
                    features.append(location)
                    features.append(recording)
                    features.append(filename)

                    dataset.append(features)

                    print(f"    Processed {filename}")

                except Exception as e:
                    print(f"Error processing {filepath}")
                    print(e)

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

columns = [f"Feature_{i+1}" for i in range(104)]
columns += ["Location", "Recording", "Filename"]

df = pd.DataFrame(dataset, columns=columns)

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

os.makedirs("output", exist_ok=True)

df.to_csv("output/features.csv", index=False)

print("\n===================================")
print("Dataset Created Successfully")
print("===================================")
print(df.head())
print("\nShape:", df.shape)