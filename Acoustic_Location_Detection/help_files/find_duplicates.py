import os
import glob
import hashlib

DATASET_DIR = "Acoustic_Location_Detection/dataset"
TEST_DIR = "Acoustic_Location_Detection/test_audio"


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    seen = {}  # hash -> list of paths

    for base_dir in [DATASET_DIR, TEST_DIR]:
        for wav_path in glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True):
            h = file_hash(wav_path)
            seen.setdefault(h, []).append(wav_path)

    duplicates_found = False
    for h, paths in seen.items():
        if len(paths) > 1:
            duplicates_found = True
            print("DUPLICATE CONTENT:")
            for p in paths:
                print(f"  {p}")
            print()

    if not duplicates_found:
        print("No exact duplicate files found.")


if __name__ == "__main__":
    main()