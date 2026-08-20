import os
import json
import time
import argparse
import numpy as np

from audio_preprocessing import preprocess_audio, list_audio_files
from feature_extraction import extract_features
from drift_detection import load_global_stats, normalize, cosine_similarity

DATASET_DIR = "Acoustic_Location_Detection/dataset"
MULTI_BASELINES_FILE = "Acoustic_Location_Detection/multi_baselines.json"
CALIBRATED_THRESHOLDS_FILE = "Acoustic_Location_Detection/calibrated_thresholds.json"


def load_multi_baselines():
    with open(MULTI_BASELINES_FILE, "r") as f:
        return json.load(f)


def score_against_location(test_vector, location_baselines):
    scores = [
        cosine_similarity(test_vector, np.array(centroid))
        for centroid in location_baselines
    ]
    return max(scores)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate STABLE/MODERATE/RELOCATION thresholds from dataset/ scores."
    )
    parser.add_argument("--stable-percentile", type=float, default=10,
                         help="Percentile of same-location scores used for stable_threshold (default 10).")
    parser.add_argument("--moderate-percentile", type=float, default=90,
                         help="Percentile of different-location scores used for moderate_threshold (default 90). "
                              "Raise this to make MODERATE_DRIFT/unknown-location less likely to fire.")
    parser.add_argument("--override-stable", type=float, default=None,
                         help="Skip the percentile calculation and use this exact value for stable_threshold.")
    parser.add_argument("--override-moderate", type=float, default=None,
                         help="Skip the percentile calculation and use this exact value for moderate_threshold. "
                              "Lower this if correct matches are being flagged as an unknown/new location too often.")
    args = parser.parse_args()

    global_mean, global_std = load_global_stats()
    baselines = load_multi_baselines()

    same_location_scores = []   # e.g. home recording vs home's own sub-baselines
    diff_location_scores = []   # e.g. home recording vs metro/office sub-baselines

    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        for wav_path in list_audio_files(location_dir):
            audio, sr = preprocess_audio(wav_path)
            vector = normalize(extract_features(audio, sr), global_mean, global_std)

            for baseline_location, location_baselines in baselines.items():
                score = score_against_location(vector, location_baselines)

                if baseline_location == location_name:
                    same_location_scores.append(score)
                else:
                    diff_location_scores.append(score)

    same = np.array(same_location_scores)
    diff = np.array(diff_location_scores)

    print(f"Same-location scores:      mean={same.mean():.4f}  min={same.min():.4f}  max={same.max():.4f}")
    print(f"Different-location scores: mean={diff.mean():.4f}  min={diff.min():.4f}  max={diff.max():.4f}")

    if args.override_stable is not None:
        stable_threshold = args.override_stable
        print(f"\nUsing manual override for stable_threshold: {stable_threshold:.4f}")
    else:
        stable_threshold = float(np.percentile(same, args.stable_percentile))

    if args.override_moderate is not None:
        moderate_threshold = args.override_moderate
        print(f"Using manual override for moderate_threshold: {moderate_threshold:.4f}")
    else:
        moderate_threshold = float(np.percentile(diff, args.moderate_percentile))

    print(f"\nThresholds in use:")
    print(f"  STABLE     if score > {stable_threshold:.4f}")
    print(f"  MODERATE   if {moderate_threshold:.4f} < score <= {stable_threshold:.4f}")
    print(f"  RELOCATION if score <= {moderate_threshold:.4f}")

    if moderate_threshold >= stable_threshold:
        print(f"\n  [NOTE] moderate_threshold ({moderate_threshold:.4f}) >= stable_threshold "
              f"({stable_threshold:.4f}) -- there's real overlap between same- and "
              f"different-location scores in this dataset (drift_detection.py's "
              f"classify_similarity() handles this order-agnostically so it still works, "
              f"but a MODERATE_DRIFT verdict will be rare/nonexistent until the baselines "
              f"separate better).")

    # CHANGE: write the calibrated pair out instead of only printing it.
    # drift_detection.py's classify_similarity() reads this file automatically,
    # so there's no more manually copying these numbers over.
    with open(CALIBRATED_THRESHOLDS_FILE, "w") as f:
        json.dump({
            "stable_threshold": stable_threshold,
            "moderate_threshold": moderate_threshold,
            "same_location_stats": {
                "mean": float(same.mean()), "min": float(same.min()), "max": float(same.max()),
                "n": int(len(same)),
            },
            "diff_location_stats": {
                "mean": float(diff.mean()), "min": float(diff.min()), "max": float(diff.max()),
                "n": int(len(diff)),
            },
            "locations": sorted(baselines.keys()),
            "manual_override": args.override_stable is not None or args.override_moderate is not None,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    print(f"\nSaved to {CALIBRATED_THRESHOLDS_FILE} -- drift_detection.py will pick these up "
          f"automatically on its next call, no manual edit needed.")


if __name__ == "__main__":
    main()