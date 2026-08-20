import os
import glob
import json
import numpy as np

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features
from drift_detection import load_global_stats, normalize, cosine_similarity

DATASET_DIR = "Acoustic_Location_Detection_AMR/dataset"
MULTI_BASELINES_FILE = "Acoustic_Location_Detection_AMR/multi_baselines.json"


def load_multi_baselines():
    with open(MULTI_BASELINES_FILE, "r") as f:
        return json.load(f)


def score_against_location(test_vector, location_baselines):
    """A location can have multiple baselines (different conditions).
    The location's score is the BEST match among its own sub-baselines —
    the test clip only needs to resemble ONE known condition for that
    location, not all of them. Centroids in multi_baselines.json are
    already normalized (they were built from normalized vectors), so no
    extra normalize() call is needed here."""
    scores = [
        cosine_similarity(test_vector, np.array(centroid))
        for centroid in location_baselines
    ]
    return max(scores)


def main():
    global_mean, global_std = load_global_stats()
    baselines = load_multi_baselines()

    same_location_scores = []   # e.g. home recording vs home's own sub-baselines
    diff_location_scores = []   # e.g. home recording vs metro/office sub-baselines

    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        from audio_preprocessing import list_audio_files
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

    # A single 10th/90th percentile pair can get flipped by a handful of
    # outlier clips (see diagnose_calibration.py for finding those). Trying
    # several percentile pairs and only keeping the ones where
    # stable_threshold > moderate_threshold actually holds shows the real
    # tradeoff: tighter pairs (closer to the median) are more robust to
    # outliers but call more borderline scores MODERATE_DRIFT; wider pairs
    # (closer to 10/90) give a narrower ambiguous band but are more exposed
    # to outliers flipping the ordering entirely, as just happened.
    candidate_pairs = [(10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (40, 60)]

    print(f"\nCandidate threshold pairs (same-percentile / diff-percentile):")
    print(f"{'same %ile':>10}{'diff %ile':>10}{'stable_threshold':>18}{'moderate_threshold':>20}{'valid?':>10}")
    valid_pairs = []
    for same_pct, diff_pct in candidate_pairs:
        stable_threshold = np.percentile(same, same_pct)
        moderate_threshold = np.percentile(diff, diff_pct)
        is_valid = stable_threshold > moderate_threshold
        if is_valid:
            valid_pairs.append((same_pct, diff_pct, stable_threshold, moderate_threshold))
        print(f"{same_pct:>10}{diff_pct:>10}{stable_threshold:>18.4f}{moderate_threshold:>20.4f}"
              f"{'OK' if is_valid else 'INVERTED':>10}")

    if not valid_pairs:
        print(f"\nNo candidate pair maintains stable_threshold > moderate_threshold.")
        print(f"This means same-location and different-location scores overlap too much")
        print(f"across the board -- not just at extreme percentiles. Run")
        print(f"diagnose_calibration.py to find which location(s)/clips are driving the")
        print(f"overlap before picking any threshold; pasting a broken pair into")
        print(f"drift_detection.py will silently disable the MODERATE_DRIFT bucket.")
    else:
        # Prefer the widest valid pair (closest to 10/90) -- narrowest ambiguous
        # band that still holds up against the outliers actually in this dataset.
        best = valid_pairs[0]
        same_pct, diff_pct, stable_threshold, moderate_threshold = best
        print(f"\nRecommended (widest valid pair -- {same_pct}th/{diff_pct}th percentile):")
        print(f"  STABLE     if score > {stable_threshold:.4f}")
        print(f"  MODERATE   if {moderate_threshold:.4f} < score <= {stable_threshold:.4f}")
        print(f"  RELOCATION if score <= {moderate_threshold:.4f}")
        print(f"\nOnly copy these into drift_detection.py's classify_similarity() defaults")
        print(f"once you're satisfied the outlier clips behind any narrower/invalid pairs")
        print(f"above are either fixed or acceptable (see diagnose_calibration.py).")
        print(f"Do NOT reuse the WAV project's calibrated thresholds (0.3450/0.3933) --")
        print(f"AMR's narrower 300-3400Hz feature space has a different distribution.")


if __name__ == "__main__":
    main()
