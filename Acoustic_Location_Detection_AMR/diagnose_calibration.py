import os
import numpy as np

from audio_preprocessing import preprocess_audio, list_audio_files
from feature_extraction import extract_features
from drift_detection import load_global_stats, normalize, cosine_similarity
from calibrate_thresholds import load_multi_baselines, score_against_location, DATASET_DIR

N_WORST = 5  # how many outlier clips to print per direction


def main():
    global_mean, global_std = load_global_stats()
    baselines = load_multi_baselines()

    per_location_same = {loc: [] for loc in baselines}   # loc -> [(score, file), ...]
    per_location_diff_max = {loc: [] for loc in baselines}  # loc -> [(worst_diff_score, other_loc, file), ...]

    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue
        if location_name not in baselines:
            print(f"WARNING: {location_name} has clips in dataset/ but no entry in "
                  f"multi_baselines.json -- rerun build_multi_baselines.py?")
            continue

        for clip_path in list_audio_files(location_dir):
            audio, sr = preprocess_audio(clip_path)
            vector = normalize(extract_features(audio, sr), global_mean, global_std)

            same_score = score_against_location(vector, baselines[location_name])
            per_location_same[location_name].append((same_score, clip_path))

            worst_other_loc, worst_other_score = None, -2.0
            for other_loc, other_baselines in baselines.items():
                if other_loc == location_name:
                    continue
                s = score_against_location(vector, other_baselines)
                if s > worst_other_score:
                    worst_other_score = s
                    worst_other_loc = other_loc
            per_location_diff_max[location_name].append((worst_other_score, worst_other_loc, clip_path))

    print(f"{'Location':<20}{'#clips':>8}{'same mean':>12}{'same min':>12}{'same max':>12}"
          f"{'max cross-match':>18}")
    for loc in sorted(per_location_same):
        same_scores = np.array([s for s, _ in per_location_same[loc]])
        cross_scores = np.array([s for s, _, _ in per_location_diff_max[loc]])
        if len(same_scores) == 0:
            continue
        print(f"{loc:<20}{len(same_scores):>8}{same_scores.mean():>12.4f}"
              f"{same_scores.min():>12.4f}{same_scores.max():>12.4f}{cross_scores.max():>18.4f}")

    print(f"\n--- Worst same-location matches (own clips scoring lowest against own baseline) ---")
    all_same = []
    for loc, entries in per_location_same.items():
        for score, path in entries:
            all_same.append((score, loc, path))
    all_same.sort(key=lambda x: x[0])
    for score, loc, path in all_same[:N_WORST]:
        print(f"  {score:.4f}  [{loc}]  {path}")

    print(f"\n--- Worst cross-location confusions (a clip scoring high against a DIFFERENT location) ---")
    all_cross = []
    for loc, entries in per_location_diff_max.items():
        for score, other_loc, path in entries:
            all_cross.append((score, loc, other_loc, path))
    all_cross.sort(key=lambda x: -x[0])
    for score, loc, other_loc, path in all_cross[:N_WORST]:
        print(f"  {score:.4f}  [{loc}] clip scored this high against [{other_loc}]'s baseline  {path}")

    print(f"\nIf the worst rows above cluster around one or two locations, that's your problem "
          f"location(s) -- likely too few clips (check the #clips column) or a location whose "
          f"AMR recordings are unusually noisy/quiet relative to its own baseline. Consider "
          f"adding more raw recordings (or simulate_amr_roundtrip.py output) for whichever "
          f"location(s) show up repeatedly here before re-running calibrate_thresholds.py.")


if __name__ == "__main__":
    main()
