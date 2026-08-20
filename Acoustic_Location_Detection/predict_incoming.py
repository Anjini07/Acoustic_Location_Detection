import os
import glob
import json
import shutil
import time
import numpy as np

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features
from drift_detection import load_global_stats, normalize, cosine_similarity, classify_similarity
from chunk_audio import chunk_file, CLIP_SECONDS
from quantize_and_hash import quantize_vector, compute_acoustic_hash

MULTI_BASELINES_FILE = "Acoustic_Location_Detection/multi_baselines.json"
INCOMING_DIR = "Acoustic_Location_Detection/incoming"
TEMP_CHUNK_DIR = "Acoustic_Location_Detection/incoming/_chunks"
PROCESSED_DIR = "Acoustic_Location_Detection/incoming/_processed"
PREDICTIONS_FILE = "Acoustic_Location_Detection/incoming_predictions.json"
LATEST_RUN_FILE = "Acoustic_Location_Detection/incoming_latest_run.json"

# Real captures carry the soundbox's actual hardware serial as device_id.
# Nothing in this offline pipeline has a real device attached yet, so every
# incoming file is hashed under this placeholder. Swap this for whatever
# metadata a real capture actually carries before this hash means anything
# for fraud/replay protection -- as-is it only demonstrates that the
# hashing math runs correctly, not device-specific binding.
INCOMING_DEVICE_ID = "unregistered_test_device"

# Label used when the best-matching location still scores in the
# RELOCATION_OR_FRAUD bucket -- see the note in predict_file() for why that
# bucket means something different here than it does in a claimed-location
# drift check.
UNKNOWN_LOCATION_LABEL = "UNKNOWN_NEW_LOCATION"


def load_baselines():
    with open(MULTI_BASELINES_FILE, "r") as f:
        return json.load(f)


def score_against_location(test_vector, location_baselines):
    scores = [
        cosine_similarity(test_vector, np.array(centroid))
        for centroid in location_baselines
    ]
    return max(scores)


def predict_file(file_path):
    global_mean, global_std = load_global_stats()
    baselines = load_baselines()

    # Split the incoming file into 10s windows first
    chunk_file(file_path, TEMP_CHUNK_DIR, prefix="incoming", skip_silent=True)
    chunk_paths = sorted(glob.glob(os.path.join(TEMP_CHUNK_DIR, "incoming_*.wav")))

    if not chunk_paths:
        print(f"\nFile: {file_path}")
        print(f"WARNING: no usable (non-silent) audio found — file is too quiet, "
              f"too short, or mostly silence. Cannot make a reliable prediction.")
        print(f"Suggest asking for the recording to be redone closer to the "
              f"environment, mic clearly exposed, for the full duration.")
        return {
            "file": file_path,
            "status": "REJECTED_LOW_QUALITY",
            "reason": "no usable non-silent windows found",
        }

    # Score every window against every location's multi-baseline
    per_window_best = []
    all_window_scores = {loc: [] for loc in baselines}
    window_hashes = []
    # Shared across every window in this file/run so hashes computed together
    # are comparable -- matches the spec's daily bucket, not a per-window clock.
    date_bucket = str(int(time.time()) // 86400)

    for chunk_path in chunk_paths:
        audio, sr = preprocess_audio(chunk_path)
        raw_features = extract_features(audio, sr)
        vector = normalize(raw_features, global_mean, global_std)

        window_scores = {}
        for location_name, location_baselines in baselines.items():
            score = score_against_location(vector, location_baselines)
            window_scores[location_name] = score
            all_window_scores[location_name].append(score)

        best_loc = max(window_scores, key=window_scores.get)
        per_window_best.append(best_loc)

        # Stage 6: same raw feature vector, quantized + hashed independently
        # of the scoring path above -- the hash never feeds into matching,
        # matching the spec's own compute_drift() which uses the raw float
        # vector, not the quantized one.
        E_quant = quantize_vector(raw_features, global_mean, global_std)
        window_hash, _ = compute_acoustic_hash(E_quant, INCOMING_DEVICE_ID, date_bucket=date_bucket)
        window_hashes.append(window_hash)

    # Aggregate: average score per location across all windows (more stable
    # than trusting any single 10s window), then majority vote as a cross-check
    avg_scores = {loc: float(np.mean(scores)) for loc, scores in all_window_scores.items()}
    final_match = max(avg_scores, key=avg_scores.get)

    sorted_scores = sorted(avg_scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]

    if margin > 0.15:
        confidence = "HIGH"
    elif margin > 0.05:
        confidence = "MODERATE"
    else:
        confidence = "LOW — scores too close to call reliably"

    low_window_warning = None
    if len(chunk_paths) <= 2:
        low_window_warning = (f"only {len(chunk_paths)} usable window(s) survived quality "
                               f"filtering — treat this prediction with extra caution.")

    final_score = avg_scores[final_match]
    status, action = classify_similarity(final_score)

    # classify_similarity()'s three buckets were designed for a *claimed*
    # location check (device says "I'm at office" -> is this score close
    # enough to office's own baseline?). Here there's no claim -- final_match
    # is already the BEST of every known location, so if even that top score
    # falls in RELOCATION_OR_FRAUD, it doesn't mean "moved from office" the
    # way it would in a drift check -- it means none of the known locations
    # are a good fit at all. Re-labeled accordingly for this open-set case.
    if status == "RELOCATION_OR_FRAUD":
        location_verdict = UNKNOWN_LOCATION_LABEL
        verdict_message = (
            f"This audio doesn't match any of the {len(baselines)} known "
            f"locations well enough (closest guess was '{final_match}', "
            f"score={final_score:.4f}). It may be a NEW location not yet in "
            f"the system, rather than one of the existing ones."
        )
    else:
        location_verdict = final_match
        verdict_message = None

    vote_counts = {loc: per_window_best.count(loc) for loc in baselines}

    print(f"\n{'='*60}")
    print(f"File: {file_path}")
    print(f"  ({len(chunk_paths)} internal window(s) of {CLIP_SECONDS}s scored and averaged)")
    print(f"  Average similarity per location: {avg_scores}")
    print(f"  Per-window votes: {vote_counts}")
    print(f"  Confidence: {confidence} (margin={margin:.4f})")
    if low_window_warning:
        print(f"  NOTE: {low_window_warning}")
    print(f"  acoustic_hash(es): {window_hashes}")
    if verdict_message:
        print(f">>> RESULT: DOES NOT MATCH ANY KNOWN LOCATION")
        print(f"    {verdict_message}")
    else:
        print(f">>> RESULT: {location_verdict}  (score={final_score:.4f}, {status})")
    print(f"{'='*60}")

    # Clean up temp chunks
    for p in chunk_paths:
        os.remove(p)

    return {
        "file": file_path,
        "num_windows": len(chunk_paths),
        "avg_scores": avg_scores,
        "vote_counts": vote_counts,
        "final_match": final_match,
        "final_score": final_score,
        "status": status,
        "location_verdict": location_verdict,
        "verdict_message": verdict_message,
        "margin": margin,
        "confidence": confidence,
        "acoustic_hashes": window_hashes,
        "device_id": INCOMING_DEVICE_ID,
        "date_bucket": date_bucket,
    }


def _load_existing_predictions():
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def _move_to_processed(file_path):
    """Moves an incoming file into incoming/_processed/ once it's been
    scored, so a later run of this script doesn't pick it up again. If a
    file of the same name is already there (e.g. re-running after manually
    copying a file back), it's suffixed rather than overwritten."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(file_path))
    dest = os.path.join(PROCESSED_DIR, os.path.basename(file_path))
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(PROCESSED_DIR, f"{base}_{n}{ext}")
        n += 1
    try:
        shutil.move(file_path, dest)
    except Exception as e:
        print(f"  [WARNING] could not move {file_path} to _processed/: {e}")


def main():
    os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    results = _load_existing_predictions()
    new_count = 0
    this_run_files = []  # basenames processed in THIS invocation only

    from audio_preprocessing import list_audio_files
    for file_path in list_audio_files(INCOMING_DIR):
        # Defensive: skip anything already inside the temp-chunk or
        # processed subfolders, in case list_audio_files recurses into
        # incoming/'s own subfolders rather than just its top level.
        normalized = os.path.normpath(file_path)
        parts = normalized.split(os.sep)
        if "_chunks" in parts or "_processed" in parts:
            continue

        result = predict_file(file_path)
        if result:
            results[os.path.basename(file_path)] = result
            new_count += 1
            this_run_files.append(os.path.basename(file_path))

        _move_to_processed(file_path)

    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Overwritten (not merged) every run -- this always reflects only the
    # most recent invocation, which is what plot_similarity_matrix.py reads
    # to plot just the latest batch instead of the full accumulated history.
    with open(LATEST_RUN_FILE, "w") as f:
        json.dump(this_run_files, f, indent=2)

    print(f"\nProcessed {new_count} new file(s), moved to incoming/_processed/.")
    print(f"Saved {len(results)} total result(s) to incoming_predictions.json")

    if new_count > 0:
        try:
            from plot_similarity_matrix import plot_from_incoming_predictions
            plot_from_incoming_predictions(
                PREDICTIONS_FILE,
                "Acoustic_Location_Detection/incoming_similarity_matrix.png",
                latest_only=True,
            )
        except Exception as e:
            print(f"[NOTE] Could not generate the similarity matrix plot: {e}")
    else:
        print("No new files this run -- skipping the plot (nothing new to show).")

if __name__ == "__main__":
    main()