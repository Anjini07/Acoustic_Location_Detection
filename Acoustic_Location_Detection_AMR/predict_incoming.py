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

MULTI_BASELINES_FILE = "Acoustic_Location_Detection_AMR/multi_baselines.json"
INCOMING_DIR = "Acoustic_Location_Detection_AMR/incoming"
TEMP_CHUNK_DIR = "Acoustic_Location_Detection_AMR/incoming/_chunks"
PROCESSED_DIR = "Acoustic_Location_Detection_AMR/incoming/_processed"
PREDICTED_LEDGER_FILE = "Acoustic_Location_Detection_AMR/predicted_files.json"


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

    # Split the incoming file into 5s windows first
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

    for chunk_path in chunk_paths:
        audio, sr = preprocess_audio(chunk_path)
        vector = normalize(extract_features(audio, sr), global_mean, global_std)

        window_scores = {}
        for location_name, location_baselines in baselines.items():
            score = score_against_location(vector, location_baselines)
            window_scores[location_name] = score
            all_window_scores[location_name].append(score)

        best_loc = max(window_scores, key=window_scores.get)
        per_window_best.append(best_loc)

    # Aggregate: average score per location across all windows (more stable
    # than trusting any single window), then majority vote as a cross-check
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

    print(f"Confidence: {confidence} (margin={margin:.4f})")

    # Warn if very few usable windows came through — even a "successful"
    # prediction from just 1-2 windows is much less reliable than one from many
    if len(chunk_paths) <= 2:
        print(f"NOTE: only {len(chunk_paths)} usable window(s) survived quality "
              f"filtering — treat this prediction with extra caution.")
        
    final_score = avg_scores[final_match]
    status, action = classify_similarity(final_score)

    vote_counts = {loc: per_window_best.count(loc) for loc in baselines}

    print(f"\nFile: {file_path}")
    print(f"Split into {len(chunk_paths)} windows of {CLIP_SECONDS}s")
    print(f"Average similarity per location: {avg_scores}")
    print(f"Per-window votes: {vote_counts}")
    print(f"Final match: {final_match} (avg score={final_score:.4f}) -> {status}")

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
        "margin": margin,
        "confidence": confidence,
    }


def _move_to_processed(file_path, retries=5, delay_seconds=0.5):
    """Best-effort tidy-up: moves an already-predicted file out of incoming/
    into incoming/_processed/. NOT relied on for correctness -- whether a
    file gets re-predicted is controlled entirely by predicted_files.json
    (see load_ledger/save_ledger below), because on Windows a file can stay
    locked by another process (antivirus scan, sync client, a lingering
    ffmpeg handle) for longer than any reasonable retry loop here should
    wait. Retries briefly, and just leaves the file in place with a warning
    if it still can't move -- the ledger means that's harmless, not a bug."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    dest = os.path.join(PROCESSED_DIR, os.path.basename(file_path))
    if os.path.exists(dest):
        base, ext = os.path.splitext(os.path.basename(file_path))
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(PROCESSED_DIR, f"{base}_{n}{ext}")
            n += 1

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            shutil.move(file_path, dest)
            return dest
        except PermissionError as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay_seconds)

    print(f"  NOTE: could not move {file_path} to _processed/ after {retries} "
          f"attempts ({last_error}) -- probably still locked by another process "
          f"(antivirus/sync). It's already recorded as predicted though, so it "
          f"won't be re-scored -- move it out of incoming/ manually whenever "
          f"the lock clears, purely for tidiness.")
    return None


def load_ledger():
    """Files already predicted, by filename. This -- not whether the file
    could be moved out of incoming/ -- is what prevents re-prediction, so a
    stubborn Windows file lock on the move can never cause the same audio
    to be scored over and over."""
    if os.path.exists(PREDICTED_LEDGER_FILE):
        with open(PREDICTED_LEDGER_FILE, "r") as f:
            return json.load(f)
    return {}


def save_ledger(ledger):
    with open(PREDICTED_LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=2)


def _save_result(filename, result):
    """Merges one file's result into incoming_predictions.json immediately,
    rather than batching all results to write at the very end -- so a crash
    partway through a large incoming/ folder doesn't lose predictions that
    already completed."""
    output_path = "Acoustic_Location_Detection_AMR/incoming_predictions.json"
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing = json.load(f)
    existing[filename] = result
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ledger = load_ledger()

    from audio_preprocessing import list_audio_files
    all_files = list_audio_files(INCOMING_DIR)
    new_files = [f for f in all_files if os.path.basename(f) not in ledger]

    if not new_files:
        print("No new files in incoming/ to predict (everything here is already in predicted_files.json).")
        return

    processed_count = 0
    for file_path in new_files:
        filename = os.path.basename(file_path)
        result = predict_file(file_path)

        if result:
            _save_result(filename, result)

        # Mark as predicted BEFORE attempting the move -- this is the step
        # that actually prevents re-prediction, so it must happen even if
        # the move below fails.
        ledger[filename] = {"status": (result or {}).get("status", "UNKNOWN")}
        save_ledger(ledger)
        processed_count += 1

        moved_to = _move_to_processed(file_path)
        if moved_to:
            print(f"  Moved to {moved_to}")

    print(f"\nPredicted {processed_count} new file(s). Results saved to incoming_predictions.json.")

if __name__ == "__main__":
    main()