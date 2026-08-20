"""
predict_environment.py
Predicts environment type for a new incoming AMR/WAV file of any length.
Already storage-optimal by design -- it does its own in-memory windowing on
the raw incoming file and never writes anything to disk, so no changes were
needed here for the streaming pipeline; included for a complete, drop-in set.
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from audio_preprocessing import load_audio_any_format, TARGET_SR
from feature_extraction import extract_features
from calibrate_thresholds import classify_rule_based

WINDOW_SECONDS = 5
SAMPLE_RATE = TARGET_SR


def predict(file_path: str, model_path: str, thresholds_path: str = None) -> dict:
    signal, sr = load_audio_any_format(file_path, SAMPLE_RATE)
    window_len = WINDOW_SECONDS * sr

    if len(signal) < window_len:
        windows = [signal]
    else:
        num_windows = len(signal) // window_len
        windows = [signal[i * window_len:(i + 1) * window_len] for i in range(num_windows)]

    bundle = joblib.load(model_path)
    model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]

    ml_predictions = []
    rule_predictions = []
    prob_sums = defaultdict(float)

    thresholds = None
    if thresholds_path:
        with open(thresholds_path) as f:
            thresholds = json.load(f)

    for w in windows:
        feats = extract_features(w, sr)
        x = pd.DataFrame([feats])[feature_cols]
        x_scaled = scaler.transform(x)

        ml_predictions.append(model.predict(x_scaled)[0])

        probs = model.predict_proba(x_scaled)[0]
        for class_name, p in zip(model.classes_, probs):
            prob_sums[class_name] += p

        if thresholds:
            rule_predictions.append(classify_rule_based(feats["noise_floor_rms"], thresholds))

    ml_vote = Counter(ml_predictions).most_common(1)[0]
    num_windows = len(windows)
    avg_probabilities = {cls: round(total / num_windows, 3) for cls, total in prob_sums.items()}

    result = {
        "file": file_path,
        "num_windows": num_windows,
        "ml_prediction": ml_vote[0],
        "ml_confidence": avg_probabilities[ml_vote[0]],
        "window_agreement": round(ml_vote[1] / num_windows, 3),
        "class_probabilities": avg_probabilities,
    }

    if rule_predictions:
        rule_vote = Counter(rule_predictions).most_common(1)[0]
        result["rule_based_prediction"] = rule_vote[0]
        result["agrees_with_rule_based"] = (rule_vote[0] == ml_vote[0])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict environment type for an audio file")
    parser.add_argument("file_path")
    parser.add_argument("--model_path", default="env_classifier.joblib")
    parser.add_argument("--thresholds_path", default="env_thresholds.json")
    args = parser.parse_args()

    result = predict(args.file_path, args.model_path, args.thresholds_path)
    print(json.dumps(result, indent=2))
