"""
calibrate_thresholds.py
Rule-based fallback classifier: derives quiet/noisy thresholds directly from
the features CSV's own noise_floor_rms distribution (percentile-based,
10th/90th), giving a zero-model, near-zero-memory classifier that can run
standalone, or serve as a sanity check against the trained ML model.
"""

import argparse
import json
import pandas as pd


def calibrate(csv_path: str, output_json: str) -> dict:
    df = pd.read_csv(csv_path)

    quiet_vals = df[df["label"] == "quiet"]["noise_floor_rms"]
    noisy_vals = df[df["label"] == "noisy"]["noise_floor_rms"]

    if quiet_vals.empty or noisy_vals.empty:
        raise ValueError("CSV must contain both 'quiet' and 'noisy' labeled rows")

    quiet_threshold = float(quiet_vals.quantile(0.90))
    noisy_threshold = float(noisy_vals.quantile(0.10))

    if quiet_threshold >= noisy_threshold:
        raise ValueError(
            "Invalid threshold ordering (quiet_threshold >= noisy_threshold) -- "
            "collect more/cleaner data before trusting these cutoffs"
        )

    thresholds = {
        "quiet_threshold": quiet_threshold,
        "noisy_threshold": noisy_threshold,
        "note": "noise_floor_rms <= quiet_threshold -> quiet; "
                ">= noisy_threshold -> noisy; else -> moderate",
    }

    with open(output_json, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(json.dumps(thresholds, indent=2))
    return thresholds


def classify_rule_based(noise_floor_rms: float, thresholds: dict) -> str:
    if noise_floor_rms <= thresholds["quiet_threshold"]:
        return "quiet"
    elif noise_floor_rms >= thresholds["noisy_threshold"]:
        return "noisy"
    return "moderate"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate quiet/noisy thresholds")
    parser.add_argument("--csv_path", default="Acoustic_Location_Detection2/features_labeled.csv")
    parser.add_argument("--output_json", default="Acoustic_Location_Detection2/env_thresholds.json")
    args = parser.parse_args()

    calibrate(args.csv_path, args.output_json)
