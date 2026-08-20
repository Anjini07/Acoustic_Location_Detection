import numpy as np
import json

GLOBAL_STATS_FILE = "Acoustic_Location_Detection_AMR/global_stats.json"


def load_global_stats():
    with open(GLOBAL_STATS_FILE, "r") as f:
        data = json.load(f)
    return np.array(data["mean"]), np.array(data["std"])


def normalize(vector, global_mean, global_std, clip_sigma=3.0):
    z = (np.array(vector) - global_mean) / global_std
    return np.clip(z, -clip_sigma, clip_sigma)


def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norms = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / (norms + 1e-10))

# TODO -- PLACEHOLDER THRESHOLDS, NOT YET CALIBRATED.
# These are carried over from the WAV project purely so this module is
# importable/runnable before real AMR data exists. AMR's narrower
# 300-3400Hz feature space produces a different score distribution than
# WAV, so these MUST be replaced once you have:
#   1. A populated dataset/ (run chunk_audio.py on raw_recordings_amr/)
#   2. multi_baselines.json (run build_multi_baselines.py)
#   3. Real numbers from calibrate_thresholds.py run against THIS project
# Do not reuse the WAV project's calibrated values (0.3450/0.3933) --
# copy in whatever calibrate_thresholds.py prints for this AMR dataset.
def classify_similarity(score, stable_threshold=0.4829, moderate_threshold= 0.3788):
    """PLACEHOLDER thresholds -- see TODO above. Re-run calibrate_thresholds.py
    on this project's own AMR dataset and update the defaults here before
    trusting any classification results."""

    if score > stable_threshold:
        return "STABLE", "Same location"
    elif score > moderate_threshold:
        return "MODERATE_DRIFT", "Ambiguous zone — borderline, recommend recheck"
    else:
        return "RELOCATION_OR_FRAUD", "Likely different location"

def update_baseline_ema(baseline_vector, today_vector, beta=0.95):
    baseline_vector = np.array(baseline_vector)
    today_vector = np.array(today_vector)
    return beta * baseline_vector + (1 - beta) * today_vector
