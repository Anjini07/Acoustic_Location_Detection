import numpy as np
import json
import os

GLOBAL_STATS_FILE = "Acoustic_Location_Detection/global_stats.json"
CALIBRATED_THRESHOLDS_FILE = "Acoustic_Location_Detection/calibrated_thresholds.json"

# Used only if calibrate_thresholds.py has never been run yet -- once it has,
# classify_similarity() picks up its output from CALIBRATED_THRESHOLDS_FILE
# automatically, so these fallback numbers stop mattering.
_FALLBACK_STABLE_THRESHOLD = 0.3277
_FALLBACK_MODERATE_THRESHOLD = 0.5059

_cached_thresholds = None  # (stable, moderate), loaded once per process


def load_global_stats():
    with open(GLOBAL_STATS_FILE, "r") as f:
        data = json.load(f)
    return np.array(data["mean"]), np.array(data["std"])


def load_calibrated_thresholds(force_reload=False):
    """Reads the stable/moderate threshold pair written by
    calibrate_thresholds.py, caching it for the rest of the process. Falls
    back to the hardcoded defaults above (with a one-time warning) if
    calibrate_thresholds.py hasn't been run yet -- e.g. right after adding a
    brand new location, before you've had a chance to recalibrate."""
    global _cached_thresholds
    if _cached_thresholds is not None and not force_reload:
        return _cached_thresholds

    if os.path.exists(CALIBRATED_THRESHOLDS_FILE):
        with open(CALIBRATED_THRESHOLDS_FILE, "r") as f:
            data = json.load(f)
        _cached_thresholds = (data["stable_threshold"], data["moderate_threshold"])
    else:
        print(f"  [NOTE] {CALIBRATED_THRESHOLDS_FILE} not found -- using fallback thresholds "
              f"({_FALLBACK_STABLE_THRESHOLD}, {_FALLBACK_MODERATE_THRESHOLD}). "
              f"Run calibrate_thresholds.py to generate calibrated ones.")
        _cached_thresholds = (_FALLBACK_STABLE_THRESHOLD, _FALLBACK_MODERATE_THRESHOLD)

    return _cached_thresholds


def normalize(vector, global_mean, global_std, clip_sigma=3.0):
    z = (np.array(vector) - global_mean) / global_std
    return np.clip(z, -clip_sigma, clip_sigma)


def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norms = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / (norms + 1e-10))


def classify_similarity(score, stable_threshold=None, moderate_threshold=None):
    # CHANGE: thresholds are now optional. Pass explicit values if you want
    # to override (e.g. for testing), otherwise they're loaded automatically
    # from calibrate_thresholds.py's saved output -- no more manually typing
    # numbers from its printout into this file.
    if stable_threshold is None or moderate_threshold is None:
        auto_stable, auto_moderate = load_calibrated_thresholds()
        if stable_threshold is None:
            stable_threshold = auto_stable
        if moderate_threshold is None:
            moderate_threshold = auto_moderate

    # Order-agnostic on purpose: previously this compared score >
    # stable_threshold first regardless of which of the two thresholds was
    # numerically higher -- if they were ever inverted (as the old hardcoded
    # defaults were), the MODERATE_DRIFT branch became dead code. max/min
    # here means a mislabeled pair can't silently break classification.
    hi = max(stable_threshold, moderate_threshold)
    lo = min(stable_threshold, moderate_threshold)

    if score > hi:
        return "STABLE", "Same location"
    elif score > lo:
        return "MODERATE_DRIFT", "Ambiguous zone — borderline, recommend recheck"
    else:
        return "RELOCATION_OR_FRAUD", "Likely different location"


def update_baseline_ema(baseline_vector, today_vector, beta=0.95):
    baseline_vector = np.array(baseline_vector)
    today_vector = np.array(today_vector)
    return beta * baseline_vector + (1 - beta) * today_vector