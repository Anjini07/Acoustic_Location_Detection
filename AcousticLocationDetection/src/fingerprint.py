import numpy as np
import hashlib
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Stage 6 : Normalize Feature Vector
# ============================================================

def normalize_vector(feature_vector, mean, std):
    """
    Z-score normalization

    Parameters
    ----------
    feature_vector : ndarray (104,)
    mean : ndarray (104,)
    std : ndarray (104,)

    Returns
    -------
    ndarray
    """

    feature_vector = np.asarray(feature_vector, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    # Prevent division by zero
    std[std == 0] = 1e-8

    z = (feature_vector - mean) / std

    # Clip to ±3 sigma
    z = np.clip(z, -3, 3)

    return z


# ============================================================
# Quantization
# ============================================================

def quantize_vector(z_vector):
    """
    Convert normalized values to uint8 (0-255)

    Returns
    -------
    ndarray(uint8)
    """

    q = ((z_vector + 3) / 6.0) * 255

    q = np.round(q)

    q = np.clip(q, 0, 255)

    return q.astype(np.uint8)


# ============================================================
# SHA256 Fingerprint
# ============================================================

def generate_hash(
        quantized_vector,
        device_id="DEVICE001",
        date_bucket="2026-07"):
    """
    Generate SHA256 acoustic fingerprint.

    Hash = SHA256(
        quantized_vector ||
        device_id ||
        date_bucket
    )
    """

    byte_stream = (
        quantized_vector.tobytes() +
        device_id.encode() +
        date_bucket.encode()
    )

    fingerprint = hashlib.sha256(byte_stream).hexdigest()

    return fingerprint


# ============================================================
# Complete Fingerprint Pipeline
# ============================================================

def acoustic_fingerprint(
        feature_vector,
        mean,
        std,
        device_id="DEVICE001",
        date_bucket="2026-07"):
    """
    Returns

    fingerprint
    normalized vector
    quantized vector
    """

    z = normalize_vector(
        feature_vector,
        mean,
        std
    )

    q = quantize_vector(z)

    fingerprint = generate_hash(
        q,
        device_id,
        date_bucket
    )

    return fingerprint, z, q


# ============================================================
# Cosine Similarity
# ============================================================

def compute_similarity(
        reference_vector,
        current_vector):
    """
    Cosine similarity between two feature vectors.
    """

    similarity = cosine_similarity(
        [reference_vector],
        [current_vector]
    )[0][0]

    return float(similarity)


# ============================================================
# Drift Detection
# ============================================================

def detect_environment(similarity):
    """
    Decide environment status.
    """

    if similarity >= 0.92:
        return "Stable Environment"

    elif similarity >= 0.80:
        return "Minor Drift"

    elif similarity >= 0.65:
        return "Significant Drift"

    else:
        return "Relocated / Fraud Suspected"


# ============================================================
# EMA Baseline Update
# ============================================================

def update_baseline(
        old_vector,
        new_vector,
        beta=0.95):
    """
    Exponential Moving Average

    baseline_new =
        beta * baseline_old
      + (1-beta) * current
    """

    old_vector = np.asarray(old_vector)

    new_vector = np.asarray(new_vector)

    baseline = (
        beta * old_vector +
        (1 - beta) * new_vector
    )

    return baseline