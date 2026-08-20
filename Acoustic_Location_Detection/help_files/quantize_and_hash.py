import hashlib
import time
import numpy as np

SIGMA_CLIP = 3.0


def quantize_vector(E, global_mean, global_std):
    
    E = np.array(E, dtype=np.float32)
    global_mean = np.array(global_mean, dtype=np.float32)
    global_std = np.array(global_std, dtype=np.float32)

    # Z-score normalization
    E_norm = (E - global_mean) / global_std

    # Clip to +/- 3 sigma
    E_clip = np.clip(E_norm, -SIGMA_CLIP, SIGMA_CLIP)

    # Scale to [0, 255] and convert to uint8
    E_quant = np.round(
        (E_clip + SIGMA_CLIP) / (2 * SIGMA_CLIP) * 255
    ).astype(np.uint8)

    return E_quant


def compute_acoustic_hash(E_quant, device_id, date_bucket=None):
    """
    Builds the final SHA-256 acoustic_hash.
    payload = quantized_bytes || device_id || date_bucket
    """
    if date_bucket is None:
        date_bucket = str(int(time.time()) // 86400) 

    payload = E_quant.tobytes() + device_id.encode() + date_bucket.encode()
    hex_hash = hashlib.sha256(payload).hexdigest()

    return hex_hash, date_bucket