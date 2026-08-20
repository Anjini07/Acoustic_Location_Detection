import numpy as np

def compute_baseline(vectors: np.ndarray) -> np.ndarray:
    """Average feature vector across a set of clips (e.g. part1-3) for one location."""
    return np.mean(vectors, axis=0)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))