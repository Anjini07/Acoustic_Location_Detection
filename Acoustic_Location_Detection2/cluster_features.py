"""
cluster_features.py
Unsupervised label discovery -- operates directly on the features CSV
produced by build_features_streaming.py. There are no per-window audio files
to sort into folders (none were ever written), so this just adds a 'label'
column to the CSV in place of the old file-copying approach.

Same clustering logic as before: KMeans on the 9 lightweight features,
clusters ordered by mean noise_floor_rms and mapped to quiet/moderate/noisy
(or quiet/noisy for k=2), so labels are meaningful rather than arbitrary
cluster numbers.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from feature_extraction import FEATURE_NAMES

LABELS_BY_K = {
    2: ["quiet", "noisy"],
    3: ["quiet", "moderate", "noisy"],
}


def cluster_and_label(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k not in LABELS_BY_K:
        raise ValueError(f"k must be 2 or 3, got {k}")
    if len(df) < k:
        raise ValueError(f"Only {len(df)} rows available, need at least {k} for k={k}")

    X = df[FEATURE_NAMES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["cluster_id"] = cluster_ids

    if len(df) > k:
        score = silhouette_score(X_scaled, cluster_ids)
        print(f"Silhouette score (closer to 1 = better separated clusters): {score:.3f}")

    cluster_order = (
        df.groupby("cluster_id")["noise_floor_rms"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    label_map = {c: LABELS_BY_K[k][i] for i, c in enumerate(cluster_order)}
    df["label"] = df["cluster_id"].map(label_map)

    centroids = kmeans.cluster_centers_
    df["distance_to_centroid"] = np.linalg.norm(X_scaled - centroids[cluster_ids], axis=1)

    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Cluster summary (mean feature values per label) ===")
    summary = df.groupby("label")[FEATURE_NAMES].mean()
    summary["count"] = df.groupby("label").size()
    print(summary.to_string())

    print("\n=== Most typical example per cluster ===")
    print("(spot-check with: python3 extract_clip_for_listening.py <source_files>)")
    for label, group in df.groupby("label"):
        typical = group.sort_values("distance_to_centroid").iloc[0]
        loc = typical["location"] if "location" in typical else "?"
        print(f"  {label:10s} -> {typical['source_files']}  (location={loc}, "
              f"noise_floor_rms={typical['noise_floor_rms']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised label discovery on a features CSV")
    parser.add_argument("--csv_path", default="Acoustic_Location_Detection2/features.csv")
    parser.add_argument("--output_csv", default="Acoustic_Location_Detection2/features_labeled.csv")
    parser.add_argument("--k", type=int, default=2, choices=[2, 3])
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = cluster_and_label(df, args.k)
    summarize(df)

    df.to_csv(args.output_csv, index=False)
    print(f"\nWrote labeled features to {args.output_csv}")
