"""
train_classifier.py
Trains a lightweight ML classifier (Logistic Regression by default) on the
features CSV -- whether it came from build_features_streaming.py directly
(--location_as_label) or from cluster_features.py's discovered labels.

Uses a GROUPED train/test split, not a plain random one. Consecutive 5s
windows from the same source recording are highly correlated -- a plain
random split lets windows from the same file land on both sides, which
inflates the reported accuracy (the same leakage problem documented in the
original Acoustic_Location_Detection report, Section 4). Windows are grouped
by which raw file(s) contributed to them (via the 'source_files' column),
using union-find so that even a chain of merged windows -- e.g. window A
merges files 1+2, window B merges files 2+3 -- stays together as one group,
since 1, 2, and 3 all ultimately share audio.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from feature_extraction import FEATURE_NAMES as FEATURE_COLUMNS


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def _assign_groups(df: pd.DataFrame) -> np.ndarray:
    """Groups every window by which raw source file(s) it came from, merging
    groups transitively so no two windows that share even one source file --
    directly or through a chain of merges -- end up in different groups."""
    uf = _UnionFind()

    for source_files in df["source_files"]:
        files = str(source_files).split("+")
        for f in files[1:]:
            uf.union(files[0], f)

    return df["source_files"].apply(lambda s: uf.find(str(s).split("+")[0])).values


def _grouped_split(df: pd.DataFrame, groups: np.ndarray, test_size: float = 0.25, random_state: int = 42):
    """Grouped + stratified split where possible. Falls back to a grouped
    (non-stratified) split on older scikit-learn versions that lack
    StratifiedGroupKFold, or when there aren't enough groups for a clean
    stratified split -- preventing leakage matters more than preserving
    exact class ratios in the split."""
    y = df["label"].values
    n_groups = len(set(groups))

    if n_groups < 2:
        raise ValueError(
            f"Only {n_groups} distinct source recording group(s) in this dataset -- "
            "a train/test split isn't possible without leaking the same recording "
            "across both sides. You need data from at least 2 separate sessions/files."
        )

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        n_splits = max(2, min(round(1 / test_size), n_groups))
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(skf.split(df, y, groups=groups))
    except ImportError:
        from sklearn.model_selection import GroupShuffleSplit
        print("(scikit-learn version doesn't have StratifiedGroupKFold -- "
              "falling back to a grouped, non-stratified split)")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, y, groups=groups))
    except ValueError as e:
        from sklearn.model_selection import GroupShuffleSplit
        print(f"(StratifiedGroupKFold couldn't split cleanly with only {n_groups} group(s) -- {e})")
        print("(falling back to a grouped, non-stratified split -- fine for preventing leakage, "
              "just won't guarantee balanced class ratios between train and test)")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, y, groups=groups))

    return train_idx, test_idx


def train(csv_path: str, model_type: str, output_model: str) -> None:
    df = pd.read_csv(csv_path)

    if "source_files" not in df.columns:
        raise ValueError(
            "CSV has no 'source_files' column -- can't group by source recording to "
            "avoid leakage. Make sure this CSV came from build_features_streaming.py."
        )

    groups = _assign_groups(df)
    n_groups = len(set(groups))
    print(f"{len(df)} windows grouped into {n_groups} distinct source recording group(s)")

    if n_groups < 5:
        print(f"WARNING: only {n_groups} distinct source recording(s) total. Even with grouping "
              "fixed, the split has very few ways to divide this data -- one file's worth of audio "
              "can dominate either train or test. Treat any resulting score as a rough signal, not "
              "a trustworthy accuracy number, until you have recordings from more distinct sessions.")

    groups_per_class = df.assign(_group=groups).groupby("label")["_group"].nunique()
    print("\nSource recording groups per class:")
    print(groups_per_class.to_string())
    thin_classes = groups_per_class[groups_per_class < 2]
    if not thin_classes.empty:
        print(f"\nWARNING: {list(thin_classes.index)} come from only 1 source recording group each. "
              "A grouped split has no choice but to put that entire class in train OR test, not both -- "
              "so you'll get 0% or trivial performance on that class this run, not a real measurement of it. "
              "This isn't a bug -- it means you need recordings from more than one distinct session/file for "
              "that class before a trustworthy held-out score is even possible.")

    train_idx, test_idx = _grouped_split(df, groups)

    # sanity check: confirm no group appears on both sides
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise RuntimeError(f"Grouping failed -- {len(overlap)} group(s) leaked across the split")

    X = df[FEATURE_COLUMNS]
    y = df["label"]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {len(X_train)} windows | Test: {len(X_test)} windows "
          f"(no source recording appears in both)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "tree":
        model = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
    else:
        raise ValueError("model_type must be 'logreg' or 'tree'")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\n=== Classification report (held-out, grouped split) ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}, output_model)
    print(f"\nSaved model + scaler to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train environment-type classifier")
    parser.add_argument("--csv_path", default="Acoustic_Location_Detection2/features_labeled.csv")
    parser.add_argument("--model_type", default="logreg", choices=["logreg", "tree"])
    parser.add_argument("--output_model", default="Acoustic_Location_Detection2/env_classifier.joblib")
    args = parser.parse_args()

    train(args.csv_path, args.model_type, args.output_model)