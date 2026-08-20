import os
import joblib
import numpy as np
import pandas as pd

from src.fingerprint import acoustic_fingerprint

# =====================================================
# Load Feature Dataset
# =====================================================

FEATURE_FILE = "output/features.csv"

df = pd.read_csv(FEATURE_FILE)

# =====================================================
# Extract Features
# =====================================================

X = df.iloc[:, :104].values

# =====================================================
# Compute Dataset Statistics
# =====================================================

feature_mean = X.mean(axis=0)
feature_std = X.std(axis=0)

# =====================================================
# Save Mean & Std
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(feature_mean, "models/feature_mean.pkl")
joblib.dump(feature_std, "models/feature_std.pkl")

print("Feature statistics saved.")

# =====================================================
# Generate Fingerprints
# =====================================================

fingerprints = []

for i in range(len(df)):

    feature_vector = X[i]

    fingerprint, z, q = acoustic_fingerprint(
        feature_vector,
        feature_mean,
        feature_std,
        device_id="DEVICE001",
        date_bucket="2026-07"
    )

    fingerprints.append(fingerprint)

# =====================================================
# Create Fingerprint Database
# =====================================================

fingerprint_df = pd.DataFrame({

    "Location": df["Location"],
    "Recording": df["Recording"],
    "Filename": df["Filename"],
    "Fingerprint": fingerprints

})

# =====================================================
# Save Database
# =====================================================

fingerprint_df.to_csv(
    "output/fingerprint_database.csv",
    index=False
)

print("\nFingerprint Database Created Successfully\n")

print(fingerprint_df.head())