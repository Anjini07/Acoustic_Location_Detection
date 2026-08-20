import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from src.audio_preprocessing import preprocess_audio
from src.feature_extraction import extract_features
from src.fingerprint import acoustic_fingerprint, detect_environment

# ==========================================================
# Load Saved Files
# ==========================================================

print("Loading Model...")

model = joblib.load("models/randomforest.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_mean = joblib.load("models/feature_mean.pkl")
feature_std = joblib.load("models/feature_std.pkl")

print("Model Loaded Successfully")

# ==========================================================
# Load Reference Dataset
# ==========================================================

feature_df = pd.read_csv("output/features.csv")
fingerprint_df = pd.read_csv("output/fingerprint_database.csv")

reference_features = feature_df.iloc[:, :104].values

# ==========================================================
# Predict Every Test Audio
# ==========================================================

TEST_FOLDER = "test_audio"

for filename in os.listdir(TEST_FOLDER):

    if not filename.lower().endswith(".wav"):
        continue

    print("\n===================================================")
    print("File:", filename)

    filepath = os.path.join(TEST_FOLDER, filename)

    # ------------------------------------------------------
    # Step 1 : Preprocess
    # ------------------------------------------------------

    audio, sr = preprocess_audio(filepath)

    # ------------------------------------------------------
    # Step 2 : Feature Extraction
    # ------------------------------------------------------

    features = extract_features(audio, sr)

    feature_vector = np.array(features).reshape(1, -1)

    # ------------------------------------------------------
    # Step 3 : Scale
    # ------------------------------------------------------

    feature_scaled = scaler.transform(feature_vector)

    # ------------------------------------------------------
    # Step 4 : Prediction
    # ------------------------------------------------------

    prediction = model.predict(feature_scaled)[0]

    confidence = np.max(
        model.predict_proba(feature_scaled)
    )

    # ------------------------------------------------------
    # Step 5 : Fingerprint
    # ------------------------------------------------------

    fingerprint, z, q = acoustic_fingerprint(
        np.array(features),
        feature_mean,
        feature_std
    )

    # ------------------------------------------------------
    # Step 6 : Cosine Similarity
    # ------------------------------------------------------

    similarity_scores = cosine_similarity(
        feature_vector,
        reference_features
    )[0]

    best_index = np.argmax(similarity_scores)

    best_similarity = similarity_scores[best_index]

    matched_location = fingerprint_df.iloc[best_index]["Location"]

    matched_file = fingerprint_df.iloc[best_index]["Filename"]

    # ------------------------------------------------------
    # Step 7 : Drift Detection
    # ------------------------------------------------------

    status = detect_environment(best_similarity)

    # ------------------------------------------------------
    # Print Results
    # ------------------------------------------------------

    print("Predicted Location :", prediction)
    print("Confidence         :", round(confidence*100,2), "%")

    print("Matched Location   :", matched_location)
    print("Matched File       :", matched_file)

    print("Similarity         :", round(best_similarity,4))

    print("Environment Status :", status)

    print("Fingerprint")

    print(fingerprint)

print("\nDone.")