import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

DATASET_DIR = "Acoustic_Location_Detection/dataset"
LATEST_RUN_FILE = "Acoustic_Location_Detection/incoming_latest_run.json"


def get_locations():
    """Reads location names from the dataset folder instead of hardcoding
    them, so adding/removing a location doesn't require editing this file."""
    return sorted([
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    ])


def plot_from_verification_report(report_path, output_path):
    """For verify_test_audio.py output — known test files with known
    expected locations."""
    with open(report_path) as f:
        report = json.load(f)

    locations = get_locations()
    files = list(report.keys())

    matrix = np.array([[report[f]["all_scores"][loc] for loc in locations] for f in files])
    _draw_matrix(matrix, locations, files, "Test audio vs location baselines", output_path)


def plot_from_incoming_predictions(predictions_path, output_path, latest_only=True):
    """For predict_incoming.py output — arbitrary files of any length,
    scored as an average across their 10s windows.

    latest_only=True (default): only plots files from the most recent
    predict_incoming.py run (per incoming_latest_run.json), since
    incoming_predictions.json accumulates across every run and re-plotting
    the whole history each time isn't usually what you want. Pass
    latest_only=False (or use the 'incoming-all' CLI mode) to see everything
    ever predicted.
    """
    with open(predictions_path) as f:
        results = json.load(f)

    if latest_only:
        if os.path.exists(LATEST_RUN_FILE):
            with open(LATEST_RUN_FILE) as f:
                latest_files = set(json.load(f))
            results = {f: r for f, r in results.items() if f in latest_files}
            if not results:
                print("The most recent predict_incoming.py run had no usable predictions to plot.")
                return
        else:
            print(f"[NOTE] {LATEST_RUN_FILE} not found -- plotting the full prediction "
                  f"history instead. Run predict_incoming.py to generate a latest-run list.")

    locations = get_locations()

    # Skip any file that was rejected for low quality — it has no
    # avg_scores to plot, only a status/reason
    usable_results = {
        f: r for f, r in results.items() if "avg_scores" in r
    }
    rejected = [f for f, r in results.items() if "avg_scores" not in r]

    if rejected:
        print(f"Skipping {len(rejected)} rejected file(s) from plot: {rejected}")

    if not usable_results:
        print("No usable predictions to plot — every file was rejected for low quality.")
        return

    files = list(usable_results.keys())
    matrix = np.array([
        [usable_results[f]["avg_scores"].get(loc, np.nan) for loc in locations] for f in files
    ])
    title = "Incoming audio vs location baselines (avg across 10s windows)"
    if latest_only:
        title += " — latest run"
    _draw_matrix(matrix, locations, files, title, output_path)

def _draw_matrix(matrix, locations, files, title, output_path):
    fig, ax = plt.subplots(figsize=(6, len(files) * 0.6 + 1.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=matrix.min(), vmax=matrix.max())

    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels(locations)
    ax.set_yticks(range(len(files)))
    ax.set_yticklabels(files)

    for i in range(len(files)):
        for j in range(len(locations)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")

        best_col = int(np.argmax(matrix[i]))
        ax.add_patch(plt.Rectangle(
            (best_col - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2
        ))

    plt.colorbar(im, label="Cosine similarity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verification"

    if mode == "verification":
        plot_from_verification_report(
            "Acoustic_Location_Detection/verification_report.json",
            "Acoustic_Location_Detection/similarity_matrix.png",
        )
    elif mode == "incoming":
        plot_from_incoming_predictions(
            "Acoustic_Location_Detection/incoming_predictions.json",
            "Acoustic_Location_Detection/incoming_similarity_matrix.png",
            latest_only=True,
        )
    elif mode == "incoming-all":
        plot_from_incoming_predictions(
            "Acoustic_Location_Detection/incoming_predictions.json",
            "Acoustic_Location_Detection/incoming_similarity_matrix_all.png",
            latest_only=False,
        )
    else:
        print("Usage: python plot_similarity_matrix.py [verification|incoming|incoming-all]")