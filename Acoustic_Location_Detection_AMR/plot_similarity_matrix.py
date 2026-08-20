import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

DATASET_DIR = "Acoustic_Location_Detection_AMR/dataset"
PLOTTED_LEDGER_FILE = "Acoustic_Location_Detection_AMR/plotted_incoming_files.json"


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


def _load_plotted_ledger():
    if os.path.exists(PLOTTED_LEDGER_FILE):
        with open(PLOTTED_LEDGER_FILE) as f:
            return set(json.load(f))
    return set()


def _save_plotted_ledger(plotted):
    with open(PLOTTED_LEDGER_FILE, "w") as f:
        json.dump(sorted(plotted), f, indent=2)


def plot_from_incoming_predictions(predictions_path, output_path, show_all=False):
    """For predict_incoming.py output — arbitrary files of any length,
    scored as an average across their 5s windows.

    predict_incoming.py merges results into incoming_predictions.json across
    every run instead of overwriting it, so by default this only plots files
    that haven't been plotted before (tracked in plotted_incoming_files.json)
    -- otherwise every run's plot would keep growing to include every
    incoming file ever predicted. Pass --all to see the full history."""
    with open(predictions_path) as f:
        results = json.load(f)

    locations = get_locations()

    # Skip any file that was rejected for low quality — it has no
    # avg_scores to plot, only a status/reason
    usable_results = {
        f: r for f, r in results.items() if "avg_scores" in r
    }
    rejected = [f for f, r in results.items() if "avg_scores" not in r]

    if rejected:
        print(f"Skipping {len(rejected)} rejected file(s) from plot: {rejected}")

    plotted = set() if show_all else _load_plotted_ledger()
    new_results = {f: r for f, r in usable_results.items() if f not in plotted}

    if not usable_results:
        print("No usable predictions to plot — every file was rejected for low quality.")
        return

    if not new_results:
        print("No new incoming files since the last plot (everything here was already "
              "plotted before). Pass --all to see the full history instead.")
        return

    if not show_all:
        already_plotted_count = len(usable_results) - len(new_results)
        if already_plotted_count:
            print(f"Plotting {len(new_results)} new file(s), skipping {already_plotted_count} "
                  f"already-plotted file(s) (pass --all to include them).")

    target_results = usable_results if show_all else new_results
    files = list(target_results.keys())
    matrix = np.array([
        [target_results[f]["avg_scores"].get(loc, np.nan) for loc in locations] for f in files
    ])
    title = "Incoming audio vs location baselines (avg across 5s windows)"
    if not show_all:
        title += " — new since last plot"
    _draw_matrix(matrix, locations, files, title, output_path)

    if not show_all:
        # Only the newly-plotted files get marked as seen -- if show_all was
        # used this run, the ledger is left untouched so a later default
        # (non---all) run still only shows whatever's genuinely new.
        plotted.update(new_results.keys())
        _save_plotted_ledger(plotted)


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
    args = sys.argv[1:]
    show_all = "--all" in args
    args = [a for a in args if a != "--all"]
    mode = args[0] if args else "verification"

    if mode == "verification":
        plot_from_verification_report(
            "Acoustic_Location_Detection_AMR/verification_report.json",
            "Acoustic_Location_Detection_AMR/similarity_matrix.png",
        )
    elif mode == "incoming":
        plot_from_incoming_predictions(
            "Acoustic_Location_Detection_AMR/incoming_predictions.json",
            "Acoustic_Location_Detection_AMR/incoming_similarity_matrix.png",
            show_all=show_all,
        )
    else:
        print("Usage: python plot_similarity_matrix.py [verification|incoming] [--all]")