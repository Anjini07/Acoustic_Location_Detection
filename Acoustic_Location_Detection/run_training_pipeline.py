"""
run_training_pipeline.py -- runs every training-side stage in order, as one
command:

    chunk_audio.py -> compute_global_stats.py -> build_multi_baselines.py
    -> calibrate_thresholds.py -> verify_test_audio.py

This is step 2 of the 3-step workflow:
    1. Drop raw recordings into raw_recordings/<location>/         (manual)
    2. python Acoustic_Location_Detection/run_training_pipeline.py (this script)
    3. python Acoustic_Location_Detection/predict_incoming.py      (unchanged, separate)

Each stage runs as its own subprocess (not imported directly) so every
script keeps working standalone exactly as before -- this is a thin
orchestration layer on top, not a rewrite. If any stage fails, the pipeline
stops immediately rather than continuing on to build baselines from a
half-finished chunking step, etc.

Run from the same parent directory every other script expects:
    python Acoustic_Location_Detection/run_training_pipeline.py
"""

import os
import sys
import time
import argparse
import subprocess

SCRIPT_DIR = "Acoustic_Location_Detection"


def _run_stage(name, args_list, description):
    print(f"\n{'='*70}")
    print(f"STAGE: {name}")
    print(f"  {description}")
    print(f"  $ {' '.join(args_list)}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(args_list)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n{'!'*70}")
        print(f"STOPPED: '{name}' exited with an error (code {result.returncode}), "
              f"after {elapsed:.1f}s.")
        print(f"Nothing after this stage ran. Fix the error above and re-run "
              f"this script -- earlier stages don't need to be redone; "
              f"chunk_audio.py's ledger and each stage's own file outputs "
              f"mean re-running from the top is cheap, not wasted work.")
        print(f"{'!'*70}")
        sys.exit(result.returncode)

    print(f"\n  OK — {name} finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Runs the full training pipeline (chunking through verification) in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clip-seconds", type=int, default=None,
                         help="Forwarded to chunk_audio.py (default: that script's own default).")
    parser.add_argument("--skip-verify", action="store_true",
                         help="Skip the verify_test_audio.py stage (e.g. for a quick re-run "
                              "after adding a small amount of new data).")
    parser.add_argument("--stable-percentile", type=float, default=None,
                         help="Forwarded to calibrate_thresholds.py.")
    parser.add_argument("--moderate-percentile", type=float, default=None,
                         help="Forwarded to calibrate_thresholds.py.")
    parser.add_argument("--override-stable", type=float, default=None,
                         help="Forwarded to calibrate_thresholds.py -- manually set stable_threshold.")
    parser.add_argument("--override-moderate", type=float, default=None,
                         help="Forwarded to calibrate_thresholds.py -- manually set moderate_threshold.")
    args = parser.parse_args()

    py = sys.executable  # use the same interpreter/venv this script was launched with

    overall_start = time.time()

    chunk_args = [py, os.path.join(SCRIPT_DIR, "chunk_audio.py")]
    if args.clip_seconds is not None:
        chunk_args += ["--clip-seconds", str(args.clip_seconds)]
    _run_stage(
        "1/5 chunk_audio.py", chunk_args,
        "Splits any new raw recordings into fixed-length clips (skips files already chunked).",
    )

    _run_stage(
        "2/5 compute_global_stats.py", [py, os.path.join(SCRIPT_DIR, "compute_global_stats.py")],
        "Recomputes dataset-wide mean/std used to normalize feature vectors.",
    )

    _run_stage(
        "3/5 build_multi_baselines.py", [py, os.path.join(SCRIPT_DIR, "build_multi_baselines.py")],
        "Rebuilds per-location sub-baselines (KMeans clusters) from the current dataset.",
    )

    calib_args = [py, os.path.join(SCRIPT_DIR, "calibrate_thresholds.py")]
    if args.stable_percentile is not None:
        calib_args += ["--stable-percentile", str(args.stable_percentile)]
    if args.moderate_percentile is not None:
        calib_args += ["--moderate-percentile", str(args.moderate_percentile)]
    if args.override_stable is not None:
        calib_args += ["--override-stable", str(args.override_stable)]
    if args.override_moderate is not None:
        calib_args += ["--override-moderate", str(args.override_moderate)]
    _run_stage(
        "4/5 calibrate_thresholds.py", calib_args,
        "Recomputes and saves STABLE/MODERATE/RELOCATION thresholds from the current baselines.",
    )

    if args.skip_verify:
        print(f"\n{'='*70}")
        print("STAGE: 5/5 verify_test_audio.py -- SKIPPED (--skip-verify passed)")
        print(f"{'='*70}")
    else:
        _run_stage(
            "5/5 verify_test_audio.py", [py, os.path.join(SCRIPT_DIR, "verify_test_audio.py")],
            "Checks held-out accuracy against the freshly built baselines.",
        )

    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"TRAINING PIPELINE COMPLETE in {total:.1f}s.")
    print(f"Next: python {os.path.join(SCRIPT_DIR, 'predict_incoming.py')} to score incoming/ audio.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
