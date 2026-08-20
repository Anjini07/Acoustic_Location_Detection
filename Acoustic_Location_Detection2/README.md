# Streaming Pipeline (AMR-native, near-zero extra storage)

## The core idea

You can't get chunk files back into AMR (standard ffmpeg builds have no AMR
*encoder* — it's patent-encumbered, decode-only is normal), and writing
chunks out as WAV/FLAC instead costs real disk space for something you only
need briefly: the numeric features.

So this pipeline never writes chunk audio to disk at all. It decodes each
raw file once, slices it into 5s windows in memory, extracts 9 lightweight
features per window, and discards the audio immediately. Only a small CSV
survives. Your original `.amr` files are never touched or duplicated — they
remain the only audio stored anywhere.

**Storage, for ~1000 five-second windows:**
| Approach | Size |
|---|---|
| WAV chunks | ~160 MB |
| FLAC chunks | ~70 MB |
| This pipeline (CSV only) | ~200 KB |

## Setup

```bash
pip install pydub scikit-learn pandas joblib numpy soundfile librosa noisereduce --break-system-packages
```

`pydub` needs `ffmpeg` on PATH or pointed to directly (see `audio_preprocessing.py` —
it already looks for `./ffmpeg/bin/ffmpeg.exe` relative to itself).

## Files

| File | Purpose |
|---|---|
| `audio_preprocessing.py` | Unified WAV/AMR loader (`load_audio_any_format`) + `preprocess_audio` (DC removal, pre-emphasis, normalize) |
| `feature_extraction.py` | 9 lightweight features per window: noise floor RMS, speech RMS, SNR estimate, spectral flatness, spectral centroid, ZCR mean/std, energy dynamic range, frequency-band energy ratio |
| `build_features_streaming.py` | **The core piece.** Decodes → windows (splits long files, merges short ones) → extracts features → discards audio, all in one pass. Writes only a features CSV. |
| `cluster_features.py` | Unsupervised KMeans label discovery, run directly on the CSV — no files to sort, since none exist |
| `calibrate_thresholds.py` | Zero-model rule-based fallback (percentile thresholds on `noise_floor_rms`) |
| `train_classifier.py` | Trains Logistic Regression or shallow Decision Tree on the CSV |
| `predict_environment.py` | Predicts on a new incoming file — already storage-optimal, does its own in-memory windowing |
| `extract_clip_for_listening.py` | **On-demand spot-check only.** Decodes one named raw file into a throwaway WAV so you can listen to it — used only for the few files you actually want to verify by ear, not persisted up front for everything |

## Run order

**If you already know the labels** (recordings sorted into `quiet/moderate/noisy` folders):
```bash
python3 build_features_streaming.py --input_dir raw_recordings --output_csv features_labeled.csv --location_as_label
python3 calibrate_thresholds.py --csv_path features_labeled.csv --output_json env_thresholds.json
python3 train_classifier.py --csv_path features_labeled.csv --model_type logreg --output_model env_classifier.joblib
python3 predict_environment.py path/to/incoming.amr --model_path env_classifier.joblib --thresholds_path env_thresholds.json
```

**If you don't know the labels yet** (recordings unsorted, e.g. mostly the same place):
```bash
python3 build_features_streaming.py --input_dir raw_recordings_unlabeled --output_csv features.csv --flat
python3 cluster_features.py --csv_path features.csv --output_csv features_labeled.csv --k 2
python3 calibrate_thresholds.py --csv_path features_labeled.csv --output_json env_thresholds.json
python3 train_classifier.py --csv_path features_labeled.csv --model_type logreg --output_model env_classifier.joblib
python3 predict_environment.py path/to/incoming.amr --model_path env_classifier.joblib --thresholds_path env_thresholds.json
```

`cluster_features.py` prints the most "typical" example file per cluster —
spot-check those with `extract_clip_for_listening.py <that file>` before
trusting the labels.

## What's different from the old chunk_audio.py + build_dataset.py

- No `dataset/<location>/*.wav` folder full of chunk files ever gets created
- `source_files` column in the CSV records exactly which raw file(s) contributed
  to each window (important for windows built from merged short recordings)
- `--location_as_label` lets you skip clustering entirely if your folders are
  already `quiet/moderate/noisy` — the old two-step (chunk, then build_dataset)
  collapses into one streaming pass either way

## Known tradeoffs, stated plainly

- Merged windows (from short recordings) still stitch together audio from
  more than one real recording session — same caveat as before, just tracked
  more precisely now via `source_files`
- Spot-checking a merged window previews the *whole* contributing file(s),
  not the exact sub-second slice — simpler and more robust than trying to
  reconstruct exact byte offsets, at the cost of imprecision for spot-checks
- Any leftover audio at the very end of a location's file list, shorter than
  5s with nothing left to merge into, is dropped
