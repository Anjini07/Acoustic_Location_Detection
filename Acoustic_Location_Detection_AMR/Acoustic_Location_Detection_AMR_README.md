# Acoustic_Location_Detection_AMR

**Project status: Discontinued (feasibility not established).** See
`Acoustic_Location_Detection_AMR_Final_Report.docx` for the full writeup.
Short version: AMR-NB's narrowband filtering (~300–3400Hz) and voice-activity
detection strip out the frequency content this approach depends on to tell
locations apart. On 6 fresh, held-out-from-training test recordings, only
2/6 were correctly and confidently matched — including 0/3 fresh recordings
of `office`, the location with the most training data (306 clips). This
codebase is preserved for reference in case a future hardware change (wider
audio capture) makes revisiting this approach worthwhile.

## What this project is

A fully separate, self-contained adaptation of the WAV-based
`Acoustic_Location_Detection` project for real AMR audio recorded by the
soundbox hardware. Nothing here is imported from the WAV project at runtime
except through `predict_incoming_router.py`, by design.

## The one core algorithmic change

`feature_extraction.py` restricts the Mel filterbank to `fmin=300, fmax=3400`
(AMR-NB's real narrowband passband) instead of the WAV project's `0-8000Hz`.
Every other step (STFT params, MFCC count, the 104-dim stat vector, z-score
normalize, KMeans multi-baseline, cosine similarity) is identical to the WAV
project's math.

## Pipeline (run in this order)

```
python chunk_audio.py              # raw_recordings_amr/*  -> dataset/<location>/*.wav (5s clips)
python compute_global_stats.py     # dataset/               -> global_stats.json
python build_multi_baselines.py    # dataset/ + global_stats -> multi_baselines.json (KMeans, up to 3 sub-baselines/location)
python calibrate_thresholds.py     # dataset/ + multi_baselines -> tries several percentile pairs, recommends the widest valid one
# [manual] paste the recommended pair into drift_detection.py's classify_similarity() defaults
python verify_test_audio.py        # 20%-per-location split of raw_recordings_amr/ (auto leakage-checked) -> verification_report.json
python predict_incoming.py         # incoming/*             -> incoming_predictions.json (ledger-tracked, won't re-predict)
```

Or score one arbitrary file directly:
```
python predict_incoming_router.py path/to/some_recording.amr
```

Diagnostics:
```
python diagnose_calibration.py               # per-location score breakdown + worst outlier clips
python plot_similarity_matrix.py verification # heatmap of verify_test_audio.py results
python plot_similarity_matrix.py incoming     # heatmap of new (unplotted) predict_incoming.py results; --all for full history
```

## Final calibrated thresholds (4 locations: cafeteria, metro, nature, office)

From the 25th/75th percentile pair (the narrowest valid pair found — 10th/90th
through 20th/80th were all inverted, itself a sign of heavy same-vs-different
location score overlap):

```
STABLE     if score > 0.4829
MODERATE   if 0.3788 < score <= 0.4829
RELOCATION if score <= 0.3788
```

## Final results summary

- **Held-out split (20% of raw recordings, auto-generated)**: 11/13 scored
  files matched — but 14 of the 19 held-out files were found to already be
  part of the training data (leakage), so this number is optimistic, not
  reliable. See the Final Report, Section 6.2(a) and Appendix A, for the
  full per-file breakdown and caveat.
- **Fresh, independently recorded test files (clean, no leakage)**: 2/6
  correctly and confidently matched. `metro_test` was misidentified as
  cafeteria; two of three fresh `office` recordings were misidentified as
  `nature`, and the third ranked office correctly but scored low enough to
  be flagged `RELOCATION_OR_FRAUD` anyway. See the Final Report, Section
  6.2(b), for the full table and the heatmap.

## Setup (if revisiting)

1. Copy `ffmpeg.exe` + `ffprobe.exe` into `ffmpeg/bin/` (needed for AMR
   decode/encode via `audio_preprocessing.py` / `simulate_amr_roundtrip.py`).
2. Lay out real AMR recordings as `raw_recordings_amr/<location>/*.amr`.
3. For a location without real AMR yet, bootstrap it with a genuine AMR-NB
   encode of existing WAV recordings:
   ```
   python simulate_amr_roundtrip.py --input_dir <wav_source_dir> --output_dir raw_recordings_amr/<location>
   ```
   (Keep round-tripped files out of anything meant to be a clean held-out
   test set — they approximate real AMR degradation closely, but aren't
   the genuine device-recorded thing.)

## File map

| File | Role |
|---|---|
| `audio_preprocessing.py` | Format-agnostic WAV/AMR loader + DC removal/pre-emphasis/normalize |
| `feature_extraction.py` | STFT -> Mel (300-3400Hz) -> MFCC -> 104-dim stat vector |
| `chunk_audio.py` | Raw recordings -> 5s clips (silence-skip, merge-seam crossfade, provenance tracking) |
| `compute_global_stats.py` | Dataset-wide mean/std for z-score normalization |
| `build_multi_baselines.py` | Per-location KMeans sub-baselines (up to 3: quiet/moderate/busy) |
| `calibrate_thresholds.py` | Tries multiple percentile pairs, recommends the widest one where STABLE > MODERATE actually holds |
| `diagnose_calibration.py` | Per-location score breakdown + worst same-location/cross-location outlier clips |
| `drift_detection.py` | Normalize, cosine similarity, classify_similarity (final thresholds above), EMA baseline update (unused) |
| `verify_test_audio.py` | Auto 20%-per-location split of raw_recordings_amr/ (persisted in test_split.json), leakage-checked against clip_provenance.json |
| `predict_incoming.py` | Scores new files in `incoming/`, ledger-tracked (predicted_files.json) so nothing is re-scored |
| `simulate_amr_roundtrip.py` | Bootstraps missing-location AMR data via real AMR-NB encode of existing WAV |
| `predict_incoming_router.py` | Shared entry point — routes a single file to the WAV or AMR pipeline by extension |
| `plot_similarity_matrix.py` | Heatmap visualizations for both verification and incoming results |

## Folders

- `raw_recordings_amr/<location>/` — input, real + round-tripped AMR files
- `dataset/<location>/` — output of `chunk_audio.py`, 5s WAV clips (transient PCM, never persisted as AMR)
- `incoming/` — drop files here for `predict_incoming.py`; processed files move to `incoming/_processed/` (best-effort tidying — the actual "don't re-predict" guarantee is `predicted_files.json`)
- `ffmpeg/bin/` — put `ffmpeg.exe` + `ffprobe.exe` here (not included)
