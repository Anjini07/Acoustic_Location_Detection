# Acoustic Location Detection — README

**Project:** ToneTag Acoustic Hash / physical-location verification, WAV pipeline
**Owner:** Anjini Bose, ToneTag Embedded Team (internship project)
**Status:** Active. Working end-to-end pipeline, currently being expanded with YouTube-sourced
training/test audio per team lead's direction. A separate feasibility study on real AMR
(soundbox) hardware audio was run and concluded — see §11.

This document is meant to let anyone — including a future version of you — pick this project
back up without re-deriving any of the decisions below from scratch.

---

## 1. What this project does

Given a short audio recording, identify **which known physical location** it was recorded in
(e.g. office, cafeteria, metro, traffic, nature), by comparing an acoustic "fingerprint" of the
clip against a per-location baseline built from many prior recordings of that location. This is
one input into ToneTag's Merchant Credit Score — verifying a merchant's device is physically
where it claims to be, using ambient sound rather than GPS.

The core technique: turn audio into a fixed-length statistical feature vector (loosely, a
compressed description of its frequency content over time), compare that vector to each
location's stored baseline vectors via cosine similarity, and classify the result as a confident
match, a borderline/drifted match, or "doesn't match anything known."

## 2. Repository layout

```
Acoustic_Location_Detection/
├── audio_preprocessing.py     # load/normalize audio, list_audio_files() helper, TARGET_SR
├── feature_extraction.py      # turns raw audio into the 104-dim feature vector
├── chunk_audio.py             # splits raw recordings into fixed-length clips (ledgered)
├── compute_global_stats.py    # dataset-wide mean/std used to z-score normalize features
├── build_multi_baselines.py   # KMeans sub-baselines per location (quiet/moderate/busy)
├── calibrate_thresholds.py    # computes STABLE/MODERATE/RELOCATION score thresholds
├── drift_detection.py         # similarity math + classify_similarity() threshold logic
├── verify_test_audio.py       # held-out accuracy check, with train/test leakage warning
├── predict_incoming.py        # scores new/incoming audio against the baselines
├── plot_similarity_matrix.py  # heatmap visualizations of scores
├── yt_download.py             # pulls training/test audio from YouTube (see §7)
├── raw_recordings/<location>/ # source recordings, one folder per location
├── raw_recordings/metro/, home/   # phone-recorded, not YouTube-sourced (see §7)
├── dataset/<location>/        # chunked clips, built from raw_recordings/ by chunk_audio.py
├── incoming/                  # audio waiting to be scored by predict_incoming.py
├── incoming/_processed/       # incoming files already scored, kept out of re-processing
├── incoming/_chunks/          # transient per-prediction chunking, cleaned up automatically
├── chunked_files.json         # ledger: which raw files have already been chunked
├── clip_provenance.json       # which raw file(s) contributed to each dataset/ clip
├── global_stats.json          # dataset-wide mean/std (from compute_global_stats.py)
├── multi_baselines.json       # per-location sub-baseline centroids (from build_multi_baselines.py)
├── calibrated_thresholds.json # current STABLE/MODERATE thresholds (from calibrate_thresholds.py)
├── incoming_predictions.json  # accumulated prediction history (merges across runs)
├── incoming_latest_run.json   # just the files from the most recent predict_incoming.py run
├── test_split.json            # fixed held-out split used by verify_test_audio.py
└── verification_report.json   # verify_test_audio.py's most recent output
```

## 3. Pipeline architecture

```
raw recording (WAV)
  → chunk into fixed-length clips (chunk_audio.py)
  → per clip: DC removal → pre-emphasis → normalize → STFT → 40-band Mel filterbank (0-8000Hz)
    → 20 MFCC coefficients → 104-dim stats vector (20 coeffs x 5 stats + ZCR, spectral
    centroid, spectral flatness, RMS)                                  (feature_extraction.py)
  → z-score normalize against dataset-wide mean/std, clip ±3σ         (compute_global_stats.py)
  → cluster each location's vectors into up to 3 sub-baselines (KMeans: quiet/moderate/busy)
                                                                        (build_multi_baselines.py)

incoming clip → same feature pipeline → cosine similarity vs every location's sub-baselines
  → best score per location → argmax location → classify_similarity() against calibrated
    thresholds → STABLE / MODERATE_DRIFT / RELOCATION_OR_FRAUD (relabeled "unknown location"
    for open-set incoming scoring, see §8)
```

Why **multiple sub-baselines per location** instead of one: a single averaged baseline blurred
together genuinely different conditions at the same physical location (e.g. a quiet office
morning vs. a busy office afternoon look very different acoustically but are both "office").
Splitting each location into up to 3 KMeans clusters and scoring against the best-matching
cluster fixed this.

Why **cosine-similarity scoring, not classification**: early experiments with Random Forest /
XGBoost classifiers were tried and abandoned — they don't match how the underlying spec is
meant to work (per-location baseline + drift detection), and similarity scoring generalizes to
new locations without retraining a classifier.

## 4. Setup

```bash
pip install soundfile numpy scipy scikit-learn matplotlib yt-dlp --break-system-packages
# ffmpeg must be on PATH
```

All scripts use paths like `"Acoustic_Location_Detection/raw_recordings"` — **run every command
from the parent directory that contains this folder**, not from inside it.

## 5. End-to-end run order

```bash
# 1. (Optional) pull training audio for a location from YouTube — see §7 for details
python Acoustic_Location_Detection/yt_download.py train --location nature --search "forest ambience 1 hour" --count 1

# 2. Chunk any new raw recordings into fixed-length clips
python Acoustic_Location_Detection/chunk_audio.py --clip-seconds 5

# 3. Recompute dataset-wide normalization stats
python Acoustic_Location_Detection/compute_global_stats.py

# 4. Rebuild per-location sub-baselines
python Acoustic_Location_Detection/build_multi_baselines.py

# 5. Recalibrate STABLE/MODERATE/RELOCATION thresholds
python Acoustic_Location_Detection/calibrate_thresholds.py

# 6. (Optional) check held-out accuracy
python Acoustic_Location_Detection/verify_test_audio.py

# 7. Pull test clips and score them
python Acoustic_Location_Detection/yt_download.py test --location nature --search "forest sounds" --count 5
python Acoustic_Location_Detection/predict_incoming.py

# 8. Visualize
python Acoustic_Location_Detection/plot_similarity_matrix.py incoming
```

Steps 2-5 only need re-running when `raw_recordings/` changes — `chunk_audio.py` tracks what
it's already processed (see §6), so re-running the full sequence after adding one new file is
cheap, not a full rebuild from scratch.

## 6. Design decisions and why

- **10-second capture window** (spec calls for 3s) — more practical for standalone clips than
  live streaming hardware; configurable per-run via `--clip-seconds` on `chunk_audio.py`.
- **Dataset-scale, not fleet-scale, global stats** — normalization is computed from this
  project's own dataset, not a fleet-wide baseline, since there's no fleet yet.
- **Ledgered chunking** (`chunked_files.json`) — `chunk_audio.py` only processes raw files it
  hasn't seen before, so dropping new recordings into `raw_recordings/<location>/` and
  re-running only chunks the new ones, never redoing the whole dataset.
- **Dynamic location discovery** — every script discovers locations via `os.listdir()`/dict
  keys, nowhere are location names hardcoded. Confirmed working requirement: the system handles
  2 locations or 100 with no code changes.
- **Silence handling** — clips are rejected as "mostly silent" using both a per-sample amplitude
  threshold and a whole-clip RMS dead-air check, tuned so genuinely quiet real environments
  (e.g. a quiet office) aren't mistaken for dead air/no signal.
- **Merge-seam crossfade** — when a short raw file's audio gets merged with the next file in the
  same location during chunking, a 20ms crossfade smooths the seam so the abrupt splice doesn't
  distort the feature vector; `clip_provenance.json` tracks which raw file(s) contributed to
  each resulting clip.
- **Never zero-pad short audio** — a clip shorter than the chunk window is used whole rather
  than padded with silence, since padding would distort loudness-based features. This applies
  both to short leftovers during chunking and to short (3-10s) incoming test clips.

## 7. YouTube-audio direction (current focus)

Per team lead's direction: instead of relying on the physical mic for every location, pull
representative ambient audio from YouTube to train and test the system, so pipeline/system
performance can be evaluated independent of mic-capture logistics.

- `metro` and `home` remain phone-recorded, left untouched by YouTube downloads.
- `nature`, `traffic`, `cafeteria`, and `office` are supplemented or fully sourced from YouTube.
- Kept as **plain WAV**, deliberately — this is a system-performance test on the existing WAV
  pipeline, separate from the earlier AMR feasibility question (§11).

### `yt_download.py`

```bash
# One long training recording per location
python yt_download.py train --location nature --search "forest ambience 1 hour" --count 1
python yt_download.py train --location traffic --url "https://youtube.com/watch?v=..."

# Multiple locations/URLs in one call
python yt_download.py batch --config urls.json --mode train
#   urls.json: {"nature": ["url1","url2"], "traffic": ["url1"]}

# Short (3-10s default, configurable) test clips into incoming/
python yt_download.py test --location nature --search "forest sounds" --count 5
```

Repeated invocations append rather than overwrite — output filenames continue numbering from
whatever's already in the target folder.

## 8. Supporting tooling built for this phase

- **Unknown-location detection** — `predict_incoming.py` now distinguishes "confidently matched
  location X" from "doesn't match any known location well enough" (`location_verdict:
  "UNKNOWN_NEW_LOCATION"`), useful both for catching mislabeled test clips and for eventually
  flagging genuinely new locations.
- **Threshold auto-connect** — `calibrate_thresholds.py` writes its computed thresholds to
  `calibrated_thresholds.json`; `drift_detection.py` reads them automatically. No more copying
  numbers by hand between the two scripts. Supports `--override-stable`/`--override-moderate`
  for quick manual tuning, and `--stable-percentile`/`--moderate-percentile` to adjust the
  data-driven calculation.
- **Processed-file tracking** — `predict_incoming.py` moves each incoming file to
  `incoming/_processed/` after scoring, and merges results into `incoming_predictions.json`
  instead of overwriting it, so history isn't lost as the "already scored" set grows.
- **Latest-run plotting** — `plot_similarity_matrix.py incoming` shows only the most recent
  `predict_incoming.py` run by default (via `incoming_latest_run.json`); `incoming-all` shows
  the full accumulated history.

## 9. Current results snapshot

Most recent `calibrate_thresholds.py` run (4 locations — cafeteria, nature, office, traffic;
10,518 total clips across the dataset):

| | mean | min | max |
|---|---|---|---|
| Same-location scores | 0.6701 | 0.0467 | 0.9413 |
| Different-location scores | 0.1507 | -0.3054 | 0.7510 |

Thresholds in use: **STABLE** if score > 0.4260, **MODERATE** if 0.3547–0.4260, **RELOCATION /
unknown** if ≤ 0.3547.

`verify_test_audio.py` (automatic 20%-per-location held-out split): 4/4 held-out files matched
their expected location — **note this run's held-out files overlapped with training data**
(the script's built-in leakage check flagged this), so treat this number as optimistic; rerun
after excluding those files from `chunk_audio.py`'s input for an honest figure.

Fresh YouTube test clips (ground truth from filename) show a **mix of confident-correct and
confident-wrong results** — e.g. office and nature clips matching correctly with good margins,
alongside some cafeteria/traffic clips being confused with each other. This is the expected
shape of an actively-tuned system with a growing location set, not a stop condition — see §10.

## 10. Recommended next steps

- **Cafeteria/traffic/office discrimination** — a few fresh clips land close together in score;
  worth reviewing whether those particular source videos are representative of the location
  (background music, crowd noise, etc. can pull a clip toward an unrelated location's baseline).
- **`nature` source audio has a lot of silence** — 1085 of 1114 chunked clips were skipped as
  "mostly silent" in the most recent chunking run. Worth picking source videos with more
  continuous ambient sound rather than long quiet stretches.
- **`metro` currently has no active baseline** in `multi_baselines.json` — confirm
  `raw_recordings/metro/` has enough chunked clips before relying on metro predictions; this is
  a data-population step, not a pipeline issue.
- **Re-run `verify_test_audio.py` after excluding the flagged leakage files** for an honest
  held-out accuracy number to report alongside the fresh-clip results.
- **Margin-aware acceptance** — `predict_incoming.py` already computes a confidence margin
  between the top two location scores; gating acceptance on margin as well as raw score (not
  implemented yet) would catch thin, low-confidence wins more consistently than a score
  threshold alone.

## 11. Related work: AMR (real hardware) feasibility study

A parallel investigation, `Acoustic_Location_Detection_AMR/`, tested this same approach against
audio recorded in AMR-NB (the format ToneTag's soundbox hardware actually captures — a
narrowband 300–3400Hz codec, quite different from clean WAV). That investigation concluded the
codec's frequency loss is the limiting factor rather than data quantity or tuning — more
training data did not improve fresh-clip accuracy, and the codebase there was left intact with a
full final report for reference. If the hardware/firmware path changes to output WAV, this
project (not the AMR one) is the one already validated for that input — see that project's own
README for full details and repro commands.
