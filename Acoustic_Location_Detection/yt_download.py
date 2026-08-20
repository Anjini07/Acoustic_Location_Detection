"""
yt_download.py -- pulls audio-only from YouTube and drops it straight into
the existing pipeline folders (raw_recordings/<location>/ for training,
incoming/ for short test clips). Downstream (chunk_audio.py,
compute_global_stats.py, build_multi_baselines.py, calibrate_thresholds.py,
predict_incoming.py) is untouched -- this script only produces WAV files in
the same layout those scripts already expect.

Requirements (install once):
    pip install yt-dlp --break-system-packages
    ffmpeg must be on PATH (or point FFMPEG_PATH below at your ffmpeg/ folder)

--------------------------------------------------------------------------
HOW TO DOWNLOAD *ONLY* THE AUDIO FROM A YOUTUBE VIDEO (no video track):
    yt-dlp -x --audio-format wav -o "out.wav" "<youtube_url>"
-x / --extract-audio tells yt-dlp to grab just the audio stream and drop the
video entirely (it doesn't download the full video first) -- --audio-format
wav then has ffmpeg transcode whatever audio codec YouTube served into WAV.
This script wraps exactly that command, plus a resample/mono step and
placement into the right folder.
--------------------------------------------------------------------------

USAGE

Train (long-form ambient audio per location):
    python yt_download.py train --location nature --url "<youtube_url>"
    python yt_download.py train --location nature --search "forest ambience 1 hour" --count 1
    python yt_download.py train --location traffic --url URL1 --url URL2

Test (short 3-10s clips, trimmed from separate/unrelated source videos):
    python yt_download.py test --location nature --url "<youtube_url>" --count 5
    python yt_download.py test --location traffic --search "city traffic sounds" --count 3 \
        --min-sec 3 --max-sec 10
"""

import os
import sys
import glob
import json
import random
import argparse
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_preprocessing import TARGET_SR

RAW_DIR = "Acoustic_Location_Detection/raw_recordings"
INCOMING_DIR = "Acoustic_Location_Detection/incoming"

# Locations you said you're keeping from phone recordings -- this script
# won't touch raw_recordings/metro or raw_recordings/home unless you
# explicitly pass --location metro / --location home yourself.
PHONE_RECORDED_LOCATIONS = {"metro", "home"}


def _run(cmd, description):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAILED] {description}\n{result.stderr[-2000:]}")
        return False
    return True


def _resolve_sources(url_list, search, count):
    """Returns a list of individual YouTube video URLs: explicit --url values
    as given, or --count results resolved from --search.

    NOTE: this used to just return ["ytsearch<count>:<query>"] and let
    download_audio_wav's yt-dlp call handle the search directly. That broke
    --count > 1: yt-dlp treats a multi-result search as a "playlist" of N
    entries, and download_audio_wav passes --no-playlist (needed so a
    single --url that happens to be part of a playlist doesn't pull the
    whole thing) -- --no-playlist also silently collapses a search to just
    its first result. Resolving to individual URLs here, then downloading
    each one separately (same as --url already does), avoids that
    interaction entirely.
    """
    if url_list:
        return url_list
    if search:
        return _search_urls(search, count)
    raise ValueError("Provide either --url (one or more) or --search.")


def _search_urls(query, count):
    """Resolves a search query to `count` individual video URLs via yt-dlp's
    own search (--flat-playlist avoids downloading anything here, just lists
    matches; --print url gives one URL per line)."""
    cmd = [
        "yt-dlp", f"ytsearch{count}:{query}",
        "--flat-playlist", "--print", "url", "--no-warnings",
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAILED] search for '{query}'\n{result.stderr[-1000:]}")
        return []
    urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(urls) < count:
        print(f"  [WARNING] search for '{query}' only returned {len(urls)}/{count} result(s)")
    return urls


def download_audio_wav(source, out_path, target_sr=TARGET_SR):
    """Downloads audio-only from one YouTube source (URL or ytsearch query)
    and writes it as mono WAV at target_sr directly -- yt-dlp's ffmpeg
    postprocessor handles both the audio extraction and the resample/mono
    conversion in one pass, so no separate ffmpeg call is needed."""
    tmp_template = out_path[:-4] + ".%(ext)s"  # strip .wav, let yt-dlp/ffmpeg name it
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ar {target_sr} -ac 1",
        "--no-playlist",
        "-o", tmp_template,
        source,
    ]
    ok = _run(cmd, f"download {source}")
    if ok and not os.path.exists(out_path):
        # yt-dlp may have picked a slightly different final name; find it.
        candidates = glob.glob(tmp_template.replace("%(ext)s", "*"))
        if candidates:
            os.replace(candidates[0], out_path)
        else:
            ok = False
    return ok


def _next_index(directory, prefix):
    """Scans `directory` for files already named f'{prefix}NNN....wav' and
    returns the next unused index. Used so that running this script again
    (e.g. with a different --url) continues numbering from what's already
    there instead of restarting at 0 -- which previously either overwrote
    an earlier run's file (test mode, ffmpeg -y) or got silently skipped as
    'already exists' (train mode, meaning the second URL never downloaded
    at all)."""
    if not os.path.isdir(directory):
        return 0
    existing = glob.glob(os.path.join(directory, f"{prefix}*"))
    max_idx = -1
    for path in existing:
        stem = os.path.basename(path)[len(prefix):].split(".")[0]
        if stem.isdigit():
            max_idx = max(max_idx, int(stem))
    return max_idx + 1


def cmd_train(args):
    if args.location in PHONE_RECORDED_LOCATIONS and not args.force:
        print(f"'{args.location}' is marked as phone-recorded in this script "
              f"(PHONE_RECORDED_LOCATIONS). Pass --force if you really want to "
              f"add YouTube audio to it too.")
        return

    out_dir = os.path.join(RAW_DIR, args.location)
    os.makedirs(out_dir, exist_ok=True)

    sources = _resolve_sources(args.url, args.search, args.count)
    print(f"Downloading {len(sources)} training source(s) for '{args.location}' -> {out_dir}")

    prefix = f"yt_{args.location}_"
    idx = _next_index(out_dir, prefix)
    for source in sources:
        out_path = os.path.join(out_dir, f"{prefix}{idx:03d}.wav")
        download_audio_wav(source, out_path)
        idx += 1


def cmd_test(args):
    os.makedirs(INCOMING_DIR, exist_ok=True)
    sources = _resolve_sources(args.url, args.search, args.count)

    if args.url and args.count > 1:
        print(f"  [note] --count is ignored for explicit --url downloads -- each URL "
              f"always produces exactly one incoming file (predict_incoming.py splits "
              f"it into windows internally and gives ONE result for it). To get more "
              f"test recordings, pass --url multiple times or use --search --count N "
              f"for N distinct videos.")

    print(f"Downloading {len(sources)} test recording(s) for '{args.location}' -> {INCOMING_DIR} "
          f"(1 incoming file per recording; predict_incoming.py handles windowing internally)")

    prefix = f"yttest_{args.location}_"
    idx = _next_index(INCOMING_DIR, prefix)

    with tempfile.TemporaryDirectory() as tmp:
        for si, source in enumerate(sources):
            full_path = os.path.join(tmp, f"full_{si:03d}.wav")
            if not download_audio_wav(source, full_path):
                continue

            duration = _probe_duration(full_path)

            # Trim a random clip_len-second window (per --min-sec/--max-sec)
            # from somewhere inside the download, skipping the first/last 5s
            # to avoid intros/outros/silence. Default 3-10s matches "quick
            # sample" testing; pass longer values (e.g. --min-sec 30
            # --max-sec 60) if you want this file to actually span multiple
            # of predict_incoming.py's internal windows and get the
            # cross-window averaging it does for longer recordings.
            clip_len = random.uniform(args.min_sec, args.max_sec)
            if duration is None or duration <= clip_len + 10:
                start = 0
            else:
                start = random.uniform(5, duration - clip_len - 5)

            out_path = os.path.join(INCOMING_DIR, f"{prefix}{idx:03d}.wav")
            trim_cmd = [
                "ffmpeg", "-y", "-i", full_path,
                "-ss", f"{start:.2f}", "-t", f"{clip_len:.2f}",
                "-ar", str(TARGET_SR), "-ac", "1",
                out_path,
            ]
            _run(trim_cmd, f"trim {out_path}")
            idx += 1


def cmd_batch(args):
    """Runs train or test for every location in a JSON config file in one
    call, instead of one --location invocation per location. Config format:
        {
          "nature":  ["https://youtube.com/watch?v=AAA", "https://youtube.com/watch?v=BBB"],
          "traffic": ["https://youtube.com/watch?v=CCC"]
        }
    Each location's list is treated exactly like passing --url once per
    entry -- no search resolution here, this mode is for when you already
    have the specific URLs picked out per location.
    """
    with open(args.config, "r") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Batch config must be a JSON object of {location: [urls...]}")

    print(f"Batch {args.mode}: {len(config)} location(s) from {args.config}")
    for location, urls in config.items():
        if not urls:
            print(f"  [skip] '{location}' has no URLs listed")
            continue
        print(f"\n-- {location} ({len(urls)} url(s)) --")
        ns = argparse.Namespace(location=location, url=list(urls), search=None,
                                 count=len(urls), force=args.force)
        if args.mode == "train":
            cmd_train(ns)
        else:
            ns.min_sec = args.min_sec
            ns.max_sec = args.max_sec
            cmd_test(ns)

    print("\nBatch done. Next: chunk_audio.py (train) or predict_incoming.py (test).")


def _probe_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Download long-form audio into raw_recordings/<location>/")
    p_train.add_argument("--location", required=True, help="e.g. nature, traffic")
    p_train.add_argument("--url", action="append", default=[], help="YouTube URL (repeatable)")
    p_train.add_argument("--search", default=None, help="Search query instead of explicit URL(s)")
    p_train.add_argument("--count", type=int, default=1, help="How many search results to pull (if using --search)")
    p_train.add_argument("--force", action="store_true", help="Allow adding YT audio to a phone-recorded location")
    p_train.set_defaults(func=cmd_train, kind="train")

    p_test = sub.add_parser("test", help="Download and trim short 3-10s clips into incoming/")
    p_test.add_argument("--location", required=True, help="Label used in the output filename only")
    p_test.add_argument("--url", action="append", default=[], help="YouTube URL (repeatable)")
    p_test.add_argument("--search", default=None, help="Search query instead of explicit URL(s)")
    p_test.add_argument("--count", type=int, default=1, help="How many clips/search results to pull")
    p_test.add_argument("--min-sec", type=float, default=3.0)
    p_test.add_argument("--max-sec", type=float, default=10.0)
    p_test.set_defaults(func=cmd_test, kind="test")

    p_batch = sub.add_parser(
        "batch",
        help="Download multiple locations at once from a JSON file of {location: [urls...]}",
    )
    p_batch.add_argument("--config", required=True, help="Path to JSON file: {\"location\": [\"url\", ...], ...}")
    p_batch.add_argument("--mode", required=True, choices=["train", "test"],
                          help="train -> raw_recordings/<location>/, test -> incoming/")
    p_batch.add_argument("--force", action="store_true", help="(train only) allow phone-recorded locations")
    p_batch.add_argument("--min-sec", type=float, default=3.0, help="(test only)")
    p_batch.add_argument("--max-sec", type=float, default=10.0, help="(test only)")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)

    kind = args.mode if args.command == "batch" else args.kind
    if kind == "train":
        print("\nDone. Next: run chunk_audio.py (e.g. `python chunk_audio.py --clip-seconds 10`) "
              "to fold these into dataset/.")
    else:
        print("\nDone. Next: run predict_incoming.py to score these against the existing baselines.")


if __name__ == "__main__":
    main()