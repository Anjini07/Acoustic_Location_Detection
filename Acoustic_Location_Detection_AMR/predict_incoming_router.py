"""
predict_incoming_router.py
The one shared script that connects the two otherwise-fully-separate
projects: routes an incoming file to Acoustic_Location_Detection (WAV) or
Acoustic_Location_Detection_AMR (this project) purely by file extension.

Both projects contain files with identical names (audio_preprocessing.py,
feature_extraction.py, drift_detection.py, chunk_audio.py,
predict_incoming.py) that must NOT be imported into the same sys.modules
entries at once -- doing so would silently make one domain's predictions
use the other domain's feature extraction. This router avoids that by:
  1. Purging any previously-cached same-name modules from sys.modules
  2. Temporarily putting ONLY the target project's folder on sys.path
  3. Loading that project's predict_incoming.py under a unique module name
  4. Removing the folder from sys.path again once loaded

Expected layout (sibling folders under a common parent):
    <parent>/
      Acoustic_Location_Detection/       <- WAV project
      Acoustic_Location_Detection_AMR/   <- this project (and this script)

Usage:
    python predict_incoming_router.py path/to/some_recording.wav
    python predict_incoming_router.py path/to/some_recording.amr
"""

import os
import sys
import importlib
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AMR_PROJECT_DIR = _THIS_DIR
WAV_PROJECT_DIR = os.path.join(_THIS_DIR, "..", "Acoustic_Location_Detection")

# Same-named modules that exist in BOTH project folders -- must be purged
# from sys.modules before switching domains so Python re-resolves them
# against whichever project's folder is currently on sys.path.
_SHARED_MODULE_NAMES = [
    "audio_preprocessing",
    "feature_extraction",
    "drift_detection",
    "chunk_audio",
    "predict_incoming",
]


def _load_predict_module(project_dir, unique_name):
    project_dir = os.path.abspath(project_dir)

    if not os.path.isdir(project_dir):
        raise FileNotFoundError(
            f"Expected project folder not found: {project_dir}\n"
            f"predict_incoming_router.py expects the WAV and AMR projects "
            f"to sit as sibling folders under the same parent directory."
        )

    for name in _SHARED_MODULE_NAMES:
        sys.modules.pop(name, None)

    sys.path.insert(0, project_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            unique_name, os.path.join(project_dir, "predict_incoming.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(project_dir)

    return module


def predict(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".wav":
        print(f"[router] .wav file -> routing to WAV pipeline ({WAV_PROJECT_DIR})")
        mod = _load_predict_module(WAV_PROJECT_DIR, "predict_incoming_wav_domain")
    elif ext == ".amr":
        print(f"[router] .amr file -> routing to AMR pipeline ({AMR_PROJECT_DIR})")
        mod = _load_predict_module(AMR_PROJECT_DIR, "predict_incoming_amr_domain")
    else:
        raise ValueError(f"Unsupported extension '{ext}' -- expected .wav or .amr")

    return mod.predict_file(file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Route an incoming .wav or .amr file to the correct domain pipeline"
    )
    parser.add_argument("file_path", help="Path to a .wav or .amr file to classify")
    args = parser.parse_args()

    result = predict(args.file_path)
    print("\n[router] Final result:", result)
