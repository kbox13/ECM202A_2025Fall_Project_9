#!/usr/bin/env python3
"""
kick_only_gt_demucs.py

Use Demucs to separate full-mix audio into stems, then analyze the DRUMS stem
to estimate onset times for KICK hits only.

Outputs a JSON file with structure:

{
  "track_basename": {
    "kick": [t0, t1, t2, ...]   # kick onset times in seconds
  },
  ...
}

Requirements (you install these yourself):
  pip install demucs librosa soundfile numpy scipy

Demucs is invoked via its command-line interface, so `demucs` must be on PATH.

Example usage:

  python kick_only_gt_demucs.py \
      --input path/to/song_or_folder \
      --output-json kicks.json \
      --model htdemucs \
      --device cpu
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import librosa
import numpy as np
import scipy.signal as sps


def run_demucs(audio_path: str, model: str = "htdemucs", device: str = "cpu") -> str:
    """
    Run Demucs separation on a single audio file.

    Returns the path to the DRUMS stem for this track.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        "demucs",
        "-n", model,
        "-d", device,
        audio_path,
    ]
    print(f"[demucs] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    base = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join("separated", model, base)
    drums_path = os.path.join(stem_dir, "drums.wav")

    if not os.path.isfile(drums_path):
        raise FileNotFoundError(f"Expected drums stem not found at {drums_path}")

    return drums_path


def lowpass_filter(x: np.ndarray, sr: int, cutoff: float = 150.0, order: int = 4) -> np.ndarray:
    """
    Simple zero-phase low-pass filter to emphasize kick energy.
    """
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = sps.butter(order, norm_cutoff, btype="low")
    return sps.filtfilt(b, a, x)


def merge_close_events(times: List[float], min_sep: float = 0.01) -> List[float]:
    """
    Merge events that are closer than min_sep seconds apart
    (e.g., layered kicks or flams).
    """
    if not times:
        return []
    times = sorted(times)
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] > min_sep:
            merged.append(t)
    return merged


def detect_kicks_from_drums(drums_path: str) -> List[float]:
    """
    Given a path to a drums stem, detect kick onsets.

    Strategy:
      - Load drums stem.
      - Low-pass filter to ~150 Hz to emphasize kicks.
      - Use librosa.onset.onset_detect on the filtered signal.
      - Merge close events.
    """
    print(f"[analyze] Loading drums stem: {drums_path}")
    y, sr = librosa.load(drums_path, sr=None, mono=True)

    # Emphasize low-frequency content (kick region)
    print("[analyze] Low-pass filtering drums to emphasize kicks...")
    y_low = lowpass_filter(y, sr, cutoff=150.0)

    # Onset detection on low-frequency envelope
    print("[analyze] Detecting kick onsets...")
    onset_times = librosa.onset.onset_detect(
        y=y_low,
        sr=sr,
        units="time",
        backtrack=True,
    )

    onset_times = [float(t) for t in onset_times]
    print(f"[analyze] Found {len(onset_times)} raw kick-like onsets.")

    # Merge very close onsets
    kick_times = merge_close_events(onset_times, min_sep=0.01)
    print(f"[analyze] After merging close events: {len(kick_times)} kicks.")

    return kick_times


def find_audio_files(input_path: str) -> List[str]:
    """
    Return a list of audio file paths under input_path.
    If input_path is a file, return [input_path].
    If it's a directory, search for common audio extensions.
    """
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for root, _, fnames in os.walk(input_path):
        for fn in fnames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Separate drums with Demucs and extract kick onset times only."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to an audio file or a folder of audio files.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="htdemucs",
        help="Demucs model name (default: htdemucs).",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        help="Device for Demucs (cpu or cuda, default: cpu).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip Demucs separation if drums stem already exists.",
    )

    args = parser.parse_args()

    audio_files = find_audio_files(args.input)
    if not audio_files:
        print(f"No audio files found under: {args.input}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s).")

    all_results: Dict[str, Dict[str, List[float]]] = {}

    for audio_path in audio_files:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        print("=" * 80)
        print(f"Processing: {audio_path}")

        # Expected drums stem path
        stem_dir = os.path.join("separated", args.model, base)
        drums_path = os.path.join(stem_dir, "drums.wav")

        if not (args.skip_existing and os.path.isfile(drums_path)):
            drums_path = run_demucs(audio_path, model=args.model, device=args.device)
        else:
            print(f"[demucs] Skipping separation, using existing: {drums_path}")

        kick_times = detect_kicks_from_drums(drums_path)
        all_results[base] = {"kick": kick_times}

    print(f"\n[output] Writing JSON to: {args.output_json}")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("[done] Kick onset extraction complete.")


if __name__ == "__main__":
    main()
