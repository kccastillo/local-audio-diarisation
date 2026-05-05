"""Pre-rendered waveform peak extraction.

Reads a mono Opus file with `soundfile`, computes per-bin peak amplitudes,
and writes a `waveform_peaks.json` manifest the webapp loads on init.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PEAKS_VERSION = 1
DEFAULT_TARGET_BINS = 4000
MIN_SAMPLES_PER_BIN = 100


def compute_peaks(opus_path: Path | str, target_bins: int = DEFAULT_TARGET_BINS) -> dict:
    """Decode `opus_path` and return a peaks manifest.

    Schema: {"version": 1, "bins": int, "duration_s": float, "peaks": [float, ...]}
    Each peak is in [0, 1]. Length of `peaks` == `bins`.
    """
    import soundfile as sf  # heavy import; defer

    p = Path(opus_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio source not found: {p}")

    samples, sample_rate = sf.read(str(p), always_2d=False)
    # Defensive mono squash — D6 locks source as mono Opus, but tolerate stereo.
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)

    n = len(samples)
    duration_s = float(n) / float(sample_rate) if sample_rate else 0.0

    # For very short audio, drop bins so each holds at least MIN_SAMPLES_PER_BIN samples.
    bins = min(target_bins, max(1, n // MIN_SAMPLES_PER_BIN))
    if bins < 1:
        bins = 1

    # Trim to a length divisible by bins so reshape is clean.
    usable = (n // bins) * bins
    if usable == 0:
        # Pathological: less than `bins` samples. Pad or fall back to single bin.
        peaks_arr = np.array([float(np.max(np.abs(samples))) if n else 0.0])
        bins = 1
    else:
        chunk = samples[:usable].reshape(bins, -1)
        peaks_arr = np.max(np.abs(chunk), axis=1)

    max_abs = float(np.max(peaks_arr)) if len(peaks_arr) else 0.0
    if max_abs > 0:
        peaks_arr = peaks_arr / max_abs

    return {
        "version": PEAKS_VERSION,
        "bins": int(bins),
        "duration_s": duration_s,
        "peaks": [float(x) for x in peaks_arr],
    }


def write_peaks(opus_path: Path | str, out_json_path: Path | str, target_bins: int = DEFAULT_TARGET_BINS) -> Path:
    """Compute + write peaks JSON. Returns the output path."""
    out = Path(out_json_path)
    data = compute_peaks(opus_path, target_bins=target_bins)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")
    return out
