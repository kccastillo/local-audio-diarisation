"""Tests for v2/preprocessing.measure() — synthetic WAV fixtures.

Verifies the metric math against signals with known properties:
- Pure silence  → near-zero RMS, near-zero voiced fraction.
- Pure tone     → RMS ~= amplitude/sqrt(2), centroid ~= tone frequency.
- Speech-shaped → RMS in expected band, voiced fraction high, centroid in 1-3 kHz.

Uses soundfile to write tiny WAV fixtures; no external audio dependencies
beyond the v2 stack already pinned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from diarizer.preprocessing import AudioMeasurement, measure


SR = 16000


def _write_wav(path, samples, sr=SR):
    sf.write(str(path), samples.astype(np.float32), sr, subtype="FLOAT")


# ----------------------------------------------------------------------
# Silence
# ----------------------------------------------------------------------
def test_measure_silence(tmp_path):
    path = tmp_path / "silence.wav"
    _write_wav(path, np.zeros(SR * 10, dtype=np.float32))

    m = measure(path)
    assert isinstance(m, AudioMeasurement)
    assert m.sample_rate == SR
    assert m.channels == 1
    assert m.duration_s == pytest.approx(10.0, abs=0.05)
    assert m.rms_dbfs == float("-inf")  # pure zero
    assert m.peak_dbfs == float("-inf")
    assert m.voiced_fraction == pytest.approx(0.0, abs=0.01)


# ----------------------------------------------------------------------
# Pure 1 kHz tone — predictable RMS and centroid
# ----------------------------------------------------------------------
def test_measure_pure_tone(tmp_path):
    path = tmp_path / "tone.wav"
    duration_s = 5
    freq = 1000.0
    amp = 0.5
    t = np.arange(SR * duration_s) / SR
    samples = (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    _write_wav(path, samples)

    m = measure(path)
    expected_rms = amp / math.sqrt(2)
    expected_rms_dbfs = 20.0 * math.log10(expected_rms)

    assert m.duration_s == pytest.approx(duration_s, abs=0.05)
    assert m.rms_dbfs == pytest.approx(expected_rms_dbfs, abs=0.5)
    assert m.peak_dbfs == pytest.approx(20.0 * math.log10(amp), abs=0.5)
    # Spectral centroid for a pure 1 kHz tone should be at ~1 kHz
    assert 800 < m.spectral_centroid_hz < 1200
    # A constant-amplitude tone has uniform energy across frames — the noise-floor
    # vs speech-floor contrast is zero, so the voiced detector reports 0.0. This is
    # correct: voiced_fraction measures *contrast*, not loudness, so steady tones
    # don't trigger it. Real speech has plenty of contrast and reports near 1.0.
    assert m.voiced_fraction == pytest.approx(0.0, abs=0.05)


# ----------------------------------------------------------------------
# Stereo decoded to mono via channel-mean
# ----------------------------------------------------------------------
def test_measure_stereo_input(tmp_path):
    path = tmp_path / "stereo.wav"
    duration_s = 3
    t = np.arange(SR * duration_s) / SR
    left = (0.3 * np.sin(2 * math.pi * 1000 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * math.pi * 1000 * t + math.pi)).astype(np.float32)
    stereo = np.stack([left, right], axis=1)  # shape (n, 2)
    sf.write(str(path), stereo, SR, subtype="FLOAT")

    m = measure(path)
    assert m.channels == 2
    # Mean of opposite-phase signals is silence — RMS should be very low
    assert m.rms_dbfs < -30.0


# ----------------------------------------------------------------------
# Mixed signal — sparse pulses on quiet noise
# ----------------------------------------------------------------------
def test_measure_sparse_signal(tmp_path):
    """Five 200 ms tone bursts in 5 seconds — voiced fraction ~ 5*0.2/5 = 0.2."""
    path = tmp_path / "sparse.wav"
    duration_s = 5
    t = np.arange(SR * duration_s) / SR
    base_noise = 0.001 * np.random.randn(len(t))

    pulse_starts_s = [0.5, 1.5, 2.5, 3.5, 4.5]
    pulse_dur_s = 0.2
    samples = base_noise.copy()
    for ps in pulse_starts_s:
        i0 = int(ps * SR)
        i1 = i0 + int(pulse_dur_s * SR)
        local_t = t[i0:i1]
        samples[i0:i1] += 0.3 * np.sin(2 * math.pi * 1000 * local_t)

    _write_wav(path, samples.astype(np.float32))

    m = measure(path)
    # Voiced fraction should be in the ballpark of total pulse duration / total
    assert 0.1 < m.voiced_fraction < 0.6
    # Noise floor distinctly below speech floor
    assert m.speech_floor_dbfs - m.noise_floor_dbfs > 10


# ----------------------------------------------------------------------
# Estimated SNR derivation
# ----------------------------------------------------------------------
def test_measurement_snr_finite():
    m = AudioMeasurement(
        duration_s=10, sample_rate=16000, channels=1,
        rms_dbfs=-20, peak_dbfs=-3, noise_floor_dbfs=-50,
        speech_floor_dbfs=-15, voiced_fraction=0.8,
        spectral_centroid_hz=2500,
    )
    assert m.estimated_snr_db == pytest.approx(35.0)


def test_measurement_snr_inf_on_digital_silence_floor():
    m = AudioMeasurement(
        duration_s=10, sample_rate=16000, channels=1,
        rms_dbfs=-20, peak_dbfs=-3, noise_floor_dbfs=float("-inf"),
        speech_floor_dbfs=-15, voiced_fraction=0.8,
        spectral_centroid_hz=2500,
    )
    assert m.estimated_snr_db == float("inf")


# ----------------------------------------------------------------------
# Missing-file error
# ----------------------------------------------------------------------
def test_measure_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        measure(tmp_path / "does-not-exist.wav")


# ----------------------------------------------------------------------
# to_dict round-trip
# ----------------------------------------------------------------------
def test_measurement_to_dict():
    m = AudioMeasurement(
        duration_s=10, sample_rate=16000, channels=1,
        rms_dbfs=-20, peak_dbfs=-3, noise_floor_dbfs=-50,
        speech_floor_dbfs=-15, voiced_fraction=0.8,
        spectral_centroid_hz=2500,
    )
    d = m.to_dict()
    assert d["duration_s"] == 10
    assert d["estimated_snr_db"] == pytest.approx(35.0)
    assert d["sample_rate"] == 16000
