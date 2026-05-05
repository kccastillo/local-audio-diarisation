"""Tests for v2/gates.py — pure functions, deterministic, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from diarizer.gates import (
    Severity,
    asr_sanity_gate,
    diarisation_sanity_gate,
    ingest_gate,
    output_sanity_gate,
    preprocess_damage_gate,
    voice_presence_gate,
)


# ----------------------------------------------------------------------
# Lightweight stand-in for AudioMeasurement so gate tests don't need librosa
# ----------------------------------------------------------------------
@dataclass
class FakeMeasurement:
    duration_s: float = 60.0
    sample_rate: int = 16000
    channels: int = 1
    rms_dbfs: float = -25.0
    peak_dbfs: float = -1.0
    noise_floor_dbfs: float = -50.0
    speech_floor_dbfs: float = -15.0
    voiced_fraction: float = 0.85
    spectral_centroid_hz: float = 2500.0

    @property
    def estimated_snr_db(self) -> float:
        if self.noise_floor_dbfs == float("-inf"):
            return float("inf")
        return self.speech_floor_dbfs - self.noise_floor_dbfs

    def to_dict(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "speech_floor_dbfs": self.speech_floor_dbfs,
            "voiced_fraction": self.voiced_fraction,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "estimated_snr_db": self.estimated_snr_db,
        }


# ============================================================
# ingest_gate
# ============================================================
def test_ingest_gate_pass():
    res = ingest_gate(FakeMeasurement())
    assert res.passed
    assert res.severity == Severity.INFO


def test_ingest_gate_too_short():
    res = ingest_gate(FakeMeasurement(duration_s=2.0))
    assert not res.passed
    assert res.severity == Severity.BLOCK
    assert res.is_blocking_failure()
    assert "Duration" in res.message


def test_ingest_gate_unusual_sample_rate():
    res = ingest_gate(FakeMeasurement(sample_rate=12345))
    assert not res.passed
    assert res.severity == Severity.BLOCK


def test_ingest_gate_too_many_channels():
    res = ingest_gate(FakeMeasurement(channels=12))
    assert not res.passed
    assert res.severity == Severity.BLOCK


def test_ingest_gate_silent_file():
    res = ingest_gate(FakeMeasurement(rms_dbfs=-60.0))
    assert not res.passed
    assert res.severity == Severity.BLOCK
    assert "silent" in res.message.lower()


# ============================================================
# preprocess_damage_gate
# ============================================================
def test_preprocess_damage_gate_clean_pass():
    pre = FakeMeasurement(rms_dbfs=-20.0, spectral_centroid_hz=2500)
    post = FakeMeasurement(rms_dbfs=-19.0, spectral_centroid_hz=2400)
    res = preprocess_damage_gate(pre, post)
    assert res.passed
    assert not res.is_blocking_failure()


def test_preprocess_damage_gate_rms_drop_blocks():
    pre = FakeMeasurement(rms_dbfs=-20.0)
    post = FakeMeasurement(rms_dbfs=-30.0)  # 10 dB drop
    res = preprocess_damage_gate(pre, post)
    assert not res.passed
    assert res.is_blocking_failure()
    assert "RMS dropped" in res.message


def test_preprocess_damage_gate_centroid_drop_blocks():
    pre = FakeMeasurement(rms_dbfs=-20.0, spectral_centroid_hz=2500)
    post = FakeMeasurement(rms_dbfs=-20.0, spectral_centroid_hz=900)  # 1600 Hz drop
    res = preprocess_damage_gate(pre, post)
    assert not res.passed
    assert res.is_blocking_failure()
    assert "centroid" in res.message.lower()


def test_preprocess_damage_gate_centroid_unchecked_for_high_input():
    """If pre-centroid is already > 6 kHz (already-processed audio), the centroid
    drop check is skipped — we shouldn't penalise restoring normal speech band."""
    pre = FakeMeasurement(rms_dbfs=-20.0, spectral_centroid_hz=7500)
    post = FakeMeasurement(rms_dbfs=-20.0, spectral_centroid_hz=2500)  # huge drop, but OK
    res = preprocess_damage_gate(pre, post)
    assert res.passed


def test_preprocess_damage_gate_centroid_unchecked_when_resampled():
    """When sample rate changes (typical: 48 kHz → 16 kHz preprocess), the centroid
    drops mechanically because content above the new Nyquist is truncated. That's
    expected resampling behaviour, not denoise damage — gate must not fire on it."""
    pre = FakeMeasurement(rms_dbfs=-20.0, sample_rate=48000, spectral_centroid_hz=3000)
    post = FakeMeasurement(rms_dbfs=-20.0, sample_rate=16000, spectral_centroid_hz=1100)
    res = preprocess_damage_gate(pre, post)
    assert res.passed
    assert not res.is_blocking_failure()


# ============================================================
# voice_presence_gate
# ============================================================
def test_voice_presence_gate_pass():
    res = voice_presence_gate(0.85)
    assert res.passed
    assert res.severity == Severity.INFO


def test_voice_presence_gate_silent_blocks():
    res = voice_presence_gate(0.01)
    assert not res.passed
    assert res.severity == Severity.BLOCK
    assert res.is_blocking_failure()


def test_voice_presence_gate_sparse_warns_but_passes():
    res = voice_presence_gate(0.20)
    assert res.passed
    assert res.severity == Severity.WARN
    assert "sparse" in res.message.lower()


# ============================================================
# diarisation_sanity_gate
# ============================================================
def test_diarisation_sanity_gate_pass():
    res = diarisation_sanity_gate(
        speaker_count=3,
        speaker_share={"A": 0.4, "B": 0.35, "C": 0.25},
    )
    assert res.passed
    assert res.severity == Severity.INFO


def test_diarisation_sanity_gate_zero_speakers_warns():
    res = diarisation_sanity_gate(speaker_count=0, speaker_share={})
    assert not res.passed
    assert res.severity == Severity.WARN
    # Crucially: WARN, not BLOCK
    assert not res.is_blocking_failure()


def test_diarisation_sanity_gate_below_min_hint():
    res = diarisation_sanity_gate(
        speaker_count=1,
        speaker_share={"A": 1.0},
        min_speakers_hint=2,
    )
    assert not res.passed
    assert res.severity == Severity.WARN


def test_diarisation_sanity_gate_above_max_hint():
    res = diarisation_sanity_gate(
        speaker_count=8,
        speaker_share={f"S{i}": 0.125 for i in range(8)},
        max_speakers_hint=4,
    )
    assert not res.passed
    assert res.severity == Severity.WARN


def test_diarisation_sanity_gate_monopoly_warns():
    res = diarisation_sanity_gate(
        speaker_count=2,
        speaker_share={"A": 0.99, "B": 0.01},
    )
    assert not res.passed
    assert "collapsed" in res.message.lower() or "holds" in res.message.lower()


def test_diarisation_sanity_gate_spurious_cluster():
    res = diarisation_sanity_gate(
        speaker_count=4,
        speaker_share={"A": 0.5, "B": 0.45, "C": 0.04, "D": 0.001},
    )
    # passes but with spurious-warn
    assert res.severity == Severity.WARN
    assert "spurious" in res.message.lower()


# ============================================================
# asr_sanity_gate
# ============================================================
def test_asr_sanity_gate_pass():
    res = asr_sanity_gate(
        transcript_chars=2000,
        voiced_duration_s=120.0,
        detected_language="en",
    )
    assert res.passed
    assert res.severity == Severity.INFO


def test_asr_sanity_gate_empty_transcript_warns():
    res = asr_sanity_gate(
        transcript_chars=0,
        voiced_duration_s=120.0,
        detected_language="en",
    )
    assert not res.passed
    assert res.severity == Severity.WARN
    assert "empty transcript" in res.message.lower()


def test_asr_sanity_gate_undertranscription_warns():
    """0.5 s of voiced audio with 1 char ⇒ 120 chars/min — passes."""
    """5 chars across 60 s of voiced audio ⇒ 5 chars/min — fails."""
    res = asr_sanity_gate(
        transcript_chars=5,
        voiced_duration_s=60.0,
        detected_language="en",
    )
    assert not res.passed
    assert res.severity == Severity.WARN
    assert "undertranscription" in res.message.lower() or "below" in res.message.lower()


def test_asr_sanity_gate_metrics_populated():
    res = asr_sanity_gate(
        transcript_chars=600,
        voiced_duration_s=60.0,
        detected_language="en",
    )
    assert res.metrics["transcript_chars"] == 600
    assert res.metrics["chars_per_voiced_minute"] == pytest.approx(600.0)


# ============================================================
# output_sanity_gate
# ============================================================
def test_output_sanity_gate_pass(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("some content", encoding="utf-8")
    res = output_sanity_gate(p, expected_segment_count=5)
    assert res.passed
    assert res.severity == Severity.INFO


def test_output_sanity_gate_missing_file(tmp_path):
    res = output_sanity_gate(tmp_path / "does-not-exist.txt", expected_segment_count=5)
    assert not res.passed
    assert res.severity == Severity.WARN


def test_output_sanity_gate_empty_file(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("", encoding="utf-8")
    res = output_sanity_gate(p, expected_segment_count=5)
    assert not res.passed
    assert res.severity == Severity.WARN


def test_output_sanity_gate_zero_segments(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("written but no segments", encoding="utf-8")
    res = output_sanity_gate(p, expected_segment_count=0)
    # Passes (file is fine) but warns (transcript is empty)
    assert res.passed
    assert res.severity == Severity.WARN


# ============================================================
# GateResult helpers
# ============================================================
def test_gate_result_str_repr():
    res = ingest_gate(FakeMeasurement())
    s = str(res)
    assert "ingest" in s
    assert "OK" in s


def test_gate_result_blocking_str():
    res = ingest_gate(FakeMeasurement(duration_s=1.0))
    s = str(res)
    assert "BLOCK" in s
