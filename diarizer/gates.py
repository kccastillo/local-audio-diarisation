"""v2 validation gates — cheap, fail-fast checks between pipeline stages.

Six gates, two severities:

  BLOCK  — abort the run. Cheaper to fail here than to commit GPU time.
           Gates 0 (ingest), 1 (preprocess damage), 2 (voice presence).
  WARN   — log and continue. Quality observation, not a correctness failure.
           Gates 3 (diarisation), 4 (ASR sanity), 5 (output sanity).

Each gate is a pure function of its inputs and returns a GateResult.
The pipeline orchestrator calls gates in order and decides what to do
with each result — gates themselves never raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


@dataclass
class GateResult:
    name: str
    passed: bool
    severity: Severity
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def is_blocking_failure(self) -> bool:
        return (not self.passed) and self.severity == Severity.BLOCK

    def __str__(self) -> str:
        verdict = "OK" if self.passed else self.severity.value.upper()
        return f"[{self.name}: {verdict}] {self.message}"


# Gate 0 — Ingest sanity (BLOCK)
def ingest_gate(
    measurement,  # AudioMeasurement; not type-hinted to avoid circular import
    *,
    min_duration_s: float = 10.0,
    valid_sample_rates: tuple[int, ...] = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000),
    max_channels: int = 8,
    min_rms_dbfs: float = -50.0,
) -> GateResult:
    """File opened, duration sane, audible content present.

    Failures here mean the file is corrupt, empty, or pure silence — refuse to
    spend Whisper time on it.
    """
    metrics = measurement.to_dict() if hasattr(measurement, "to_dict") else {}

    if measurement.duration_s < min_duration_s:
        return GateResult(
            name="ingest",
            passed=False,
            severity=Severity.BLOCK,
            message=f"Duration {measurement.duration_s:.1f}s < {min_duration_s}s minimum (probably corrupt or empty file).",
            metrics=metrics,
        )

    if measurement.sample_rate not in valid_sample_rates:
        return GateResult(
            name="ingest",
            passed=False,
            severity=Severity.BLOCK,
            message=f"Unusual sample rate {measurement.sample_rate} Hz; expected one of {valid_sample_rates}.",
            metrics=metrics,
        )

    if measurement.channels < 1 or measurement.channels > max_channels:
        return GateResult(
            name="ingest",
            passed=False,
            severity=Severity.BLOCK,
            message=f"Channel count {measurement.channels} outside 1..{max_channels}.",
            metrics=metrics,
        )

    if measurement.rms_dbfs < min_rms_dbfs:
        return GateResult(
            name="ingest",
            passed=False,
            severity=Severity.BLOCK,
            message=f"RMS {measurement.rms_dbfs:.1f} dBFS below {min_rms_dbfs} dBFS floor — file appears silent.",
            metrics=metrics,
        )

    return GateResult(
        name="ingest",
        passed=True,
        severity=Severity.INFO,
        message=f"Duration {measurement.duration_s:.0f}s, {measurement.sample_rate}Hz, {measurement.channels}ch, RMS {measurement.rms_dbfs:.1f} dBFS.",
        metrics=metrics,
    )


# Gate 1 — Preprocessing didn't damage signal (BLOCK)
def preprocess_damage_gate(
    pre,
    post,
    *,
    rms_drop_threshold_db: float = 6.0,
    centroid_drop_threshold_hz: float = 1000.0,
) -> GateResult:
    """Compare pre- vs post-preprocessing measurements. Loudnorm should normalise,
    not destroy. Denoise shouldn't have eaten the speech band.
    """
    metrics = {
        "pre_rms_dbfs": pre.rms_dbfs,
        "post_rms_dbfs": post.rms_dbfs,
        "rms_delta_db": post.rms_dbfs - pre.rms_dbfs,
        "pre_centroid_hz": pre.spectral_centroid_hz,
        "post_centroid_hz": post.spectral_centroid_hz,
        "centroid_delta_hz": post.spectral_centroid_hz - pre.spectral_centroid_hz,
    }

    rms_drop = pre.rms_dbfs - post.rms_dbfs
    if rms_drop > rms_drop_threshold_db:
        return GateResult(
            name="preprocess_damage",
            passed=False,
            severity=Severity.BLOCK,
            message=f"Post-preprocess RMS dropped {rms_drop:.1f} dB (>{rms_drop_threshold_db} dB). Filter chain destroyed signal.",
            metrics=metrics,
        )

    # Centroid dropping a lot can indicate over-aggressive denoise eating mid-band content.
    # Only flag if the input wasn't already pre-processed (centroid in normal range).
    if pre.spectral_centroid_hz < 6000.0:
        centroid_drop = pre.spectral_centroid_hz - post.spectral_centroid_hz
        if centroid_drop > centroid_drop_threshold_hz:
            return GateResult(
                name="preprocess_damage",
                passed=False,
                severity=Severity.BLOCK,
                message=f"Post-preprocess spectral centroid dropped {centroid_drop:.0f} Hz (>{centroid_drop_threshold_hz}). Denoise damaged speech band.",
                metrics=metrics,
            )

    return GateResult(
        name="preprocess_damage",
        passed=True,
        severity=Severity.INFO,
        message=f"RMS shift {metrics['rms_delta_db']:+.1f} dB, centroid shift {metrics['centroid_delta_hz']:+.0f} Hz.",
        metrics=metrics,
    )


# Gate 2 — Voice presence (BLOCK on absence)
def voice_presence_gate(
    voiced_fraction: float,
    *,
    min_voiced_fraction: float = 0.05,
    sparse_threshold: float = 0.30,
) -> GateResult:
    """At least `min_voiced_fraction` of audio frames must be above noise floor.
    Below that, the file is silent — refuse to transcribe.
    """
    metrics = {"voiced_fraction": voiced_fraction}

    if voiced_fraction < min_voiced_fraction:
        return GateResult(
            name="voice_presence",
            passed=False,
            severity=Severity.BLOCK,
            message=f"Voiced fraction {voiced_fraction:.1%} below {min_voiced_fraction:.0%} threshold; refusing to transcribe near-silent audio.",
            metrics=metrics,
        )

    if voiced_fraction < sparse_threshold:
        return GateResult(
            name="voice_presence",
            passed=True,
            severity=Severity.WARN,
            message=f"Sparse audio detected ({voiced_fraction:.1%} voiced) — VAD will skip large silent stretches; expect short transcript.",
            metrics=metrics,
        )

    return GateResult(
        name="voice_presence",
        passed=True,
        severity=Severity.INFO,
        message=f"Voiced fraction {voiced_fraction:.1%}.",
        metrics=metrics,
    )


# Gate 3 — Diarisation sanity (WARN)
def diarisation_sanity_gate(
    speaker_count: int,
    speaker_share: dict[str, float],
    *,
    min_speakers_hint: Optional[int] = None,
    max_speakers_hint: Optional[int] = None,
    monopoly_threshold: float = 0.98,
    spurious_threshold: float = 0.005,
) -> GateResult:
    """Sanity check pyannote output. Failures warn but don't block — the
    transcript is still useful with messy speaker labels.
    """
    metrics = {
        "speaker_count": speaker_count,
        "speaker_share": speaker_share,
    }

    if speaker_count == 0:
        return GateResult(
            name="diarisation_sanity",
            passed=False,
            severity=Severity.WARN,
            message="Diarisation produced 0 speakers — transcript will have no attribution.",
            metrics=metrics,
        )

    if min_speakers_hint is not None and speaker_count < min_speakers_hint:
        return GateResult(
            name="diarisation_sanity",
            passed=False,
            severity=Severity.WARN,
            message=f"Only {speaker_count} speaker(s) detected; hint asked for ≥{min_speakers_hint}.",
            metrics=metrics,
        )

    if max_speakers_hint is not None and speaker_count > max_speakers_hint:
        return GateResult(
            name="diarisation_sanity",
            passed=False,
            severity=Severity.WARN,
            message=f"{speaker_count} speakers detected; hint capped at {max_speakers_hint}. Likely oversegmentation.",
            metrics=metrics,
        )

    if speaker_share:
        max_share = max(speaker_share.values())
        if max_share > monopoly_threshold:
            return GateResult(
                name="diarisation_sanity",
                passed=False,
                severity=Severity.WARN,
                message=f"One speaker holds {max_share:.1%} of voiced time — diarisation may have collapsed.",
                metrics=metrics,
            )

        spurious = [s for s, v in speaker_share.items() if v < spurious_threshold]
        if spurious:
            return GateResult(
                name="diarisation_sanity",
                passed=True,
                severity=Severity.WARN,
                message=f"Spurious speaker cluster(s) holding <{spurious_threshold:.1%} each: {spurious}. Probably noise mis-clustered.",
                metrics=metrics,
            )

    return GateResult(
        name="diarisation_sanity",
        passed=True,
        severity=Severity.INFO,
        message=f"{speaker_count} speakers with sensible share distribution.",
        metrics=metrics,
    )


# Gate 4 — ASR output sanity (WARN)
def asr_sanity_gate(
    transcript_chars: int,
    voiced_duration_s: float,
    detected_language: Optional[str],
    *,
    min_chars_per_voiced_minute: float = 10.0,
) -> GateResult:
    """Did Whisper produce something? Hallucination detector — empty output
    on long voiced audio is suspicious; massive output on tiny audio is too.
    """
    minutes = max(voiced_duration_s, 1.0) / 60.0
    chars_per_minute = transcript_chars / minutes
    metrics = {
        "transcript_chars": transcript_chars,
        "voiced_duration_s": voiced_duration_s,
        "chars_per_voiced_minute": chars_per_minute,
        "detected_language": detected_language,
    }

    if transcript_chars == 0:
        return GateResult(
            name="asr_sanity",
            passed=False,
            severity=Severity.WARN,
            message="Whisper produced empty transcript on non-empty audio — model issue or aggressive VAD.",
            metrics=metrics,
        )

    if chars_per_minute < min_chars_per_voiced_minute:
        return GateResult(
            name="asr_sanity",
            passed=False,
            severity=Severity.WARN,
            message=f"Only {chars_per_minute:.1f} chars/voiced-minute (expected ≥{min_chars_per_voiced_minute}). Likely undertranscription.",
            metrics=metrics,
        )

    return GateResult(
        name="asr_sanity",
        passed=True,
        severity=Severity.INFO,
        message=f"{transcript_chars} chars over {voiced_duration_s:.0f}s voiced ({chars_per_minute:.0f}/min), language={detected_language}.",
        metrics=metrics,
    )


# Gate 5 — Output sanity (WARN)
def output_sanity_gate(
    output_path: Path | str,
    expected_segment_count: int,
) -> GateResult:
    """File written, non-empty, segment count looks right."""
    output_path = Path(output_path)
    metrics = {
        "output_path": str(output_path),
        "expected_segments": expected_segment_count,
    }

    if not output_path.exists():
        return GateResult(
            name="output_sanity",
            passed=False,
            severity=Severity.WARN,
            message=f"Output file {output_path} does not exist after write.",
            metrics=metrics,
        )

    size = output_path.stat().st_size
    metrics["output_bytes"] = size
    if size == 0:
        return GateResult(
            name="output_sanity",
            passed=False,
            severity=Severity.WARN,
            message=f"Output file {output_path} is empty.",
            metrics=metrics,
        )

    if expected_segment_count == 0:
        return GateResult(
            name="output_sanity",
            passed=True,
            severity=Severity.WARN,
            message="Output written but transcript has 0 segments.",
            metrics=metrics,
        )

    return GateResult(
        name="output_sanity",
        passed=True,
        severity=Severity.INFO,
        message=f"Output written: {size} bytes, {expected_segment_count} segments.",
        metrics=metrics,
    )
