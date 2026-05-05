"""v2 pipeline — faster-whisper + pyannote.audio 3.3 with adaptive preprocessing
and validation gates between stages.

Replaces v1's TranscriptionManager and the abandoned WhisperX-anchored Pipeline.

Stage order:
  0. Ingest      — measure(audio) -> AudioMeasurement; gate 0
  1. Preprocess  — adaptive ffmpeg; re-measure; gate 1 (damage check)
  2. Voice gate  — voiced_fraction from measurement; gate 2 (block on silence)
  3. Diarise     — pyannote pipeline 3.1; gate 3 (warn on weird counts)
  4. ASR         — faster-whisper large-v3-turbo, vad_filter=True; gate 4
  5. Attribute   — per-segment max-overlap diarisation lookup
  6. Output      — caller writes via v2.output

Gates 0-2 are BLOCKING — failures raise PipelineGateFailure.
Gates 3-4 are WARN — failures log and continue.
Gate 5 is left to the caller (output writer).
"""

from __future__ import annotations

import gc
import logging
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from diarizer.config import Config
from diarizer.gates import (
    GateResult,
    Severity,
    asr_sanity_gate,
    diarisation_sanity_gate,
    ingest_gate,
    preprocess_damage_gate,
    voice_presence_gate,
)
from diarizer.preprocessing import AudioMeasurement, measure, preprocess

logger = logging.getLogger(__name__)


class PipelineGateFailure(RuntimeError):
    """Raised when a BLOCK-severity gate fails. Contains the failing GateResult."""

    def __init__(self, gate: GateResult):
        super().__init__(str(gate))
        self.gate = gate


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list[dict] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    source_path: str
    model_name: str
    language: Optional[str]
    segments: List[Segment] = field(default_factory=list)
    duration_seconds: Optional[float] = None
    speaker_count: int = 0
    gate_results: list[dict] = field(default_factory=list)
    measurement: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    """Stateless across calls; instantiate per process and reuse `transcribe()` per file."""

    def __init__(self, config: Config):
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio input not found: {audio_path}")

        gate_log: list[GateResult] = []

        # Stage 0 — Ingest measurement
        logger.info("Measuring %s", audio_path)
        pre_m = measure(audio_path)
        gate0 = ingest_gate(pre_m)
        gate_log.append(gate0)
        self._log_gate(gate0)
        if gate0.is_blocking_failure():
            raise PipelineGateFailure(gate0)

        # Stage 1 — Adaptive preprocess (always: 16k mono; conditional: filters)
        if self.config.preprocessing.enabled:
            processed_path = self._preprocess_to_temp(audio_path, pre_m)
            post_m = measure(processed_path)
            gate1 = preprocess_damage_gate(pre_m, post_m)
            gate_log.append(gate1)
            self._log_gate(gate1)
            if gate1.is_blocking_failure():
                raise PipelineGateFailure(gate1)
            working_audio = processed_path
            working_m = post_m
        else:
            logger.info("Preprocessing disabled by config; using input directly.")
            working_audio = audio_path
            working_m = pre_m

        # Stage 2 — Voice presence gate (block on near-silence)
        gate2 = voice_presence_gate(working_m.voiced_fraction)
        gate_log.append(gate2)
        self._log_gate(gate2)
        if gate2.is_blocking_failure():
            raise PipelineGateFailure(gate2)

        # Stage 3 — Diarisation (optional)
        speaker_timeline: list[tuple[float, float, str]] = []
        speaker_count = 0
        if self.config.diarisation.enabled:
            speaker_timeline = self._run_diarisation(working_audio)
            speaker_count = len({lbl for _, _, lbl in speaker_timeline})

            speaker_share = self._speaker_share(speaker_timeline)
            gate3 = diarisation_sanity_gate(
                speaker_count=speaker_count,
                speaker_share=speaker_share,
                min_speakers_hint=self.config.diarisation.min_speakers,
                max_speakers_hint=self.config.diarisation.max_speakers,
            )
            gate_log.append(gate3)
            self._log_gate(gate3)

        # Stage 4 — ASR (faster-whisper)
        segments, info = self._run_asr(working_audio)

        voiced_duration = working_m.duration_s * working_m.voiced_fraction
        transcript_chars = sum(len(s.text) for s in segments)
        gate4 = asr_sanity_gate(
            transcript_chars=transcript_chars,
            voiced_duration_s=voiced_duration,
            detected_language=info.get("language"),
        )
        gate_log.append(gate4)
        self._log_gate(gate4)

        # Stage 5 — Attribution
        if speaker_timeline:
            self._attribute(segments, speaker_timeline)

        return TranscriptionResult(
            source_path=str(audio_path),
            model_name=self.config.model.name,
            language=info.get("language"),
            segments=segments,
            duration_seconds=working_m.duration_s,
            speaker_count=speaker_count,
            gate_results=[self._gate_to_dict(g) for g in gate_log],
            measurement=working_m.to_dict(),
        )

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    def _preprocess_to_temp(self, audio_path: Path, measurement: AudioMeasurement) -> Path:
        temp_dir = Path(self.config.paths.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="v2_pre_", suffix=".wav", dir=temp_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        return preprocess(audio_path, tmp_path, self.config.preprocessing, measurement)

    def _run_diarisation(self, audio_path: Path) -> list[tuple[float, float, str]]:
        """Run pyannote.audio diarisation. Returns [(start, end, speaker_label), ...]."""
        from pyannote.audio import Pipeline as PyannotePipeline  # lazy

        device = self._resolve_device()
        token = self.config.resolved_hf_token()
        if not token:
            logger.warning("Diarisation enabled but no HF token resolved; skipping speaker assignment.")
            return []

        logger.info("Loading pyannote diarisation pipeline on %s", device)
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        try:
            self._move_pyannote_to_device(pipeline, device)

            kwargs = {}
            if self.config.diarisation.min_speakers is not None:
                kwargs["min_speakers"] = self.config.diarisation.min_speakers
            if self.config.diarisation.max_speakers is not None:
                kwargs["max_speakers"] = self.config.diarisation.max_speakers

            logger.info("Running diarisation %s", kwargs or "(no speaker hints)")
            diarization = pipeline(str(audio_path), **kwargs)
        finally:
            del pipeline
            self._cuda_cache_clear()

        timeline: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append((float(turn.start), float(turn.end), str(speaker)))
        timeline.sort(key=lambda t: t[0])
        return timeline

    def _run_asr(self, audio_path: Path) -> tuple[list[Segment], dict]:
        """Run faster-whisper transcription. Returns (segments, info_dict)."""
        from faster_whisper import WhisperModel  # lazy

        device = self._resolve_device()
        compute_type = self.config.model.compute_type

        try:
            model = self._load_whisper(device, compute_type)
        except Exception as e:  # noqa: BLE001 — broad to catch backend OOM types
            if "memory" in str(e).lower() or "OOM" in str(e):
                logger.warning("ASR OOM with compute_type=%s; retrying int8.", compute_type)
                model = self._load_whisper(device, "int8")
            else:
                raise

        try:
            initial_prompt = self.config.resolved_initial_prompt()
            beam_size = self.config.model.beam_size
            condition_on_previous_text = self.config.model.condition_on_previous_text
            logger.info(
                "Transcribing %s (beam_size=%d, condition_on_previous_text=%s, initial_prompt=%s)",
                audio_path, beam_size, condition_on_previous_text,
                "set" if initial_prompt else "none",
            )
            segs_iter, info = model.transcribe(
                str(audio_path),
                beam_size=beam_size,
                vad_filter=True,                       # built-in Silero — silence trim before model sees it
                word_timestamps=True,
                condition_on_previous_text=condition_on_previous_text,
                # Hallucination chain break is handled by faster-whisper's existing
                # prompt_reset_on_temperature=0.5 default + vad_filter trim, so
                # condition_on_previous_text=True is safer than the literature suggests
                # for continuously-voiced meeting audio.
                initial_prompt=initial_prompt,
                language=self.config.model.language,
            )
            language = info.language

            segments: list[Segment] = []
            for s in segs_iter:
                words = []
                if getattr(s, "words", None):
                    for w in s.words:
                        words.append({
                            "start": float(w.start) if w.start is not None else None,
                            "end": float(w.end) if w.end is not None else None,
                            "text": w.word,
                            "probability": float(w.probability) if w.probability is not None else None,
                        })
                segments.append(Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=s.text.strip(),
                    speaker=None,
                    words=words,
                ))
            return segments, {"language": language}
        finally:
            del model
            self._cuda_cache_clear()

    def _attribute(self, segments: list[Segment], timeline: list[tuple[float, float, str]]) -> None:
        """Assign speaker label to each ASR segment by max-overlap with the diariser timeline.

        Sets `segment.speaker` in place. Falls back to "Unknown Speaker" if no
        diariser turn overlaps the segment.
        """
        for seg in segments:
            best_label: Optional[str] = None
            best_overlap = 0.0
            for d_start, d_end, label in timeline:
                if d_end <= seg.start:
                    continue
                if d_start >= seg.end:
                    break  # timeline is sorted; no later turns can overlap
                overlap = min(seg.end, d_end) - max(seg.start, d_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label
            seg.speaker = best_label or "Unknown Speaker"

    # ------------------------------------------------------------------
    # Backends, devices, memory hygiene
    # ------------------------------------------------------------------
    def _load_whisper(self, device: str, compute_type: str):
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper '%s' on %s (compute_type=%s)",
                    self.config.model.name, device, compute_type)
        return WhisperModel(
            self.config.model.name,
            device=device,
            compute_type=compute_type,
        )

    def _resolve_device(self) -> str:
        device = self.config.model.device
        if device != "cuda":
            return device
        if not self.config.model.cpu_fallback:
            return "cuda"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        logger.warning("CUDA unavailable; falling back to CPU.")
        return "cpu"

    def _move_pyannote_to_device(self, pipeline, device: str) -> None:
        """pyannote.audio.Pipeline doesn't have an obvious .to(device); do it via torch."""
        if device != "cuda":
            return
        try:
            import torch
            pipeline.to(torch.device("cuda"))
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not move pyannote pipeline to CUDA (%s); continuing on CPU.", e)

    @staticmethod
    def _cuda_cache_clear() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _speaker_share(timeline: list[tuple[float, float, str]]) -> dict[str, float]:
        if not timeline:
            return {}
        durations: dict[str, float] = defaultdict(float)
        total = 0.0
        for s, e, lbl in timeline:
            d = e - s
            durations[lbl] += d
            total += d
        if total <= 0:
            return {}
        return {lbl: dur / total for lbl, dur in durations.items()}

    @staticmethod
    def _gate_to_dict(g: GateResult) -> dict:
        return {
            "name": g.name,
            "passed": g.passed,
            "severity": g.severity.value,
            "message": g.message,
            "metrics": g.metrics,
        }

    @staticmethod
    def _log_gate(g: GateResult) -> None:
        if g.passed and g.severity == Severity.INFO:
            logger.info("Gate %s: %s", g.name, g.message)
        elif g.severity == Severity.WARN:
            logger.warning("Gate %s: %s", g.name, g.message)
        elif g.severity == Severity.BLOCK and not g.passed:
            logger.error("Gate %s BLOCK: %s", g.name, g.message)
        else:
            logger.info("Gate %s: %s", g.name, g.message)
