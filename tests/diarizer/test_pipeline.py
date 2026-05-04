"""Tests for v2/pipeline.py — orchestration only, real backends mocked.

We mock at three layers:
  1. v2.pipeline.measure — return a hand-crafted AudioMeasurement.
  2. sys.modules['faster_whisper'] — fake WhisperModel.
  3. sys.modules['pyannote.audio'] — fake Pipeline.from_pretrained.

The pure attribution math (_attribute) is also exercised directly with
hand-built timelines, no mocking needed.

Real-model end-to-end testing lives in PLAN 240 Step 10 (smoke run on
recordings/*.m4a).
"""

from __future__ import annotations

import sys
import types

import pytest

from diarizer.config import Config
from diarizer.pipeline import Pipeline, PipelineGateFailure, Segment, TranscriptionResult
from diarizer.preprocessing import AudioMeasurement


def _measurement(
    *,
    duration_s=300.0,
    sample_rate=16000,
    channels=1,
    rms_dbfs=-25.0,
    peak_dbfs=-1.0,
    noise_floor_dbfs=-50.0,
    speech_floor_dbfs=-15.0,
    voiced_fraction=0.85,
    spectral_centroid_hz=2500.0,
) -> AudioMeasurement:
    return AudioMeasurement(
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        noise_floor_dbfs=noise_floor_dbfs,
        speech_floor_dbfs=speech_floor_dbfs,
        voiced_fraction=voiced_fraction,
        spectral_centroid_hz=spectral_centroid_hz,
    )


@pytest.fixture
def fake_audio_file(tmp_path):
    """Existence-only fixture — measure() is mocked, so contents don't matter."""
    p = tmp_path / "fake.wav"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
    return p


@pytest.fixture
def patch_measure(monkeypatch):
    """Patch v2.pipeline.measure to return whatever the test wants."""
    state = {"measurement": _measurement(), "post_measurement": None}

    def fake_measure(path):
        # First call (input) → state["measurement"]
        # Subsequent calls (post-preprocess) → state["post_measurement"] or same
        if state["post_measurement"] is None:
            return state["measurement"]
        m, state["post_measurement"] = state["post_measurement"], None
        return m

    monkeypatch.setattr("diarizer.pipeline.measure", fake_measure)
    return state


@pytest.fixture
def patch_preprocess(monkeypatch):
    """Patch v2.pipeline.preprocess so we never actually call ffmpeg."""
    def fake_preprocess(input_path, output_path, config, measurement=None):
        # Pretend the output exists at output_path; pipeline reads metadata only.
        from pathlib import Path
        Path(output_path).write_bytes(b"fake processed wav")
        return Path(output_path)

    monkeypatch.setattr("diarizer.pipeline.preprocess", fake_preprocess)


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Inject a fake faster_whisper module with WhisperModel."""
    fake = types.ModuleType("faster_whisper")
    state = {"oom_first_load": False, "load_count": 0, "compute_types_used": []}

    class FakeWord:
        def __init__(self, start, end, word, probability):
            self.start = start
            self.end = end
            self.word = word
            self.probability = probability

    class FakeSegment:
        def __init__(self, start, end, text, words=None):
            self.start = start
            self.end = end
            self.text = text
            self.words = words or []

    class FakeInfo:
        def __init__(self, language="en"):
            self.language = language

    class FakeWhisperModel:
        def __init__(self, name, device, compute_type):
            state["load_count"] += 1
            state["compute_types_used"].append(compute_type)
            if state["oom_first_load"] and state["load_count"] == 1:
                raise RuntimeError("CUDA out of memory (simulated OOM)")
            self.name = name
            self.device = device
            self.compute_type = compute_type

        def transcribe(self, audio_path, **kwargs):
            seg1 = FakeSegment(
                start=0.0, end=2.5,
                text="Hello, this is a test segment.",
                words=[FakeWord(0.0, 0.4, "Hello,", 0.95), FakeWord(0.4, 1.0, "this", 0.9)],
            )
            seg2 = FakeSegment(start=2.5, end=4.8, text="And here is a reply.")
            seg3 = FakeSegment(start=4.8, end=6.0, text="Final segment.")
            return iter([seg1, seg2, seg3]), FakeInfo("en")

    fake.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return state


@pytest.fixture
def fake_pyannote(monkeypatch):
    """Inject a fake pyannote.audio module with Pipeline.from_pretrained."""
    fake = types.ModuleType("pyannote.audio")
    state = {"called": False, "kwargs": None}

    class FakeTurn:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class FakeDiarization:
        def __init__(self, turns):
            self._turns = turns

        def itertracks(self, yield_label=True):
            for turn, label in self._turns:
                yield turn, "_", label

    class FakePyannotePipeline:
        @classmethod
        def from_pretrained(cls, name, use_auth_token):
            assert use_auth_token, "must receive an HF token"
            return cls()

        def to(self, device):
            return self

        def __call__(self, audio_path, **kwargs):
            state["called"] = True
            state["kwargs"] = kwargs
            return FakeDiarization([
                (FakeTurn(0.0, 2.5), "SPEAKER_00"),
                (FakeTurn(2.5, 4.8), "SPEAKER_01"),
                (FakeTurn(4.8, 6.0), "SPEAKER_00"),
            ])

    fake.Pipeline = FakePyannotePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake)
    return state


# ============================================================
# transcribe — happy path
# ============================================================
def test_transcribe_full_pipeline(fake_audio_file, patch_measure, patch_preprocess,
                                   fake_faster_whisper, fake_pyannote, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "fake_token")

    config = Config()
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(fake_audio_file)

    assert isinstance(result, TranscriptionResult)
    assert result.language == "en"
    assert len(result.segments) == 3
    assert result.speaker_count == 2
    # Attribution should have flowed through
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_01"
    assert result.segments[2].speaker == "SPEAKER_00"


def test_transcribe_no_diarisation(fake_audio_file, patch_measure, patch_preprocess,
                                    fake_faster_whisper, fake_pyannote, tmp_path):
    config = Config()
    config.diarisation.enabled = False
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(fake_audio_file)

    assert len(result.segments) == 3
    assert result.speaker_count == 0
    assert all(s.speaker is None for s in result.segments)


def test_transcribe_no_preprocess(fake_audio_file, patch_measure,
                                   fake_faster_whisper, fake_pyannote, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "fake_token")

    config = Config()
    config.preprocessing.enabled = False
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(fake_audio_file)

    assert len(result.segments) == 3
    # Preprocess gate should not appear in gate_results when disabled
    gate_names = {g["name"] for g in result.gate_results}
    assert "preprocess_damage" not in gate_names


def test_transcribe_diarisation_skipped_without_token(fake_audio_file, patch_measure, patch_preprocess,
                                                       fake_faster_whisper, fake_pyannote, monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    config = Config()
    config.diarisation.enabled = True
    config.auth.hf_token = None
    config.auth.token_path = None
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(fake_audio_file)

    # No token → diarisation skipped → segments untagged
    assert all(s.speaker is None for s in result.segments)


# ============================================================
# Blocking gate failures
# ============================================================
def test_transcribe_blocks_on_short_input(fake_audio_file, patch_measure, patch_preprocess,
                                           fake_faster_whisper, fake_pyannote, tmp_path):
    patch_measure["measurement"] = _measurement(duration_s=2.0)  # below 10s

    config = Config()
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)
    config.diarisation.enabled = False

    pipeline = Pipeline(config)
    with pytest.raises(PipelineGateFailure) as exc:
        pipeline.transcribe(fake_audio_file)
    assert exc.value.gate.name == "ingest"


def test_transcribe_blocks_on_silent_file(fake_audio_file, patch_measure, patch_preprocess,
                                           fake_faster_whisper, fake_pyannote, tmp_path):
    patch_measure["measurement"] = _measurement(rms_dbfs=-60.0)  # below floor

    config = Config()
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)
    config.diarisation.enabled = False

    pipeline = Pipeline(config)
    with pytest.raises(PipelineGateFailure) as exc:
        pipeline.transcribe(fake_audio_file)
    assert exc.value.gate.name == "ingest"


def test_transcribe_blocks_on_voice_absent(fake_audio_file, patch_measure, patch_preprocess,
                                            fake_faster_whisper, fake_pyannote, tmp_path):
    patch_measure["measurement"] = _measurement(voiced_fraction=0.01)

    config = Config()
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)
    config.preprocessing.enabled = False  # skip preprocess gate
    config.diarisation.enabled = False

    pipeline = Pipeline(config)
    with pytest.raises(PipelineGateFailure) as exc:
        pipeline.transcribe(fake_audio_file)
    assert exc.value.gate.name == "voice_presence"


# ============================================================
# OOM retry
# ============================================================
def test_transcribe_oom_retry_falls_back_to_int8(fake_audio_file, patch_measure, patch_preprocess,
                                                  fake_faster_whisper, fake_pyannote, tmp_path):
    fake_faster_whisper["oom_first_load"] = True

    config = Config()
    config.diarisation.enabled = False
    config.model.device = "cpu"
    config.model.compute_type = "int8_float16"
    config.paths.temp_dir = str(tmp_path)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(fake_audio_file)

    # Should have loaded twice — first failed, retry succeeded
    assert fake_faster_whisper["load_count"] == 2
    assert fake_faster_whisper["compute_types_used"] == ["int8_float16", "int8"]
    assert len(result.segments) == 3


# ============================================================
# Attribution math (no mocking — pure function on Pipeline)
# ============================================================
def test_attribute_max_overlap():
    config = Config()
    pipeline = Pipeline(config)

    segments = [
        Segment(start=0.0, end=2.0, text="A"),  # fully inside SPEAKER_00 (0-3)
        Segment(start=2.5, end=4.5, text="B"),  # spans 0.5 of SP_00 + 1.5 of SP_01 → SP_01 wins
        Segment(start=10.0, end=12.0, text="C"),  # no overlap
    ]
    timeline = [
        (0.0, 3.0, "SPEAKER_00"),
        (3.0, 5.0, "SPEAKER_01"),
    ]

    pipeline._attribute(segments, timeline)
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[1].speaker == "SPEAKER_01"
    assert segments[2].speaker == "Unknown Speaker"


def test_attribute_empty_timeline():
    config = Config()
    pipeline = Pipeline(config)
    segments = [Segment(start=0.0, end=2.0, text="hello")]
    pipeline._attribute(segments, [])
    # Empty timeline → no overlap → "Unknown Speaker"
    assert segments[0].speaker == "Unknown Speaker"


# ============================================================
# Speaker share math
# ============================================================
def test_speaker_share_distribution():
    timeline = [
        (0.0, 5.0, "A"),
        (5.0, 10.0, "B"),
        (10.0, 15.0, "A"),
    ]
    share = Pipeline._speaker_share(timeline)
    assert share["A"] == pytest.approx(2 / 3)
    assert share["B"] == pytest.approx(1 / 3)


def test_speaker_share_empty_timeline():
    assert Pipeline._speaker_share([]) == {}


# ============================================================
# Device resolution
# ============================================================
def test_resolve_device_cpu():
    config = Config()
    config.model.device = "cpu"
    pipeline = Pipeline(config)
    assert pipeline._resolve_device() == "cpu"


def test_resolve_device_cuda_no_fallback():
    config = Config()
    config.model.device = "cuda"
    config.model.cpu_fallback = False
    pipeline = Pipeline(config)
    # Returns 'cuda' regardless — caller responsible if not actually available
    assert pipeline._resolve_device() == "cuda"


# ============================================================
# Missing input
# ============================================================
def test_transcribe_missing_input_raises(tmp_path):
    config = Config()
    pipeline = Pipeline(config)
    with pytest.raises(FileNotFoundError):
        pipeline.transcribe(tmp_path / "nope.wav")


# ============================================================
# Result dict round-trip
# ============================================================
def test_transcription_result_to_dict(fake_audio_file, patch_measure, patch_preprocess,
                                       fake_faster_whisper, fake_pyannote, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "fake_token")
    config = Config()
    config.model.device = "cpu"
    config.paths.temp_dir = str(tmp_path)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(fake_audio_file)
    d = result.to_dict()

    assert d["model_name"] == "large-v3-turbo"
    assert d["language"] == "en"
    assert d["speaker_count"] == 2
    assert len(d["segments"]) == 3
    assert d["measurement"]["sample_rate"] == 16000
    assert isinstance(d["gate_results"], list)
