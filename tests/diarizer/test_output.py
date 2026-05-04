"""Tests for v2/output.py — TXT, JSON, SRT writers on synthetic results."""

from __future__ import annotations

import json

import pytest

from diarizer.config import OutputConfig
from diarizer.output import _format_srt_timestamp, _format_timestamp, write_json, write_output, write_srt, write_txt
from diarizer.pipeline import Segment, TranscriptionResult


@pytest.fixture
def sample_result():
    return TranscriptionResult(
        source_path="/tmp/sample.wav",
        model_name="large-v3-turbo",
        language="en",
        segments=[
            Segment(start=0.0, end=2.5, text="Hello, this is the first segment.", speaker="SPEAKER_00"),
            Segment(start=2.5, end=4.8, text="And here is a reply.", speaker="SPEAKER_01"),
            Segment(start=4.8, end=6.0, text="Unattributed line.", speaker=None),
        ],
        duration_seconds=6.0,
    )


def test_format_timestamp():
    assert _format_timestamp(0) == "00:00:00"
    assert _format_timestamp(65.4) == "00:01:05"
    assert _format_timestamp(3661) == "01:01:01"
    assert _format_timestamp(-1) == "00:00:00"


def test_format_srt_timestamp():
    assert _format_srt_timestamp(0) == "00:00:00,000"
    assert _format_srt_timestamp(1.5) == "00:00:01,500"
    assert _format_srt_timestamp(3661.999) == "01:01:01,999"


def test_write_txt_with_speakers_and_timestamps(tmp_path, sample_result):
    out = write_txt(sample_result, tmp_path / "out.txt", OutputConfig(format="txt"))
    text = out.read_text(encoding="utf-8")
    assert "[SPEAKER_00] 00:00:00 Hello, this is the first segment." in text
    assert "[SPEAKER_01] 00:00:02 And here is a reply." in text
    assert "Unattributed line." in text


def test_write_txt_no_speakers(tmp_path, sample_result):
    cfg = OutputConfig(format="txt", include_speakers=False, include_timestamps=False)
    out = write_txt(sample_result, tmp_path / "out.txt", cfg)
    text = out.read_text(encoding="utf-8")
    assert "[SPEAKER_00]" not in text
    assert "00:00:00" not in text
    assert "Hello, this is the first segment." in text


def test_write_json_roundtrip(tmp_path, sample_result):
    out = write_json(sample_result, tmp_path / "out.json", OutputConfig(format="json"))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["model_name"] == "large-v3-turbo"
    assert len(payload["segments"]) == 3
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["segments"][2]["speaker"] is None


def test_write_srt(tmp_path, sample_result):
    out = write_srt(sample_result, tmp_path / "out.srt", OutputConfig(format="srt"))
    text = out.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,500\n[SPEAKER_00] Hello, this is the first segment." in text
    assert "2\n00:00:02,500 --> 00:00:04,800\n[SPEAKER_01] And here is a reply." in text
    assert "3\n00:00:04,800 --> 00:00:06,000\nUnattributed line." in text


def test_write_output_dispatch(tmp_path, sample_result):
    write_output(sample_result, tmp_path / "out.txt", OutputConfig(format="txt"))
    write_output(sample_result, tmp_path / "out.json", OutputConfig(format="json"))
    write_output(sample_result, tmp_path / "out.srt", OutputConfig(format="srt"))
    assert (tmp_path / "out.txt").exists()
    assert (tmp_path / "out.json").exists()
    assert (tmp_path / "out.srt").exists()


def test_write_output_unknown_format(tmp_path, sample_result):
    with pytest.raises(ValueError):
        write_output(sample_result, tmp_path / "out.bin", OutputConfig(format="bin"))


def test_write_output_creates_parent_dir(tmp_path, sample_result):
    deep = tmp_path / "a" / "b" / "c" / "out.txt"
    write_output(sample_result, deep, OutputConfig(format="txt"))
    assert deep.exists()
