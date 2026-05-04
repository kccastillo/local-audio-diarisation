"""Tests for v2/preprocessing.py — ffmpeg invocation contract, error handling.

These tests mock subprocess.run and shutil.which so they run without ffmpeg
installed and without touching real audio. Real-audio preprocessing is
exercised by the hardware-validation step in PLAN 202605040300.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from diarizer.config import PreprocessingConfig
from diarizer.preprocessing import FFmpegError, preprocess_audio


@pytest.fixture
def fake_input(tmp_path):
    p = tmp_path / "input.wav"
    p.write_bytes(b"RIFF....fake")
    return p


@pytest.fixture
def fake_output(tmp_path):
    return tmp_path / "out" / "preprocessed.wav"


def _patch_ffmpeg(mocker, returncode=0, stderr=""):
    """Patch shutil.which to report ffmpeg present and subprocess.run to succeed."""
    mocker.patch("diarizer.preprocessing.shutil.which", return_value="/usr/bin/ffmpeg")
    completed = mocker.Mock()
    completed.returncode = returncode
    completed.stderr = stderr
    completed.stdout = ""
    run_mock = mocker.patch("diarizer.preprocessing.subprocess.run", return_value=completed)
    return run_mock


def test_preprocess_with_all_filters(mocker, fake_input, fake_output):
    run_mock = _patch_ffmpeg(mocker)
    cfg = PreprocessingConfig(enabled=True, denoise=True, normalise=True, target_sample_rate=16000)

    result = preprocess_audio(fake_input, fake_output, cfg)

    assert result == fake_output
    assert fake_output.parent.exists()
    cmd = run_mock.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and str(fake_input) in cmd
    assert "-ac" in cmd and "1" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "-af" in cmd
    af_index = cmd.index("-af")
    filters = cmd[af_index + 1]
    assert "afftdn" in filters
    assert "loudnorm" in filters


def test_preprocess_no_filters_when_disabled(mocker, fake_input, fake_output):
    run_mock = _patch_ffmpeg(mocker)
    cfg = PreprocessingConfig(enabled=False, denoise=True, normalise=True)

    preprocess_audio(fake_input, fake_output, cfg)

    cmd = run_mock.call_args[0][0]
    assert "-af" not in cmd


def test_preprocess_denoise_only(mocker, fake_input, fake_output):
    run_mock = _patch_ffmpeg(mocker)
    cfg = PreprocessingConfig(enabled=True, denoise=True, normalise=False)

    preprocess_audio(fake_input, fake_output, cfg)

    cmd = run_mock.call_args[0][0]
    af_index = cmd.index("-af")
    filters = cmd[af_index + 1]
    assert "afftdn" in filters
    assert "loudnorm" not in filters


def test_preprocess_normalise_only(mocker, fake_input, fake_output):
    run_mock = _patch_ffmpeg(mocker)
    cfg = PreprocessingConfig(enabled=True, denoise=False, normalise=True)

    preprocess_audio(fake_input, fake_output, cfg)

    cmd = run_mock.call_args[0][0]
    af_index = cmd.index("-af")
    filters = cmd[af_index + 1]
    assert "loudnorm" in filters
    assert "afftdn" not in filters


def test_preprocess_custom_sample_rate(mocker, fake_input, fake_output):
    run_mock = _patch_ffmpeg(mocker)
    cfg = PreprocessingConfig(enabled=True, denoise=False, normalise=False, target_sample_rate=8000)

    preprocess_audio(fake_input, fake_output, cfg)

    cmd = run_mock.call_args[0][0]
    ar_index = cmd.index("-ar")
    assert cmd[ar_index + 1] == "8000"


def test_preprocess_missing_ffmpeg_raises(mocker, fake_input, fake_output):
    mocker.patch("diarizer.preprocessing.shutil.which", return_value=None)
    cfg = PreprocessingConfig()

    with pytest.raises(FFmpegError, match="ffmpeg not found"):
        preprocess_audio(fake_input, fake_output, cfg)


def test_preprocess_ffmpeg_failure_raises(mocker, fake_input, fake_output):
    _patch_ffmpeg(mocker, returncode=1, stderr="some ffmpeg complaint")
    cfg = PreprocessingConfig()

    with pytest.raises(FFmpegError, match="ffmpeg failed"):
        preprocess_audio(fake_input, fake_output, cfg)


def test_preprocess_missing_input_raises(mocker, tmp_path):
    _patch_ffmpeg(mocker)
    cfg = PreprocessingConfig()

    with pytest.raises(FileNotFoundError):
        preprocess_audio(tmp_path / "nope.wav", tmp_path / "out.wav", cfg)


def test_preprocess_creates_output_parent_dir(mocker, fake_input, tmp_path):
    _patch_ffmpeg(mocker)
    deep = tmp_path / "a" / "b" / "c" / "out.wav"
    cfg = PreprocessingConfig()

    preprocess_audio(fake_input, deep, cfg)

    assert deep.parent.exists()
