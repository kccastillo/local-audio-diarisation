"""Tests for v2/config.py — defaults, YAML override, partial override, env-token resolution."""

from __future__ import annotations

import pytest
import yaml

from diarizer.config import Config, load_config


def test_defaults_no_path():
    config = load_config(None)
    assert config.model.name == "large-v3-turbo"
    assert config.model.compute_type == "float16"
    assert config.diarisation.enabled is True
    assert config.preprocessing.enabled is True
    assert config.output.format == "txt"
    assert config.auth.hf_token is None
    # ASR-quality knobs default to the verified-baseline behaviour
    assert config.model.beam_size == 1
    assert config.model.condition_on_previous_text is False
    assert config.model.initial_prompt is None
    assert config.model.initial_prompt_path is None
    assert config.resolved_initial_prompt() is None


def test_resolved_initial_prompt_inline():
    config = Config()
    config.model.initial_prompt = "Cybersecurity meeting"
    assert config.resolved_initial_prompt() == "Cybersecurity meeting"


def test_resolved_initial_prompt_from_file(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("OT/CSM meeting; names: Mary, Joseph", encoding="utf-8")
    config = Config()
    config.model.initial_prompt_path = str(prompt_file)
    assert config.resolved_initial_prompt() == "OT/CSM meeting; names: Mary, Joseph"


def test_resolved_initial_prompt_inline_beats_path(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from file", encoding="utf-8")
    config = Config()
    config.model.initial_prompt = "from inline"
    config.model.initial_prompt_path = str(prompt_file)
    assert config.resolved_initial_prompt() == "from inline"


def test_resolved_initial_prompt_path_missing_file_returns_none():
    config = Config()
    config.model.initial_prompt_path = "/does/not/exist.txt"
    assert config.resolved_initial_prompt() is None


def test_yaml_override_new_asr_knobs(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model": {
            "beam_size": 5,
            "condition_on_previous_text": True,
            "initial_prompt": "Meeting about X",
        },
    }), encoding="utf-8")

    config = load_config(cfg_path)
    assert config.model.beam_size == 5
    assert config.model.condition_on_previous_text is True
    assert config.resolved_initial_prompt() == "Meeting about X"


def test_yaml_override(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model": {"name": "medium", "compute_type": "int8"},
        "output": {"format": "srt"},
    }), encoding="utf-8")

    config = load_config(cfg_path)
    assert config.model.name == "medium"
    assert config.model.compute_type == "int8"
    assert config.output.format == "srt"
    # untouched section still has defaults
    assert config.preprocessing.enabled is True
    assert config.diarisation.enabled is True


def test_partial_section_override(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "diarisation": {"min_speakers": 2, "max_speakers": 5},
    }), encoding="utf-8")

    config = load_config(cfg_path)
    assert config.diarisation.enabled is True   # default preserved
    assert config.diarisation.min_speakers == 2
    assert config.diarisation.max_speakers == 5


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_unknown_field_raises(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model": {"name": "medium", "totally_made_up_field": "x"},
    }), encoding="utf-8")

    with pytest.raises(TypeError):
        load_config(cfg_path)


def test_resolved_hf_token_explicit():
    config = Config()
    config.auth.hf_token = "explicit_value"
    assert config.resolved_hf_token() == "explicit_value"


def test_resolved_hf_token_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_value")
    config = Config()
    assert config.resolved_hf_token() == "env_value"


def test_resolved_hf_token_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    config = Config()
    assert config.resolved_hf_token() is None


def test_to_dict_roundtrip():
    config = Config()
    d = config.to_dict()
    assert "model" in d and d["model"]["name"] == "large-v3-turbo"
    assert "diarisation" in d and d["diarisation"]["enabled"] is True
