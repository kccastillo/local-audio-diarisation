"""v2 configuration — dataclass-backed, YAML-loaded, no singleton.

Each caller owns its own Config instance; pass it explicitly to Pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PathsConfig:
    recordings_dir: str = "recordings"
    output_dir: str = "output"
    temp_dir: str = "temp"
    logs_dir: str = "logs"


@dataclass
class ModelConfig:
    name: str = "large-v3-turbo"
    compute_type: str = "float16"
    device: str = "cuda"
    cpu_fallback: bool = True
    language: Optional[str] = None


@dataclass
class DiarisationConfig:
    enabled: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


@dataclass
class PreprocessingConfig:
    # Decode + resample is always done when enabled=True; the flags below gate filters.
    enabled: bool = True
    # Loudness normalisation: applied adaptively when measured RMS is too quiet (<-32 dBFS).
    normalise: bool = True
    # Denoising: default OFF — RESEARCH 230 confirms learned and FFT denoisers degrade
    # Whisper WER on already-clean audio. Set True to opt in for genuinely noisy inputs.
    denoise: bool = False
    target_sample_rate: int = 16000


@dataclass
class OutputConfig:
    format: str = "txt"
    include_timestamps: bool = True
    include_speakers: bool = True


@dataclass
class AuthConfig:
    hf_token: Optional[str] = None
    token_path: Optional[str] = None


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diarisation: DiarisationConfig = field(default_factory=DiarisationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    def resolved_hf_token(self) -> Optional[str]:
        if self.auth.hf_token:
            return self.auth.hf_token
        if self.auth.token_path:
            token_file = Path(self.auth.token_path)
            if token_file.exists():
                return token_file.read_text(encoding="utf-8").strip() or None
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def to_dict(self) -> dict:
        return asdict(self)


_SECTION_TO_DATACLASS = {
    "paths": PathsConfig,
    "model": ModelConfig,
    "diarisation": DiarisationConfig,
    "preprocessing": PreprocessingConfig,
    "output": OutputConfig,
    "auth": AuthConfig,
}


def load_config(path: Optional[Path | str] = None) -> Config:
    """Load v2 config from YAML, falling back to baked-in defaults.

    Partial YAML is supported — any section absent from the file uses the
    dataclass default. Unknown keys inside a known section raise TypeError
    (caller mistyped a field name).
    """
    if path is None:
        return Config()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sections = {}
    for section_name, section_cls in _SECTION_TO_DATACLASS.items():
        section_data = raw.get(section_name, {}) or {}
        sections[section_name] = section_cls(**section_data)

    return Config(**sections)
