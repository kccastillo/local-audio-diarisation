"""ASR config panel — same audio, multiple Whisper config knobs, single comparable
output set. Reuses preprocessing + diarisation across runs (those are deterministic
and don't change with the swept configs).

Usage:
  venv/Scripts/python.exe scripts/run_asr_panel.py \
      --input "recordings/meeting 28 Apr at 1-01 pm.m4a" \
      --auth-token <hf> \
      --prompt-file prompts/app-control.txt \
      --output-dir output/panel/app-control \
      [--min-speakers 3 --max-speakers 3]

Configs run by default:
  A_baseline                 beam=1, cond=False, prompt=None
  B_prompt                   beam=1, cond=False, prompt=set
  C_beam5                    beam=5, cond=False, prompt=None
  D_beam5_prompt             beam=5, cond=False, prompt=set
  E_cond                     beam=1, cond=True,  prompt=None
  F_cond_prompt              beam=1, cond=True,  prompt=set
  G_full_stack               beam=5, cond=True,  prompt=set
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diarizer.config import Config, OutputConfig
from diarizer.output import write_output
from diarizer.pipeline import Pipeline, Segment, TranscriptionResult
from diarizer.preprocessing import measure, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("panel")


@dataclass
class PanelConfig:
    name: str
    beam_size: int
    condition_on_previous_text: bool
    use_prompt: bool


PANEL = [
    PanelConfig("A_baseline",       beam_size=1, condition_on_previous_text=False, use_prompt=False),
    PanelConfig("B_prompt",         beam_size=1, condition_on_previous_text=False, use_prompt=True),
    PanelConfig("C_beam5",          beam_size=5, condition_on_previous_text=False, use_prompt=False),
    PanelConfig("D_beam5_prompt",   beam_size=5, condition_on_previous_text=False, use_prompt=True),
    PanelConfig("E_cond",           beam_size=1, condition_on_previous_text=True,  use_prompt=False),
    PanelConfig("F_cond_prompt",    beam_size=1, condition_on_previous_text=True,  use_prompt=True),
    PanelConfig("G_full_stack",     beam_size=5, condition_on_previous_text=True,  use_prompt=True),
]


def _resolve_token(arg_token: Optional[str], token_path: Optional[Path]) -> Optional[str]:
    if arg_token:
        return arg_token
    if token_path and token_path.exists():
        return token_path.read_text(encoding="utf-8").strip() or None
    return None


def _attribute(segments: list[Segment], timeline: list[tuple[float, float, str]]) -> None:
    for seg in segments:
        best_label: Optional[str] = None
        best_overlap = 0.0
        for d_start, d_end, label in timeline:
            if d_end <= seg.start:
                continue
            if d_start >= seg.end:
                break
            overlap = min(seg.end, d_end) - max(seg.start, d_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        seg.speaker = best_label or "Unknown Speaker"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--auth-token", default=None)
    p.add_argument("--token-file", type=Path, default=Path("D:/Projects/Tokens/kc_diariser.txt"))
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--temp-dir", type=Path, default=Path("temp"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.prompt_file.exists():
        print(f"prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 2

    token = _resolve_token(args.auth_token, args.token_file)
    if not token:
        print("HF token not resolved. Pass --auth-token or set --token-file.", file=sys.stderr)
        return 2

    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    logger.info("Initial prompt loaded (%d chars).", len(prompt_text))

    # ----- once per audio: preprocess + diarise + load Whisper model -----
    cfg = Config()
    cfg.diarisation.min_speakers = args.min_speakers
    cfg.diarisation.max_speakers = args.max_speakers
    cfg.auth.hf_token = token
    cfg.paths.temp_dir = str(args.temp_dir)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    logger.info("Measuring %s", args.input)
    pre_m = measure(args.input)
    timings["measure_in_s"] = time.perf_counter() - t0
    logger.info("Pre measurement: %s", pre_m.to_dict())

    with tempfile.NamedTemporaryFile(prefix="panel_pre_", suffix=".wav", dir=args.temp_dir, delete=False) as tmp:
        processed_wav = Path(tmp.name)

    t0 = time.perf_counter()
    logger.info("Preprocessing → %s", processed_wav)
    preprocess(args.input, processed_wav, cfg.preprocessing, pre_m)
    timings["preprocess_s"] = time.perf_counter() - t0

    pipe = Pipeline(cfg)
    t0 = time.perf_counter()
    logger.info("Running diarisation once (cached for all configs)…")
    timeline = pipe._run_diarisation(processed_wav)
    timings["diarisation_s"] = time.perf_counter() - t0
    logger.info("Diarisation: %d turns, %d distinct speakers.", len(timeline), len({lbl for _,_,lbl in timeline}))

    # Load Whisper once; vary kwargs per transcribe() call.
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    logger.info("Loading faster-whisper '%s' on %s (compute_type=%s)", cfg.model.name, device, cfg.model.compute_type)
    whisper = WhisperModel(cfg.model.name, device=device, compute_type=cfg.model.compute_type)
    timings["whisper_load_s"] = time.perf_counter() - t0

    # ----- per-config ASR + attribution + output -----
    out_cfg = OutputConfig(format="txt")
    summary_rows = []
    for pc in PANEL:
        prompt = prompt_text if pc.use_prompt else None
        logger.info(
            "[%s] beam=%d cond=%s prompt=%s",
            pc.name, pc.beam_size, pc.condition_on_previous_text, "set" if prompt else "none",
        )
        t0 = time.perf_counter()
        segs_iter, info = whisper.transcribe(
            str(processed_wav),
            beam_size=pc.beam_size,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=pc.condition_on_previous_text,
            initial_prompt=prompt,
            language=cfg.model.language,
        )
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
        elapsed = time.perf_counter() - t0
        _attribute(segments, timeline)

        result = TranscriptionResult(
            source_path=str(args.input),
            model_name=cfg.model.name,
            language=info.language,
            segments=segments,
            duration_seconds=pre_m.duration_s,
            speaker_count=len({s.speaker for s in segments if s.speaker and s.speaker != "Unknown Speaker"}),
        )
        out_path = args.output_dir / f"{pc.name}.txt"
        write_output(result, out_path, out_cfg)
        summary_rows.append((pc.name, elapsed, len(segments), sum(len(s.text) for s in segments)))
        logger.info("[%s] wrote %d segments in %.1f s → %s", pc.name, len(segments), elapsed, out_path)

    # cleanup model
    del whisper
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # summary
    print()
    print(f"{'config':<18} {'wall_s':>8} {'segments':>9} {'chars':>7}")
    print("-" * 46)
    for name, elapsed, n_seg, n_chars in summary_rows:
        print(f"{name:<18} {elapsed:>8.1f} {n_seg:>9} {n_chars:>7}")
    print()
    print("Setup timings (one-shot, shared across all configs):")
    for k, v in timings.items():
        print(f"  {k:<18} {v:.1f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
