"""v2 CLI entrypoint. Run with: python -m v2.cli --input path/to/audio.mp4"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from diarizer.config import Config, load_config
from diarizer.output import write_output
from diarizer.pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diarizer",
        description="Diarizer — speaker diarisation + transcription (faster-whisper + pyannote.audio 3.3).",
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to input audio or video file.")
    p.add_argument("--config", "-c", type=Path, default=None, help="Path to YAML config (default: baked-in defaults).")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output path (default: <output_dir>/<input-stem>.<format>).")
    p.add_argument("--format", "-f", choices=["txt", "json", "srt"], default=None, help="Output format override.")
    p.add_argument("--model", default=None, help="Whisper model name override (e.g. large-v3-turbo, medium).")
    p.add_argument("--min-speakers", type=int, default=None, help="Minimum speakers hint for diarisation.")
    p.add_argument("--max-speakers", type=int, default=None, help="Maximum speakers hint for diarisation.")
    p.add_argument("--auth-token", default=None, help="Hugging Face token override (highest precedence).")
    p.add_argument("--no-preprocess", action="store_true", help="Skip FFmpeg preprocessing.")
    p.add_argument("--no-diarisation", action="store_true", help="Skip speaker diarisation; transcription only.")
    p.add_argument("--beam-size", type=int, default=None, help="Whisper beam size (default 1; try 5 for proper-noun + disfluency gain).")
    p.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Pass prior chunk's text to the next chunk (helps trailing-word fidelity; resets on temperature fallback).",
    )
    p.add_argument("--initial-prompt", default=None, help="Inline initial prompt (biases proper nouns and domain vocab).")
    p.add_argument("--initial-prompt-file", type=Path, default=None, help="File with initial prompt; overridden by --initial-prompt if both set.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.format is not None:
        config.output.format = args.format
    if args.model is not None:
        config.model.name = args.model
    if args.min_speakers is not None:
        config.diarisation.min_speakers = args.min_speakers
    if args.max_speakers is not None:
        config.diarisation.max_speakers = args.max_speakers
    if args.auth_token is not None:
        config.auth.hf_token = args.auth_token
    if args.no_preprocess:
        config.preprocessing.enabled = False
    if args.no_diarisation:
        config.diarisation.enabled = False
    if args.beam_size is not None:
        config.model.beam_size = args.beam_size
    if args.condition_on_previous_text:
        config.model.condition_on_previous_text = True
    if args.initial_prompt is not None:
        config.model.initial_prompt = args.initial_prompt
    if args.initial_prompt_file is not None:
        config.model.initial_prompt_path = str(args.initial_prompt_file)
    return config


def _resolve_output_path(args: argparse.Namespace, config: Config) -> Path:
    if args.output is not None:
        return args.output
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{args.input.stem}.{config.output.format}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    config = _apply_overrides(config, args)
    output_path = _resolve_output_path(args, config)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(args.input)
    write_output(result, output_path, config.output)

    # Plain ASCII so Windows cp1252 console doesn't UnicodeEncodeError on the arrow.
    print(f"Wrote {len(result.segments)} segments to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
