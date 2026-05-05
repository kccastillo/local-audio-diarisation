"""Diarizer CLI. Two subcommands:

  diarizer run --input <file>          # current pipeline behaviour
  diarizer serve <session-dir>         # transcript-review webapp on 127.0.0.1

Back-compat: invocations with no subcommand but `--input` present (the legacy
flat-arg form, e.g. `diarizer --input foo.wav`) are routed to `run`.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import webbrowser
from pathlib import Path

from diarizer.config import Config, load_config
from diarizer.output import write_opus, write_output
from diarizer.pipeline import Pipeline
from diarizer.session import create_session_dir


SUBCOMMANDS = {"run", "serve"}


def _add_run_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--beam-size", type=int, default=None, help="Whisper beam size.")
    p.add_argument("--condition-on-previous-text", action="store_true",
                   help="Pass prior chunk's text to the next chunk.")
    p.add_argument("--initial-prompt", default=None, help="Inline initial prompt.")
    p.add_argument("--initial-prompt-file", type=Path, default=None, help="File with initial prompt.")
    p.add_argument("--no-session-dir", action="store_true",
                   help="Skip session-dir creation (legacy single-file output only).")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diarizer",
        description="Diarizer — speaker diarisation + transcription (faster-whisper + pyannote.audio 3.3).",
    )
    sub = p.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run the pipeline on a single audio/video file.")
    _add_run_args(run_p)

    serve_p = sub.add_parser("serve", help="Boot the transcript-review webapp on 127.0.0.1.")
    serve_p.add_argument("session_dir", type=Path, help="Path to a session directory produced by `run`.")
    serve_p.add_argument("--port", type=int, default=8765, help="Loopback port (default 8765).")
    serve_p.add_argument("--no-browser", action="store_true", help="Don't auto-open the default browser.")
    return p


def _is_legacy_invocation(argv: list[str]) -> bool:
    """Return True if argv looks like the legacy flat-arg form (no subcommand, has --input)."""
    if not argv:
        return False
    if argv[0] in SUBCOMMANDS:
        return False
    return any(a in ("--input", "-i") or a.startswith("--input=") or a.startswith("-i=") for a in argv)


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


def _run(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    config = _apply_overrides(config, args)
    output_path = _resolve_output_path(args, config)

    pipeline = Pipeline(config)
    result = pipeline.transcribe(args.input)
    write_output(result, output_path, config.output)
    print(f"Wrote {len(result.segments)} segments to {output_path}")

    if args.no_session_dir:
        return 0

    # Session dir finalisation: Opus playback artefact + transcript copy.
    output_dir = Path(config.paths.output_dir)
    session_dir = create_session_dir(output_dir, args.input)
    try:
        write_opus(args.input, session_dir / "source.opus")
    except Exception as e:
        print(f"Warning: Opus export failed: {e}", file=sys.stderr)
    # Always also produce a JSON copy beside output_path so transcript.json exists.
    legacy_json = output_dir / f"{args.input.stem}.json"
    if config.output.format != "json":
        # Force a JSON dump for the webapp.
        from diarizer.config import OutputConfig
        json_cfg = OutputConfig(**{**config.output.__dict__})
        json_cfg.format = "json"
        write_output(result, legacy_json, json_cfg)
    if legacy_json.exists():
        shutil.copyfile(legacy_json, session_dir / "transcript.json")
    print(f"Session dir: {session_dir}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    sd = args.session_dir
    required = ["session.json", "source.opus", "transcript.json"]
    missing = [n for n in required if not (sd / n).exists()]
    if missing:
        print(f"Session dir {sd} is missing: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. `pip install uvicorn fastapi` to use `serve`.", file=sys.stderr)
        return 3

    from diarizer.webapp.app import create_app
    app = create_app(sd)
    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"Serving {sd} at {url}  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    # Back-compat: legacy `diarizer --input foo.wav` → route to `run`.
    if _is_legacy_invocation(raw):
        raw = ["run", *raw]

    if not raw:
        parser.print_help()
        return 0

    args = parser.parse_args(raw)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "run":
        return _run(args)
    if args.cmd == "serve":
        return _serve(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
