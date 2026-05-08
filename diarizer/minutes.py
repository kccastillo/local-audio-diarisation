"""Meeting-minutes generation via the Anthropic API.

Optional, opt-in feature. Sending transcript content to Anthropic breaks the
project's offline-by-default guarantee — see README.md for the caveat. The webapp
calls this module only when the user explicitly clicks Generate, and only when
ANTHROPIC_API_KEY is set.

Public API:
    extract_emotive_segments(transcript, peaks, n) -> list[dict]
    build_prompt(transcript, attendees, emotive_segments) -> list[dict]
    generate(transcript, attendees, model, peaks) -> dict
    resolve_refs(statements, transcript) -> list[dict]
"""

from __future__ import annotations

import json
import os
import uuid
from difflib import SequenceMatcher
from typing import Any

REQUIRED_SECTIONS = (
    "agenda",
    "discussion",
    "decisions",
    "action_items",
    "next_steps",
    "tensions",
)

FUZZY_THRESHOLD = 0.6


class MinutesError(Exception):
    """Base for minutes-generation errors."""


class MissingAPIKeyError(MinutesError):
    """ANTHROPIC_API_KEY is not set."""


class SchemaError(MinutesError):
    """LLM response did not conform to the expected schema."""


class APIError(MinutesError):
    """Anthropic API call failed."""


# ---------- Emotive-segment ranking ----------


def extract_emotive_segments(
    transcript: list[dict], peaks: dict, n: int = 15
) -> list[dict]:
    """Rank transcript segments by audio energy and return the top-N.

    `peaks` is the `waveform_peaks.json` shape: {bins, duration_s, peaks: [0..1]}.
    Energy score for a segment is mean(|peak|) over the segment's [start, end]
    time range, mapped onto the peak bins via duration_s.

    Returns: [{seg_index, speaker, energy_score, text, start, end}, ...] sorted
    by energy_score desc, length min(n, len(transcript)).
    """
    if not transcript:
        return []
    bins = peaks.get("bins") or 0
    duration_s = peaks.get("duration_s") or 0.0
    peak_arr = peaks.get("peaks") or []
    if bins <= 0 or duration_s <= 0 or not peak_arr:
        return []

    scored: list[dict] = []
    for i, seg in enumerate(transcript):
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if end <= start:
            continue
        i0 = int((start / duration_s) * bins)
        i1 = int((end / duration_s) * bins)
        i0 = max(0, min(bins - 1, i0))
        i1 = max(i0 + 1, min(bins, i1))
        slice_ = peak_arr[i0:i1]
        if not slice_:
            continue
        energy = sum(abs(float(x)) for x in slice_) / len(slice_)
        scored.append(
            {
                "seg_index": i,
                "speaker": seg.get("speaker") or "",
                "energy_score": energy,
                "text": seg.get("text") or "",
                "start": start,
                "end": end,
            }
        )

    scored.sort(key=lambda r: r["energy_score"], reverse=True)
    return scored[: max(0, int(n))]


# ---------- Prompt construction ----------


SYSTEM_PROMPT = """You are a meeting-minutes assistant. You are given a verbatim transcript of a meeting (with speaker labels and per-segment indices), the list of attendees, and a list of segments that were notably louder than the rest of the meeting (a hint that those moments may carry emotional weight or tension).

Produce concise, professional meeting minutes in standard format. Output strict JSON matching this schema:

{
  "agenda": [<statement>, ...],
  "discussion": [<statement>, ...],
  "decisions": [<statement>, ...],
  "action_items": [<statement>, ...],
  "next_steps": [<statement>, ...],
  "tensions": [<statement>, ...]
}

Each <statement> is an object:
{
  "id": "<short unique string>",
  "text": "<the minute itself, one or two sentences>",
  "quote": "<a short verbatim snippet (5-15 words) from the transcript that grounds this statement>",
  "cited_index": <integer segment index that contains the quote>,
  "secondary": [{"quote": "<verbatim snippet>", "cited_index": <int>}, ...],
  "kind": "<optional, one of: decision, action, tension, statement_of_interest>"
}

Rules:
- Every statement MUST have a `quote` AND a `cited_index`. The quote must be a verbatim substring of the segment at `cited_index` whenever possible.
- `secondary` is a list of additional supporting references (later mentions of the same point). Empty list if none.
- The `tensions` section is for moments of disagreement, friction, or notable emotional weight — use the louder-segments hint, but rely primarily on the actual content. Use `kind: "tension"` for these.
- Keep `text` concise — a minute, not a paragraph.
- Do NOT include any narrative outside the JSON. Output a single JSON object and nothing else.
- Empty sections are allowed (use `[]`).
"""


def build_prompt(
    transcript: list[dict],
    attendees: list[str],
    emotive_segments: list[dict],
) -> list[dict]:
    """Construct the messages list for the Anthropic API.

    Returns a list suitable for passing as `messages` to client.messages.create.
    The system prompt is exposed via SYSTEM_PROMPT and passed separately.
    """
    transcript_lines = []
    for i, seg in enumerate(transcript):
        speaker = seg.get("speaker") or "?"
        text = (seg.get("text") or "").replace("\n", " ").strip()
        transcript_lines.append(f"[{i}] {speaker}: {text}")
    transcript_block = "\n".join(transcript_lines)

    if attendees:
        attendees_block = ", ".join(attendees)
    else:
        attendees_block = "(none provided)"

    if emotive_segments:
        emotive_lines = []
        for r in emotive_segments:
            emotive_lines.append(
                f"[{r['seg_index']}] {r.get('speaker', '?')} "
                f"(energy={r['energy_score']:.3f}): "
                f"{(r.get('text') or '').strip()}"
            )
        emotive_block = "\n".join(emotive_lines)
    else:
        emotive_block = "(no emotive-segment hints available)"

    user_text = (
        f"Attendees: {attendees_block}\n\n"
        f"Louder-than-average segments (consider for tension detection):\n"
        f"{emotive_block}\n\n"
        f"Transcript:\n{transcript_block}\n\n"
        "Produce the meeting minutes JSON object now."
    )

    return [{"role": "user", "content": user_text}]


# ---------- Reference resolution ----------


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def _resolve_one_ref(
    quote: str, cited_index: int | None, transcript: list[dict]
) -> tuple[int | None, str]:
    """Resolve a single (quote, cited_index) pair.

    Returns (resolved_index, resolution_kind) where resolution_kind is "exact"
    or "fuzzy" or "drop". `resolved_index` is None when "drop".
    """
    n = len(transcript)
    nq = _normalise(quote)

    if isinstance(cited_index, int) and 0 <= cited_index < n and nq:
        cited_text = _normalise(transcript[cited_index].get("text") or "")
        if nq and nq in cited_text:
            return cited_index, "exact"

    if not nq:
        # No quote to match — fall back to the cited_index if it's at least valid.
        if isinstance(cited_index, int) and 0 <= cited_index < n:
            return cited_index, "exact"
        return None, "drop"

    best_idx = -1
    best_ratio = 0.0
    for i, seg in enumerate(transcript):
        seg_text = _normalise(seg.get("text") or "")
        if not seg_text:
            continue
        if nq in seg_text:
            return i, "fuzzy"
        ratio = SequenceMatcher(None, nq, seg_text).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio >= FUZZY_THRESHOLD and best_idx >= 0:
        return best_idx, "fuzzy"
    return None, "drop"


def resolve_refs(
    statements: list[dict], transcript: list[dict]
) -> list[dict]:
    """Validate and normalise each statement's references.

    Input shape (LLM-emitted):
        {id, text, quote, cited_index, secondary: [{quote, cited_index}], kind?}

    Output shape (canonical):
        {id, text, quote, primary_ref, secondary_refs, kind?, ref_resolution?}

    Drops statements whose primary reference falls below FUZZY_THRESHOLD.
    Drops individual secondary refs the same way (without dropping the parent).
    """
    out: list[dict] = []
    for st in statements or []:
        quote = st.get("quote") or ""
        cited = st.get("cited_index")
        primary_ref, kind = _resolve_one_ref(quote, cited, transcript)
        if primary_ref is None:
            continue

        secondary_refs: list[int] = []
        for ref in st.get("secondary") or []:
            if not isinstance(ref, dict):
                continue
            sq = ref.get("quote") or ""
            sc = ref.get("cited_index")
            sidx, _skind = _resolve_one_ref(sq, sc, transcript)
            if sidx is not None and sidx not in secondary_refs and sidx != primary_ref:
                secondary_refs.append(sidx)

        new = {
            "id": st.get("id") or uuid.uuid4().hex[:8],
            "text": st.get("text") or "",
            "quote": quote,
            "primary_ref": primary_ref,
            "secondary_refs": secondary_refs,
        }
        if st.get("kind"):
            new["kind"] = st["kind"]
        if kind == "fuzzy":
            new["ref_resolution"] = "fuzzy"
        out.append(new)
    return out


# ---------- Top-level generate ----------


def _strip_code_fence(s: str) -> str:
    """If the LLM wrapped its JSON in a markdown fence, peel it."""
    s = s.strip()
    if s.startswith("```"):
        # drop opening fence (with or without language tag) and closing fence
        first_nl = s.find("\n")
        if first_nl == -1:
            return s
        s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def generate(
    transcript: list[dict],
    attendees: list[str],
    model: str,
    peaks: dict,
    *,
    top_n_emotive: int = 15,
) -> dict:
    """Generate meeting minutes via the Anthropic API.

    Raises MissingAPIKeyError, APIError, or SchemaError. Returns a minutes dict
    with sections populated and references resolved.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set")

    try:
        import anthropic  # heavy import deferred
    except ImportError as e:  # pragma: no cover — covered by deps install
        raise APIError(f"anthropic SDK not installed: {e}") from e

    emotive = extract_emotive_segments(transcript, peaks, top_n_emotive)
    messages = build_prompt(transcript, attendees, emotive)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except Exception as e:
        raise APIError(f"Anthropic API call failed: {e}") from e

    # Concatenate text blocks from the response.
    raw_text = ""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            raw_text += getattr(block, "text", "")
    raw_text = _strip_code_fence(raw_text)

    if not raw_text:
        raise SchemaError("LLM returned empty response")

    try:
        body = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise SchemaError(f"LLM response is not valid JSON: {e}") from e

    if not isinstance(body, dict):
        raise SchemaError("LLM response root is not an object")

    sections: dict[str, list[dict]] = {}
    for name in REQUIRED_SECTIONS:
        raw_section = body.get(name) or []
        if not isinstance(raw_section, list):
            raise SchemaError(f"Section '{name}' is not a list")
        sections[name] = resolve_refs(raw_section, transcript)

    return {
        "model": model,
        "attendees": list(attendees),
        "sections": sections,
    }
