"""Score panel outputs against a manual-corrections ground-truth file.

Three metrics per output:
  1. Proper-noun recall — count occurrences of each named entity in the output
     vs. how many times it appears in the manual-corrections file.
  2. Domain-term recall — same idea for the cyber/control vocabulary.
  3. Token-set Jaccard — sanity check that nothing went sideways elsewhere.

Output shape: a single table to stdout, plus per-config diff annotations.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------- vocabulary ----------------
PROPER_NOUNS = [
    "Pradeep", "Sadeed", "Declan", "Bryan", "Brian", "Leigh", "Oliver",
    "Jaremin", "Steve Russell", "Andrew", "Emma", "Harry", "Scott",
    "Rachel", "Freddie", "Mick", "Georgia", "Ken", "Peter", "Ejaz",
    "Seqwater", "Pradeep's", "Declan's",
]

DOMAIN_TERMS = [
    "incident", "incident management", "audit", "audit mode",
    "cyber", "cybersecurity", "Airlock", "OTP", "RITM", "BAU",
    "Citrix", "Ring Zero", "Ring 0", "out-of-hours", "out of hours",
    "blacklisting", "whitelisting", "configuration", "executive",
    "unexecuted", "fiddle", "service request", "break glass",
    "P1", "P2", "P3", "P4", "SOE", "Intune", "Rapid7",
]

# Whisper sometimes outputs "Cyber" capitalised even when the speaker uses lowercase
# in a non-name sense. Counts are case-insensitive for fairness.
DEGENERATION_MARKERS = [
    # short single-word lines = sign of fragmentation
    "shave it",          # known hallucination ending we saw
]


def normalise(text: str) -> str:
    """Strip diarisation prefix + timestamp; lowercase; remove punctuation; tokenise-friendly form."""
    out_lines = []
    for line in text.splitlines():
        # Strip [SPEAKER_xx] HH:MM:SS or [HH:MM:SS --> HH:MM:SS] SPEAKER_xx: or [KC] HH:MM:SS
        line = re.sub(r"^\[[^\]]+\]\s*\d{2}:\d{2}:\d{2}\s*", "", line)
        line = re.sub(r"^\[\d{2}:\d{2}:\d{2}\s*-->\s*\d{2}:\d{2}:\d{2}\]\s*[^:]+:\s*", "", line)
        line = re.sub(r"^\[[A-Z]+\]\s*\d{2}:\d{2}:\d{2}\s*", "", line)
        out_lines.append(line.strip())
    return " ".join(out_lines)


def tokens(text: str) -> set[str]:
    """Lowercase word tokens, alpha-only, length ≥ 2."""
    return {t for t in re.findall(r"[a-zA-Z']{2,}", text.lower())}


def count_phrase(text: str, phrase: str) -> int:
    """Case-insensitive whole-token match for a phrase."""
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def vocab_score(reference: str, hypothesis: str, vocab: list[str]) -> tuple[int, int, float, list[tuple[str, int, int]]]:
    """For a vocabulary list, return (total_ref_count, total_hyp_count, recall, per_term_breakdown)."""
    breakdown = []
    total_ref = 0
    total_hit = 0
    for term in vocab:
        r = count_phrase(reference, term)
        h = count_phrase(hypothesis, term)
        breakdown.append((term, r, h))
        total_ref += r
        total_hit += min(r, h)  # cap hits at reference count (no credit for over-production)
    recall = total_hit / max(1, total_ref)
    return total_ref, total_hit, recall, breakdown


def degenerate_signal(text: str) -> dict:
    """Heuristics for output degeneration."""
    lines = [l for l in text.splitlines() if l.strip()]
    n_lines = len(lines)
    short_lines = sum(1 for l in lines if len(l.split()) <= 4)
    avg_words = sum(len(l.split()) for l in lines) / max(1, n_lines)
    return {
        "lines": n_lines,
        "short_lines": short_lines,
        "short_pct": short_lines / max(1, n_lines),
        "avg_words_per_line": avg_words,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--per-term", action="store_true", help="Print per-term recall breakdown.")
    args = p.parse_args()

    if not args.reference.exists():
        print(f"reference not found: {args.reference}", file=sys.stderr)
        return 2

    ref_text = normalise(args.reference.read_text(encoding="utf-8"))
    ref_tokens = tokens(ref_text)

    rows = []
    panel_files = sorted(args.panel_dir.glob("*.txt"))
    if not panel_files:
        print(f"no .txt files in {args.panel_dir}", file=sys.stderr)
        return 2

    for f in panel_files:
        hyp = normalise(f.read_text(encoding="utf-8"))
        hyp_toks = tokens(hyp)

        pn_ref, pn_hit, pn_recall, pn_breakdown = vocab_score(ref_text, hyp, PROPER_NOUNS)
        dt_ref, dt_hit, dt_recall, dt_breakdown = vocab_score(ref_text, hyp, DOMAIN_TERMS)

        jac = jaccard(ref_tokens, hyp_toks)
        deg = degenerate_signal(f.read_text(encoding="utf-8"))

        rows.append({
            "name": f.stem,
            "lines": deg["lines"],
            "avg_w": deg["avg_words_per_line"],
            "short_pct": deg["short_pct"],
            "pn_recall": pn_recall,
            "pn_hit": pn_hit,
            "pn_ref": pn_ref,
            "dt_recall": dt_recall,
            "dt_hit": dt_hit,
            "dt_ref": dt_ref,
            "jaccard": jac,
            "pn_breakdown": pn_breakdown,
            "dt_breakdown": dt_breakdown,
        })

    print(f"reference: {args.reference} ({len(ref_tokens)} unique tokens)")
    print()
    print(f"{'config':<18} {'lines':>6} {'avg_w':>6} {'short%':>7} | {'PN_recall':>10} {'PN_hit/ref':>11} | {'DT_recall':>10} {'DT_hit/ref':>11} | {'Jaccard':>7}")
    print("-" * 110)
    for r in rows:
        pn_frac = f"{r['pn_hit']}/{r['pn_ref']}"
        dt_frac = f"{r['dt_hit']}/{r['dt_ref']}"
        print(
            f"{r['name']:<18} {r['lines']:>6} {r['avg_w']:>6.1f} {r['short_pct']*100:>6.1f}% | "
            f"{r['pn_recall']*100:>9.1f}% {pn_frac:>11} | "
            f"{r['dt_recall']*100:>9.1f}% {dt_frac:>11} | "
            f"{r['jaccard']*100:>6.1f}%"
        )

    if args.per_term:
        print()
        for r in rows:
            print(f"\n=== {r['name']} — proper nouns (term: ref -> hyp) ===")
            for term, ref, hyp in r["pn_breakdown"]:
                if ref or hyp:
                    flag = "  " if hyp >= ref else "X "
                    print(f"  {flag}{term:<22} ref={ref:>3}  hyp={hyp:>3}")
            print(f"\n=== {r['name']} — domain terms ===")
            for term, ref, hyp in r["dt_breakdown"]:
                if ref or hyp:
                    flag = "  " if hyp >= ref else "X "
                    print(f"  {flag}{term:<22} ref={ref:>3}  hyp={hyp:>3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
