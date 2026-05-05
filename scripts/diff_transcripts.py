"""Side-by-side diff of v1 (Whisper+pyannote 5-stage) and v2 transcripts.

Aligns roughly by start-second, then prints v1 and v2 lines together so a
human can scan for differences. v1 format: [HH:MM:SS --> HH:MM:SS] SPEAKER_XX: text.
v2 format: [SPEAKER_XX] HH:MM:SS text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_v1(text: str):
    """[00:00:00 --> 00:00:03] SPEAKER_00: text"""
    rows = []
    pat = re.compile(r"\[(\d\d):(\d\d):(\d\d) --> (\d\d):(\d\d):(\d\d)\]\s+(\S[^:]*):\s*(.*)")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        h1, mi1, s1, h2, mi2, s2, spk, txt = m.groups()
        start = int(h1) * 3600 + int(mi1) * 60 + int(s1)
        end = int(h2) * 3600 + int(mi2) * 60 + int(s2)
        rows.append((start, end, spk.strip(), txt.strip()))
    return rows


def parse_v2(text: str):
    """[SPEAKER_XX] HH:MM:SS text"""
    rows = []
    pat = re.compile(r"\[([^\]]+)\]\s+(\d\d):(\d\d):(\d\d)\s+(.*)")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        spk, h, mi, s, txt = m.groups()
        start = int(h) * 3600 + int(mi) * 60 + int(s)
        rows.append((start, None, spk.strip(), txt.strip()))
    return rows


def fmt(t):
    if t is None:
        return "      "
    return f"{t // 60:02d}:{t % 60:02d}"


def main():
    v1 = parse_v1(Path("output/Escape From the Bloodkeep Episode 1_20250624_072504.txt").read_text(encoding="utf-8"))
    v2 = parse_v2(Path("output/Escape From the Bloodkeep Episode 1_v2.txt").read_text(encoding="utf-8"))

    print(f"v1: {len(v1)} segments | v2: {len(v2)} segments\n")
    print(f"{'time':<6} | {'v1':<60} | {'v2':<60}")
    print("-" * 132)

    # interleave by start time
    i = j = 0
    while i < len(v1) or j < len(v2):
        v1_t = v1[i][0] if i < len(v1) else 1e9
        v2_t = v2[j][0] if j < len(v2) else 1e9
        if v1_t == v2_t:
            spk1, t1 = v1[i][2], v1[i][3]
            spk2, t2 = v2[j][2], v2[j][3]
            print(f"{fmt(v1_t):<6} | {(spk1 + ': ' + t1)[:60]:<60} | {(spk2 + ': ' + t2)[:60]:<60}")
            i += 1
            j += 1
        elif v1_t < v2_t:
            spk1, t1 = v1[i][2], v1[i][3]
            print(f"{fmt(v1_t):<6} | {(spk1 + ': ' + t1)[:60]:<60} | ")
            i += 1
        else:
            spk2, t2 = v2[j][2], v2[j][3]
            print(f"{fmt(v2_t):<6} | {' ':<60} | {(spk2 + ': ' + t2)[:60]:<60}")
            j += 1


if __name__ == "__main__":
    main()
