"""Reference implementation of the crosstalk-region heuristic.

Mirror of `diarizer/webapp/static/app.js:computeCrosstalkRegions`. Keep the two
algorithms in lockstep — the pytest fixtures in `tests/diarizer/test_crosstalk.py`
are the canonical correctness gate; the JS-vs-Python parity check is item 0
of the Step 5 manual walkthrough.

Heuristic: walk segments in start-time order. For each starting index i, find
the maximal contiguous run [i, j] such that segments[j].start - segments[i].start
<= window_sec. Count speaker-swaps in that run. If >= threshold, flag every
segment in [i, j] and merge [segments[i].start, segments[j].end] into the
flagged-ranges list.
"""

from __future__ import annotations


def compute_crosstalk_regions(
    segments: list[dict],
    window_sec: float = 5.0,
    threshold: int = 4,
) -> dict:
    if not segments:
        return {"flagged_segments": [], "flagged_ranges": []}

    flagged: set[int] = set()
    ranges: list[dict] = []

    n = len(segments)
    for i in range(n):
        # Find maximal j such that segments[j].start - segments[i].start <= window_sec.
        j = i
        while j + 1 < n and segments[j + 1]["start"] - segments[i]["start"] <= window_sec:
            j += 1
        if j == i:
            continue
        # Count speaker swaps in [i+1, j].
        swaps = 0
        for k in range(i + 1, j + 1):
            if segments[k].get("speaker") != segments[k - 1].get("speaker"):
                swaps += 1
        if swaps >= threshold:
            for idx in range(i, j + 1):
                flagged.add(idx)
            new_range = {"start": float(segments[i]["start"]), "end": float(segments[j]["end"])}
            # Merge with last range if they overlap or touch.
            if ranges and new_range["start"] <= ranges[-1]["end"]:
                ranges[-1]["end"] = max(ranges[-1]["end"], new_range["end"])
            else:
                ranges.append(new_range)

    return {
        "flagged_segments": sorted(flagged),
        "flagged_ranges": ranges,
    }
