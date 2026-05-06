"""Algorithm-correctness tests for the crosstalk heuristic.

Exercises the canonical Python reference at diarizer/webapp/crosstalk.py.
The JS mirror in diarizer/webapp/static/app.js is verified for surface
parity here (grep) and for behavioural parity by the manual walkthrough
(Step 5 item 0 of PLAN 202605060400).
"""

from __future__ import annotations

from diarizer.webapp.crosstalk import compute_crosstalk_regions


def _seg(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker, "text": ""}


def test_crosstalk_empty_returns_empty():
    r = compute_crosstalk_regions([])
    assert r == {"flagged_segments": [], "flagged_ranges": []}


def test_crosstalk_single_speaker_session_no_flags():
    segs = [_seg(i * 0.5, i * 0.5 + 0.4, "SPEAKER_00") for i in range(10)]
    r = compute_crosstalk_regions(segs)
    assert r["flagged_segments"] == []
    assert r["flagged_ranges"] == []


def test_crosstalk_two_segments_below_threshold():
    segs = [_seg(0, 0.5, "A"), _seg(0.5, 1.0, "B")]
    r = compute_crosstalk_regions(segs)
    assert r["flagged_segments"] == []


def test_crosstalk_dense_swaps_flagged():
    """5 segments at [0, 0.5, 1, 1.5, 2] with alternating A/B/A/B/A → 4 swaps in 2s window."""
    segs = [
        _seg(0.0, 0.5, "A"),
        _seg(0.5, 1.0, "B"),
        _seg(1.0, 1.5, "A"),
        _seg(1.5, 2.0, "B"),
        _seg(2.0, 2.5, "A"),
    ]
    r = compute_crosstalk_regions(segs)
    assert r["flagged_segments"] == [0, 1, 2, 3, 4]
    assert len(r["flagged_ranges"]) == 1
    assert r["flagged_ranges"][0]["start"] == 0.0
    assert r["flagged_ranges"][0]["end"] == 2.5


def test_crosstalk_sparse_pause_no_flags():
    """Two quick swaps, then 30s of silence, then two more quick swaps. Below threshold each cluster."""
    segs = [
        _seg(0.0, 0.5, "A"), _seg(0.5, 1.0, "B"),                           # 1 swap
        _seg(31.0, 31.5, "A"), _seg(31.5, 32.0, "B"),                       # 1 swap
    ]
    r = compute_crosstalk_regions(segs)
    assert r["flagged_segments"] == []


def test_crosstalk_long_segment_does_not_break_window_walk():
    """One 60s monologue then rapid swaps after — the long segment shouldn't artificially extend swap counts."""
    segs = [
        _seg(0.0, 60.0, "A"),
        _seg(60.0, 60.5, "B"),
        _seg(60.5, 61.0, "A"),
        _seg(61.0, 61.5, "B"),
        _seg(61.5, 62.0, "A"),
        _seg(62.0, 62.5, "B"),
    ]
    r = compute_crosstalk_regions(segs)
    # Starting at i=1 (60.0), j-walk extends to i=5 (62.0) since 62.0 - 60.0 = 2.0 <= 5.0.
    # That run has 4 swaps among [1..5]. Threshold met → segments 1..5 flagged.
    assert 0 not in r["flagged_segments"]
    assert set(r["flagged_segments"]) == {1, 2, 3, 4, 5}


def test_crosstalk_threshold_boundary_exactly_at_threshold():
    """Exactly 4 swaps in a 5s window with threshold=4 → flagged (>= comparison)."""
    segs = [
        _seg(0.0, 0.5, "A"),
        _seg(0.5, 1.0, "B"),
        _seg(1.0, 1.5, "A"),
        _seg(1.5, 2.0, "B"),
        _seg(2.0, 2.5, "A"),  # 4 swaps so far
    ]
    r = compute_crosstalk_regions(segs, window_sec=5.0, threshold=4)
    assert r["flagged_segments"] == [0, 1, 2, 3, 4]


def test_crosstalk_just_below_threshold():
    """3 swaps among 4 segments, threshold=4 → no flags."""
    segs = [
        _seg(0.0, 0.5, "A"),
        _seg(0.5, 1.0, "B"),
        _seg(1.0, 1.5, "A"),
        _seg(1.5, 2.0, "B"),  # 3 swaps
    ]
    r = compute_crosstalk_regions(segs, window_sec=5.0, threshold=4)
    assert r["flagged_segments"] == []


def test_crosstalk_overlapping_windows_merge_ranges():
    """Two crosstalk clusters within 1s → ranges merge into one."""
    segs = [
        _seg(0.0, 0.4, "A"),
        _seg(0.4, 0.8, "B"),
        _seg(0.8, 1.2, "A"),
        _seg(1.2, 1.6, "B"),
        _seg(1.6, 2.0, "A"),  # cluster 1 ends ~2.0 (4 swaps in 1.6s)
        _seg(3.0, 3.4, "B"),
        _seg(3.4, 3.8, "A"),
        _seg(3.8, 4.2, "B"),
        _seg(4.2, 4.6, "A"),  # cluster 2 starts at 3.0
    ]
    r = compute_crosstalk_regions(segs)
    # Both clusters should flag; the j-walk from i=0 extends through i=8 because
    # segments[8].start - segments[0].start = 4.2 <= 5.0, accumulating 8 swaps.
    # → single merged range covering [0.0, 4.6].
    assert len(r["flagged_ranges"]) == 1
    assert r["flagged_ranges"][0]["start"] == 0.0
    assert r["flagged_ranges"][0]["end"] >= 4.6


def test_app_js_has_crosstalk_helper():
    """Surface parity: JS function name, key state vars, render hooks present."""
    js = open("diarizer/webapp/static/app.js", encoding="utf-8").read()
    assert "computeCrosstalkRegions" in js
    assert "flaggedSegments" in js
    assert "flaggedRanges" in js
    assert "recomputeCrosstalk" in js
    css = open("diarizer/webapp/static/style.css", encoding="utf-8").read()
    assert ".segment.crosstalk" in css and "border-right" in css
