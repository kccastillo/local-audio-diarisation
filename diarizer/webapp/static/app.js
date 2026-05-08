// Diarizer transcript-review frontend v2.
// No live audio-graph analyser — pre-rendered waveform peaks instead.
// Vanilla JS, no build step.

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const transcriptEl = $("transcript");
const canvas = $("waveform");
const ctx2d = canvas.getContext("2d");

// ---------- State ----------

let segments = [];          // current edit state
let originalSegments = [];
let lastSavedSegments = []; // snapshot of segments at last save (or init/load)
let diffBaseSegments = []; // what diff view compares CURRENT against (default: original)
let diffBaseLabel = "original";
let speakers = [];
let diffMode = false;

// Waveform / zoom state
let peaks = null;            // Float32Array of peak amplitudes [0,1]
let peaksDuration = 0;       // total seconds
let zoomLevel = 1;           // 1 .. 32, doubling
const ZOOM_MAX = 32;
let windowStartTime = 0;     // left edge of visible window in seconds

// Sync mode
let syncMode = false;
let speakerModalRow = null;  // remembered row for focus restoration

// Crosstalk
let crosstalk = { flaggedSegments: new Set(), flaggedRanges: [] };

// Minutes panel state
let minutes = null;            // current minutes object {model, attendees, sections, generated_at?}
let minutesDirty = false;      // tracks unsaved changes vs last persisted
let minutesHighlight = null;   // {primary: {start, end}, secondaries: [{start, end}, ...]}

// Active-speaker highlight (left-click on a speaker pill toggles)
let selectedSpeaker = null;
let selectedSpeakerRanges = [];

function computeSpeakerRanges(speakerName) {
  if (!speakerName) return [];
  const ranges = [];
  for (const s of segments) {
    if (s.speaker === speakerName) {
      if (ranges.length && s.start <= ranges[ranges.length - 1].end) {
        ranges[ranges.length - 1].end = Math.max(ranges[ranges.length - 1].end, s.end);
      } else {
        ranges.push({ start: s.start, end: s.end });
      }
    }
  }
  return ranges;
}

function refreshSelectedSpeakerRanges() {
  // Clear selection if the speaker no longer exists in segments (e.g. all
  // their segments were renamed away).
  if (selectedSpeaker && !segments.some(s => s.speaker === selectedSpeaker)) {
    selectedSpeaker = null;
  }
  selectedSpeakerRanges = computeSpeakerRanges(selectedSpeaker);
}

function setSelectedSpeaker(name) {
  selectedSpeaker = (selectedSpeaker === name) ? null : name;
  refreshSelectedSpeakerRanges();
  // Refresh pill highlight on transcript without rebuilding rows.
  for (const pill of transcriptEl.querySelectorAll(".speaker")) {
    pill.classList.toggle("speaker-active", selectedSpeaker !== null && pill.textContent === selectedSpeaker);
  }
}

// Mirror of diarizer/webapp/crosstalk.py:compute_crosstalk_regions.
// Algorithm assumes segments[].start is monotonically non-decreasing; the
// pipeline guarantees this. If a future edit path ever re-orders segments,
// sort by start before calling.
function computeCrosstalkRegions(segments, windowSec = 10, threshold = 3) {
  if (!segments || segments.length === 0) {
    return { flaggedSegments: new Set(), flaggedRanges: [] };
  }
  const flagged = new Set();
  const ranges = [];
  const n = segments.length;
  for (let i = 0; i < n; i++) {
    let j = i;
    while (j + 1 < n && segments[j + 1].start - segments[i].start <= windowSec) j++;
    if (j === i) continue;
    let swaps = 0;
    for (let k = i + 1; k <= j; k++) {
      if (segments[k].speaker !== segments[k - 1].speaker) swaps++;
    }
    if (swaps >= threshold) {
      for (let idx = i; idx <= j; idx++) flagged.add(idx);
      const newRange = { start: segments[i].start, end: segments[j].end };
      if (ranges.length && newRange.start <= ranges[ranges.length - 1].end) {
        ranges[ranges.length - 1].end = Math.max(ranges[ranges.length - 1].end, newRange.end);
      } else {
        ranges.push(newRange);
      }
    }
  }
  return { flaggedSegments: flagged, flaggedRanges: ranges };
}

function recomputeCrosstalk() {
  crosstalk = computeCrosstalkRegions(segments);
}

// ---------- Dirty tracking ----------

function hasUnsavedChanges() {
  if (segments.length !== lastSavedSegments.length) return true;
  for (let i = 0; i < segments.length; i++) {
    if (segments[i].text !== lastSavedSegments[i].text) return true;
    if (segments[i].speaker !== lastSavedSegments[i].speaker) return true;
  }
  return false;
}

function snapshotSaved() {
  lastSavedSegments = JSON.parse(JSON.stringify(segments));
}

function hasChangesVsOriginal() {
  if (segments.length !== originalSegments.length) return true;
  for (let i = 0; i < segments.length; i++) {
    if (segments[i].text !== originalSegments[i].text) return true;
    if (segments[i].speaker !== originalSegments[i].speaker) return true;
  }
  return false;
}

function hasAnyChanges() {
  // Any saved edits on disk OR unsaved differences vs original.
  const versionsSel = $("versions-dropdown");
  const editsExist = versionsSel && versionsSel.options.length > 1;
  return editsExist || hasChangesVsOriginal();
}

function updateDirtyButtons() {
  $("btn-save").disabled = !hasUnsavedChanges();
  $("btn-diff").disabled = !hasAnyChanges();
}

// ---------- helpers ----------

function fmtTime(s) {
  if (!isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

function showToast(msg, ms = 4000) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), ms);
}

function uniqueSpeakers(segs) {
  const seen = new Set();
  for (const s of segs) if (s.speaker) seen.add(s.speaker);
  return [...seen];
}

function visibleSeconds() { return peaksDuration / zoomLevel; }

function clampWindow() {
  const vs = visibleSeconds();
  windowStartTime = Math.max(0, Math.min(windowStartTime, peaksDuration - vs));
}

function timeToX(t) {
  const vs = visibleSeconds();
  return ((t - windowStartTime) / vs) * canvas.width;
}

function xToTime(x) {
  const vs = visibleSeconds();
  return windowStartTime + (x / canvas.width) * vs;
}

// ---------- Waveform render ----------

function drawWaveform() {
  const w = canvas.width, h = canvas.height;
  ctx2d.fillStyle = "#000";
  ctx2d.fillRect(0, 0, w, h);
  if (!peaks || peaks.length === 0) return;

  const vs = visibleSeconds();

  // Crosstalk background stripes — drawn first, behind the amplitude bars.
  if (crosstalk.flaggedRanges && crosstalk.flaggedRanges.length) {
    ctx2d.fillStyle = "rgba(240, 160, 0, 0.18)";  // var(--warn) at low alpha
    for (const r of crosstalk.flaggedRanges) {
      const visEnd = windowStartTime + vs;
      if (r.end < windowStartTime || r.start > visEnd) continue;
      const x1 = Math.max(0, timeToX(r.start));
      const x2 = Math.min(w, timeToX(r.end));
      if (x2 > x1) ctx2d.fillRect(x1, 0, x2 - x1, h);
    }
  }

  const mid = h / 2;
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent-wave").trim() || "#5fd97a";
  const highlight = getComputedStyle(document.documentElement).getPropertyValue("--accent-speaker").trim() || "#9ddfff";

  for (let x = 0; x < w; x++) {
    const t = windowStartTime + (x / w) * vs;
    if (t < 0 || t > peaksDuration) continue;
    const idx = Math.min(peaks.length - 1, Math.max(0, Math.floor((t / peaksDuration) * peaks.length)));
    const peak = peaks[idx];
    const half = peak * (h / 2);
    let inSpeaker = false;
    if (selectedSpeakerRanges.length) {
      for (const r of selectedSpeakerRanges) {
        if (t >= r.start && t < r.end) { inSpeaker = true; break; }
      }
    }
    ctx2d.fillStyle = inSpeaker ? highlight : accent;
    ctx2d.fillRect(x, mid - half, 1, half * 2 || 1);
  }

  // Minutes-anchor highlight (drawn over waveform, before playhead).
  if (minutesHighlight) {
    const visEnd = windowStartTime + vs;
    if (minutesHighlight.primary) {
      const r = minutesHighlight.primary;
      if (!(r.end < windowStartTime || r.start > visEnd)) {
        ctx2d.fillStyle = "rgba(78, 161, 255, 0.30)";
        const x1 = Math.max(0, timeToX(r.start));
        const x2 = Math.min(w, timeToX(r.end));
        if (x2 > x1) ctx2d.fillRect(x1, 0, x2 - x1, h);
      }
    }
    if (minutesHighlight.secondaries) {
      ctx2d.fillStyle = "rgba(78, 161, 255, 0.55)";
      for (const r of minutesHighlight.secondaries) {
        if (r.end < windowStartTime || r.start > visEnd) continue;
        const x1 = Math.max(0, timeToX(r.start));
        const x2 = Math.min(w, timeToX(r.end));
        const w2 = Math.max(2, x2 - x1);
        ctx2d.fillRect(x1, 0, w2, 4);
        ctx2d.fillRect(x1, h - 4, w2, 4);
      }
    }
  }

  // Playhead
  if (audio.currentTime >= windowStartTime && audio.currentTime <= windowStartTime + vs) {
    ctx2d.strokeStyle = "#fff";
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    const px = timeToX(audio.currentTime);
    ctx2d.moveTo(px, 0);
    ctx2d.lineTo(px, h);
    ctx2d.stroke();
  }
}

function renderLoop() {
  requestAnimationFrame(renderLoop);
  // Jump-pan when zoomed and playing
  if (zoomLevel > 1 && !audio.paused && peaksDuration > 0) {
    const vs = visibleSeconds();
    if (audio.currentTime > windowStartTime + vs * 0.75) {
      windowStartTime = audio.currentTime - vs * 0.25;
      clampWindow();
    }
  }
  drawWaveform();
}

// ---------- Click to seek + auto-play ----------

canvas.addEventListener("mousedown", (e) => {
  if (!peaksDuration) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const t = xToTime(x);
  audio.currentTime = Math.max(0, Math.min(peaksDuration, t));
  enterSync();
});

// Ctrl+wheel to zoom
canvas.addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const cursorTime = xToTime(x);
  if (e.deltaY < 0) zoomTo(zoomLevel * 2, cursorTime);
  else zoomTo(zoomLevel / 2, cursorTime);
}, { passive: false });

// ---------- Zoom ----------

function zoomTo(level, centreTime) {
  level = Math.min(ZOOM_MAX, Math.max(1, level));
  zoomLevel = level;
  if (level === 1) {
    windowStartTime = 0;
  } else {
    const vs = peaksDuration / zoomLevel;
    const c = (centreTime !== undefined) ? centreTime : audio.currentTime;
    windowStartTime = c - vs / 2;
    clampWindow();
  }
  $("zoom-readout").textContent = `${zoomLevel}×`;
}

$("btn-zoom-in").addEventListener("click", () => zoomTo(zoomLevel * 2));
$("btn-zoom-out").addEventListener("click", () => zoomTo(zoomLevel / 2));

// ---------- Transcript render ----------

function renderTranscript() {
  if (diffMode) return renderDiff();
  transcriptEl.innerHTML = "";
  speakers = uniqueSpeakers(segments);
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const row = document.createElement("div");
    row.className = "segment";
    row.dataset.index = i;
    if (crosstalk.flaggedSegments.has(i)) {
      row.classList.add("crosstalk");
      row.title = "rapid speaker-swapping in this region";
    }

    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = seg.speaker || "—";
    if (selectedSpeaker !== null && seg.speaker === selectedSpeaker) {
      speaker.classList.add("speaker-active");
    }
    speaker.title = "Left-click: highlight this speaker on the waveform · Right-click: rename / reassign";
    speaker.addEventListener("click", (e) => {
      e.stopPropagation();
      if (seg.speaker) setSelectedSpeaker(seg.speaker);
    });
    speaker.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
      openSpeakerModal(i);
    });

    const ts = document.createElement("div");
    ts.className = "ts";
    ts.textContent = fmtTime(seg.start);
    ts.title = "Click to seek + play from here";
    ts.addEventListener("click", (e) => {
      e.stopPropagation();
      audio.currentTime = seg.start;
      enterSync();
    });

    const text = document.createElement("div");
    text.className = "text";
    text.contentEditable = "true";
    text.spellcheck = true;
    text.textContent = seg.text || "";
    text.addEventListener("input", () => {
      segments[i].text = text.textContent;
      updateDirtyButtons();
    });
    text.addEventListener("focus", () => {
      // Focus came in via click or arrow nav — drop sync if currently in sync
      if (syncMode) exitSync();
    });

    // Row body click (anywhere except speaker pill, ts, or text) → drop into edit mode on this row.
    // Clicks INSIDE the text cell are NOT hijacked — the browser's native caret-from-click
    // positioning is what the user wants for inline edits.
    row.addEventListener("click", (e) => {
      if (e.target === speaker || e.target === ts) return;
      if (e.target === text || text.contains(e.target)) {
        if (syncMode) exitSync();
        return;
      }
      dropOutOfSync(i);
    });

    row.appendChild(speaker);
    row.appendChild(ts);
    row.appendChild(text);
    transcriptEl.appendChild(row);
  }
}

function renderDiff() {
  transcriptEl.innerHTML = "";
  const base = diffBaseSegments;
  const n = Math.max(segments.length, base.length);
  for (let i = 0; i < n; i++) {
    const o = base[i];
    const e = segments[i];
    const row = document.createElement("div");
    row.className = "segment";
    row.dataset.index = i;
    if (!o) row.classList.add("diff-added");
    else if (!e) row.classList.add("diff-removed");
    else if (o.text !== e.text || o.speaker !== e.speaker) row.classList.add("diff-changed");

    const speaker = document.createElement("div");
    speaker.className = "speaker";
    if (o && e && o.speaker !== e.speaker) {
      speaker.innerHTML = `<span class="diff-old">${escapeHtml(o.speaker || "—")}</span><span class="diff-new">${escapeHtml(e.speaker || "—")}</span>`;
    } else {
      speaker.textContent = (e || o).speaker || "—";
    }

    const ts = document.createElement("div");
    ts.className = "ts";
    ts.textContent = fmtTime((e || o).start);

    const text = document.createElement("div");
    text.className = "text";
    if (o && e && o.text !== e.text) {
      text.innerHTML = `<span class="diff-old">${escapeHtml(o.text)}</span><span class="diff-new">${escapeHtml(e.text)}</span>`;
    } else {
      text.textContent = (e || o).text || "";
    }

    row.appendChild(speaker);
    row.appendChild(ts);
    row.appendChild(text);
    transcriptEl.appendChild(row);
  }
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

// ---------- Active-segment highlight ----------

function tickHighlight() {
  const rows = transcriptEl.querySelectorAll(".segment");
  if (!syncMode || diffMode) {
    for (const r of rows) r.classList.remove("active");
    return;
  }
  const now = audio.currentTime;
  let activeRow = null;
  for (const row of rows) {
    const i = parseInt(row.dataset.index, 10);
    const seg = segments[i];
    if (!seg) continue;
    const isActive = now >= seg.start && now < seg.end;
    row.classList.toggle("active", isActive);
    if (isActive) activeRow = row;
  }
  if (activeRow) activeRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

audio.addEventListener("timeupdate", () => {
  $("time-now").textContent = fmtTime(audio.currentTime);
  tickHighlight();
});
audio.addEventListener("loadedmetadata", () => {
  $("time-total").textContent = fmtTime(audio.duration);
});

// ---------- Sync state machine ----------

function enterSync() {
  syncMode = true;
  audio.play().catch((e) => { console.warn("play() rejected:", e); });
  $("btn-playpause").textContent = "Pause";
}

function exitSync() {
  syncMode = false;
  audio.pause();
  $("btn-playpause").textContent = "Play";
  const rows = transcriptEl.querySelectorAll(".segment.active");
  for (const r of rows) r.classList.remove("active");
}

function dropOutOfSync(targetIndex) {
  exitSync();
  if (typeof targetIndex === "number" && targetIndex >= 0) {
    const row = transcriptEl.querySelector(`.segment[data-index="${targetIndex}"]`);
    if (row) {
      const text = row.querySelector(".text");
      if (text) {
        text.focus();
        // Place caret at end
        const range = document.createRange();
        range.selectNodeContents(text);
        range.collapse(false);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }
  }
}

$("btn-playpause").addEventListener("click", () => {
  if (syncMode) exitSync();
  else enterSync();
});

$("btn-back5").addEventListener("click", () => {
  audio.currentTime = Math.max(0, audio.currentTime - 5);
});
$("btn-fwd5").addEventListener("click", () => {
  audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 5);
});
$("vol").addEventListener("input", (e) => {
  audio.volume = parseFloat(e.target.value);
});

// ---------- Up/Down arrow navigation ----------

document.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
  const ae = document.activeElement;
  if (!ae || !transcriptEl.contains(ae)) return;
  e.preventDefault();
  // Find current index
  let row = ae.closest(".segment");
  if (!row) return;
  const idx = parseInt(row.dataset.index, 10);
  if (isNaN(idx)) return;
  let target = idx + (e.key === "ArrowDown" ? 1 : -1);
  target = Math.max(0, Math.min(segments.length - 1, target));
  if (target === idx) return;
  dropOutOfSync(target);
});

// ---------- Speaker modal ----------

function openSpeakerModal(idx) {
  if (syncMode) exitSync();
  speakerModalRow = idx;
  const seg = segments[idx];
  const currentSpeaker = seg.speaker || "";
  const sel = $("speaker-existing");
  sel.innerHTML = `<option value="">(new label below)</option>`;
  for (const sp of uniqueSpeakers(segments)) {
    const opt = document.createElement("option");
    opt.value = sp; opt.textContent = sp;
    sel.appendChild(opt);
  }
  sel.value = currentSpeaker;
  $("speaker-new").value = "";
  $("speaker-replace-all").checked = false;
  $("speaker-replace-label").textContent = `Replace all segments currently labelled "${currentSpeaker}"`;
  $("speaker-modal").showModal();
}

$("speaker-ok").addEventListener("click", (e) => {
  if (speakerModalRow == null) return;
  const idx = speakerModalRow;
  const currentSpeaker = segments[idx].speaker || "";
  const newRaw = $("speaker-new").value;
  const newTrim = newRaw.trim();
  const fromDropdown = $("speaker-existing").value;
  const replaceAll = $("speaker-replace-all").checked;

  // Resolve target label.
  // Typing an existing label is treated the same as picking it from the
  // dropdown — silently reassigns to that speaker, no error.
  let target;
  if (newTrim) {
    target = newTrim;
  } else if (fromDropdown) {
    target = fromDropdown;
  } else {
    // No-op
    return restoreFocus();
  }

  if (replaceAll && currentSpeaker) {
    for (const s of segments) if (s.speaker === currentSpeaker) s.speaker = target;
    // Follow the global rename — keep the highlight pinned to the renamed speaker.
    if (selectedSpeaker === currentSpeaker) selectedSpeaker = target;
  } else {
    segments[idx].speaker = target;
  }
  recomputeCrosstalk();
  refreshSelectedSpeakerRanges();
  renderTranscript();
  updateDirtyButtons();
  restoreFocus();
});

$("speaker-modal").addEventListener("close", () => restoreFocus());

function restoreFocus() {
  if (speakerModalRow == null) return;
  const row = transcriptEl.querySelector(`.segment[data-index="${speakerModalRow}"]`);
  if (row) {
    const text = row.querySelector(".text");
    if (text) text.focus();
  }
}

// ---------- Save / diff / versions / export ----------

$("btn-save").addEventListener("click", async () => {
  const payload = { segments };
  const r = await fetch("/api/transcript/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!r.ok) { showToast("Save failed: " + r.status); return; }
  const j = await r.json();
  showToast(`Saved ${j.filename}` + (j.wrapped ? " — WRAP! overwrote _00" : ""), j.wrapped ? 8000 : 3000);
  snapshotSaved();
  recomputeCrosstalk();
  await loadVersions();
  updateDiffBaseDropdown();
  updateDirtyButtons();
});

$("btn-diff").addEventListener("click", () => {
  diffMode = !diffMode;
  $("btn-diff").textContent = diffMode ? "Edit" : "Diff";
  renderTranscript();
  updateDirtyButtons();
});

$("btn-export").addEventListener("click", async () => {
  const r = await fetch("/api/transcript/export.txt", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ segments }),
  });
  if (!r.ok) { showToast("Export failed: " + r.status); return; }
  const blob = await r.blob();
  const cd = r.headers.get("content-disposition") || "";
  const m = cd.match(/filename="?([^";]+)"?/i);
  const filename = m ? m[1] : "transcript_export.txt";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$("diff-base-dropdown").addEventListener("change", async (e) => {
  await setDiffBase(e.target.value);
});

$("versions-dropdown").addEventListener("change", async (e) => {
  const fn = e.target.value;
  if (!fn) return;
  const r = await fetch(`/api/transcript/edit/${encodeURIComponent(fn)}`);
  if (!r.ok) { showToast("Load failed"); return; }
  const j = await r.json();
  segments = j.segments || [];
  snapshotSaved();
  recomputeCrosstalk();
  refreshSelectedSpeakerRanges();
  renderTranscript();
  updateDirtyButtons();
});

async function loadVersions() {
  const r = await fetch("/api/session");
  if (!r.ok) return;
  const j = await r.json();
  const sel = $("versions-dropdown");
  sel.innerHTML = `<option value="">(load version: ${j.edits_on_disk.length})</option>`;
  for (const fn of j.edits_on_disk) {
    const opt = document.createElement("option");
    opt.value = fn; opt.textContent = fn;
    sel.appendChild(opt);
  }
}

async function updateDiffBaseDropdown() {
  const r = await fetch("/api/session");
  if (!r.ok) return;
  const j = await r.json();
  const sel = $("diff-base-dropdown");
  if (!sel) return;
  const previous = sel.value;
  sel.innerHTML = `<option value="__original__">vs original</option>`;
  for (const fn of j.edits_on_disk) {
    const opt = document.createElement("option");
    opt.value = fn; opt.textContent = `vs ${fn}`;
    sel.appendChild(opt);
  }
  // Preserve previous selection if still valid; else default to original.
  sel.value = (previous && Array.from(sel.options).some(o => o.value === previous)) ? previous : "__original__";
}

async function setDiffBase(value) {
  if (value === "__original__" || !value) {
    diffBaseSegments = JSON.parse(JSON.stringify(originalSegments));
    diffBaseLabel = "original";
  } else {
    const r = await fetch(`/api/transcript/edit/${encodeURIComponent(value)}`);
    if (!r.ok) { showToast("Couldn't load version: " + r.status); return; }
    const j = await r.json();
    diffBaseSegments = JSON.parse(JSON.stringify(j.segments || []));
    diffBaseLabel = value;
  }
  if (diffMode) renderTranscript();
}

async function loadWaveform() {
  const r = await fetch("/api/waveform");
  if (!r.ok) {
    showToast("Waveform unavailable: " + r.status);
    return;
  }
  const j = await r.json();
  peaks = new Float32Array(j.peaks);
  peaksDuration = j.duration_s;
  zoomTo(1);
}

// ---------- Minutes panel ----------

const SECTION_LABELS = {
  agenda: "Agenda",
  discussion: "Discussion",
  decisions: "Decisions",
  action_items: "Action items",
  next_steps: "Next steps",
  tensions: "Tensions / statements of interest",
};
const SECTION_ORDER = ["agenda", "discussion", "decisions", "action_items", "next_steps", "tensions"];

function setMinutesStatus(text, kind) {
  const el = $("minutes-status");
  if (!text) { el.classList.add("hidden"); el.textContent = ""; return; }
  el.classList.remove("hidden", "error", "busy");
  if (kind === "error") el.classList.add("error");
  if (kind === "busy") el.classList.add("busy");
  el.textContent = text;
}

function prefillAttendees() {
  const input = $("minutes-attendees");
  if (input.value.trim()) return;
  input.value = uniqueSpeakers(segments).join(", ");
}

function readAttendees() {
  return $("minutes-attendees").value.split(",").map(s => s.trim()).filter(Boolean);
}

function renderMinutes() {
  const body = $("minutes-body");
  body.innerHTML = "";
  if (!minutes) {
    const empty = document.createElement("div");
    empty.id = "minutes-empty";
    empty.className = "muted";
    empty.textContent = "No minutes yet — click Generate to create them from the transcript. (Sends transcript content to Anthropic; opt-in.)";
    body.appendChild(empty);
    $("btn-minutes-save").disabled = true;
    return;
  }
  const sections = minutes.sections || {};
  for (const key of SECTION_ORDER) {
    const items = sections[key] || [];
    const det = document.createElement("details");
    det.className = "minutes-section" + (key === "tensions" ? " tensions" : "");
    det.open = items.length > 0;
    const sum = document.createElement("summary");
    sum.textContent = `${SECTION_LABELS[key]} (${items.length})`;
    det.appendChild(sum);
    for (const st of items) {
      const row = document.createElement("div");
      row.className = "minutes-statement";
      row.dataset.id = st.id;

      const main = document.createElement("div");
      main.className = "stmt-text";
      main.textContent = st.text || "";
      if (st.quote) {
        const q = document.createElement("span");
        q.className = "stmt-quote";
        q.textContent = `"${st.quote}"`;
        main.appendChild(q);
      }
      row.appendChild(main);

      if (st.ref_resolution === "fuzzy") {
        const flag = document.createElement("span");
        flag.className = "fuzzy-flag";
        flag.textContent = "≈";
        flag.title = "Reference resolved by fuzzy match — anchor may be approximate";
        row.appendChild(flag);
      }

      row.addEventListener("click", () => activateMinuteStatement(st, row));
      det.appendChild(row);
    }
    body.appendChild(det);
  }
  $("btn-minutes-save").disabled = !minutesDirty;
}

function segRangeFor(idx) {
  const s = segments[idx];
  if (!s) return null;
  return { start: s.start, end: s.end };
}

function activateMinuteStatement(st, rowEl) {
  // De-mark prior active rows
  document.querySelectorAll(".minutes-statement.active").forEach(r => r.classList.remove("active"));
  if (rowEl) rowEl.classList.add("active");

  const primary = segRangeFor(st.primary_ref);
  const secondaries = (st.secondary_refs || [])
    .map(i => segRangeFor(i))
    .filter(Boolean);
  minutesHighlight = primary ? { primary, secondaries } : null;

  if (primary && peaksDuration > 0) {
    // Pan the waveform window so the highlight is visible (when zoomed).
    if (zoomLevel > 1) {
      const vs = visibleSeconds();
      windowStartTime = Math.max(0, primary.start - vs * 0.25);
      clampWindow();
    }
  }

  // Scroll transcript to primary_ref
  if (typeof st.primary_ref === "number") {
    const row = transcriptEl.querySelector(`.segment[data-index="${st.primary_ref}"]`);
    if (row) row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

async function loadMinutes() {
  const r = await fetch("/api/minutes");
  if (r.status === 404) { minutes = null; minutesDirty = false; renderMinutes(); return; }
  if (!r.ok) { setMinutesStatus("Could not load minutes: " + r.status, "error"); return; }
  minutes = await r.json();
  minutesDirty = false;
  renderMinutes();
  setMinutesStatus(minutes.generated_at ? `Loaded (generated ${minutes.generated_at})` : "Loaded");
}

async function generateMinutes() {
  // First-use confirmation modal per session.
  const ok = sessionStorage.getItem("minutesConsent") === "1";
  if (!ok) {
    const dlg = $("minutes-confirm-modal");
    const result = await new Promise((resolve) => {
      const handler = () => {
        dlg.removeEventListener("close", handler);
        resolve(dlg.returnValue);
      };
      dlg.addEventListener("close", handler);
      dlg.showModal();
    });
    if (result !== "ok") return;
    sessionStorage.setItem("minutesConsent", "1");
  }

  const btn = $("btn-minutes-generate");
  btn.disabled = true;
  setMinutesStatus("Generating minutes — this can take 10–30 seconds…", "busy");
  try {
    const r = await fetch("/api/minutes/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        attendees: readAttendees(),
        model: $("minutes-model").value,
      }),
    });
    if (!r.ok) {
      let detail = `${r.status}`;
      try { const j = await r.json(); detail = j.detail || detail; } catch (_) {}
      setMinutesStatus("Generate failed: " + detail, "error");
      return;
    }
    minutes = await r.json();
    minutesDirty = true;
    renderMinutes();
    setMinutesStatus(`Generated. Save to persist.`);
  } catch (e) {
    setMinutesStatus("Generate failed: " + e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function saveMinutes() {
  if (!minutes) return;
  const r = await fetch("/api/minutes/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(minutes),
  });
  if (!r.ok) { setMinutesStatus("Save failed: " + r.status, "error"); return; }
  const j = await r.json();
  minutesDirty = false;
  renderMinutes();
  setMinutesStatus(`Saved ${j.filename}` + (j.wrapped ? " — WRAP! overwrote _00" : ""));
}

$("btn-minutes-generate").addEventListener("click", generateMinutes);
$("btn-minutes-save").addEventListener("click", saveMinutes);

// ---------- Pane divider drag ----------

(function () {
  const divider = $("pane-divider");
  const minutesEl = $("minutes-panel");
  const split = $("split");
  if (!divider || !minutesEl || !split) return;

  // Restore persisted width.
  const saved = parseInt(localStorage.getItem("minutesPanelWidth") || "0", 10);
  if (saved > 200) minutesEl.style.flexBasis = saved + "px";

  let dragging = false;
  divider.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    divider.classList.add("dragging");
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const splitRect = split.getBoundingClientRect();
    const newWidth = Math.max(280, Math.min(splitRect.width - 200, splitRect.right - e.clientX));
    minutesEl.style.flexBasis = newWidth + "px";
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove("dragging");
    document.body.style.userSelect = "";
    const w = parseInt(getComputedStyle(minutesEl).flexBasis, 10);
    if (w > 0) localStorage.setItem("minutesPanelWidth", String(w));
  });
})();

// ---------- Init ----------

async function init() {
  const r = await fetch("/api/transcript");
  const j = await r.json();
  originalSegments = JSON.parse(JSON.stringify(j.segments || []));
  await loadVersions();
  const sel = $("versions-dropdown");
  if (sel.options.length > 1) {
    const latest = sel.options[sel.options.length - 1].value;
    sel.value = latest;
    const er = await fetch(`/api/transcript/edit/${encodeURIComponent(latest)}`);
    const ej = await er.json();
    segments = ej.segments || [];
  } else {
    segments = JSON.parse(JSON.stringify(originalSegments));
  }
  snapshotSaved();
  diffBaseSegments = JSON.parse(JSON.stringify(originalSegments));
  diffBaseLabel = "original";
  recomputeCrosstalk();
  renderTranscript();
  updateDiffBaseDropdown();
  updateDirtyButtons();
  await loadWaveform();
  prefillAttendees();
  await loadMinutes();
  renderLoop();
}

init();
