// Diarizer transcript-review frontend. Vanilla JS, no build step.

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const transcriptEl = $("transcript");
const canvas = $("spectrogram");
const ctx2d = canvas.getContext("2d");

let audioCtx = null;
let analyser = null;
let freqData = null;
let segments = [];   // current edit state
let originalSegments = [];
let speakers = [];   // unique speakers in current state
let diffMode = false;

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

// ---------- Web Audio wiring ----------

async function ensureAudioCtx() {
  if (audioCtx) {
    if (audioCtx.state === "suspended") await audioCtx.resume();
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaElementSource(audio);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  analyser.connect(audioCtx.destination);   // critical — without this, no audio.
  freqData = new Uint8Array(analyser.frequencyBinCount);
  if (audioCtx.state === "suspended") await audioCtx.resume();
}

// ---------- Spectrogram render ----------

let scrollX = 0;
function drawSpectrogram() {
  requestAnimationFrame(drawSpectrogram);
  const w = canvas.width, h = canvas.height;
  if (!analyser) {
    ctx2d.fillStyle = "#000";
    ctx2d.fillRect(0, 0, w, h);
    return;
  }
  analyser.getByteFrequencyData(freqData);
  // Scroll existing image left by 1px, draw new column at right.
  const imageData = ctx2d.getImageData(1, 0, w - 1, h);
  ctx2d.putImageData(imageData, 0, 0);
  ctx2d.fillStyle = "#000";
  ctx2d.fillRect(w - 1, 0, 1, h);
  const bins = freqData.length;
  // Log-frequency Y (bottom = low freq).
  for (let y = 0; y < h; y++) {
    const frac = 1 - y / h;
    const binIdx = Math.floor(Math.pow(frac, 2) * bins);
    const v = freqData[binIdx] || 0;
    ctx2d.fillStyle = viridis(v / 255);
    ctx2d.fillRect(w - 1, y, 1, 1);
  }
  // Playhead overlay (vertical line near right edge — playback edge).
  ctx2d.strokeStyle = "#fff";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(w - 1, 0);
  ctx2d.lineTo(w - 1, h);
  ctx2d.stroke();
}

function viridis(t) {
  // 4-stop interpolation; close enough.
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [253, 231, 37],
  ];
  const idx = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
  const local = t * (stops.length - 1) - idx;
  const a = stops[idx], b = stops[idx + 1];
  const r = a[0] + (b[0] - a[0]) * local;
  const g = a[1] + (b[1] - a[1]) * local;
  const bl = a[2] + (b[2] - a[2]) * local;
  return `rgb(${r|0},${g|0},${bl|0})`;
}

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

    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = seg.speaker || "—";
    speaker.title = "Click to reassign this segment's speaker";
    speaker.addEventListener("click", (e) => { e.stopPropagation(); openReassignModal(i); });

    const ts = document.createElement("div");
    ts.className = "ts";
    ts.textContent = fmtTime(seg.start);
    ts.title = "Click to seek";
    ts.addEventListener("click", () => { audio.currentTime = seg.start; });

    const text = document.createElement("div");
    text.className = "text";
    text.contentEditable = "true";
    text.spellcheck = true;
    text.textContent = seg.text || "";
    text.addEventListener("input", () => { segments[i].text = text.textContent; });

    row.appendChild(speaker);
    row.appendChild(ts);
    row.appendChild(text);
    transcriptEl.appendChild(row);
  }
}

function renderDiff() {
  transcriptEl.innerHTML = "";
  const n = Math.max(segments.length, originalSegments.length);
  for (let i = 0; i < n; i++) {
    const o = originalSegments[i];
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
      speaker.innerHTML = `<span class="diff-old">${o.speaker || "—"}</span><span class="diff-new">${e.speaker || "—"}</span>`;
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
  if (diffMode) return;
  const now = audio.currentTime;
  const rows = transcriptEl.querySelectorAll(".segment");
  for (const row of rows) {
    const i = parseInt(row.dataset.index, 10);
    const seg = segments[i];
    if (!seg) continue;
    const active = now >= seg.start && now < seg.end;
    row.classList.toggle("active", active);
  }
}
audio.addEventListener("timeupdate", () => {
  $("time-now").textContent = fmtTime(audio.currentTime);
  tickHighlight();
});
audio.addEventListener("loadedmetadata", () => {
  $("time-total").textContent = fmtTime(audio.duration);
});

// ---------- Player controls ----------

$("btn-playpause").addEventListener("click", async () => {
  await ensureAudioCtx();
  if (audio.paused) { await audio.play(); $("btn-playpause").textContent = "Pause"; }
  else { audio.pause(); $("btn-playpause").textContent = "Play"; }
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

// ---------- Speaker rename / reassign ----------

let reassignIndex = -1;
function openReassignModal(idx) {
  reassignIndex = idx;
  const sel = $("reassign-to");
  sel.innerHTML = "";
  for (const sp of speakers) {
    const opt = document.createElement("option");
    opt.value = sp; opt.textContent = sp;
    sel.appendChild(opt);
  }
  sel.value = segments[idx].speaker || "";
  $("reassign-modal").showModal();
}
$("reassign-ok").addEventListener("click", () => {
  if (reassignIndex >= 0) {
    segments[reassignIndex].speaker = $("reassign-to").value;
    renderTranscript();
  }
});

// Global rename: reuse reassign modal isn't right — wire the rename modal.
// Add a "Rename" toolbar button via keyboard shortcut: shift+R.
window.addEventListener("keydown", (e) => {
  if (e.shiftKey && (e.key === "R" || e.key === "r")) openRenameModal();
});

function openRenameModal() {
  const sel = $("rename-from");
  sel.innerHTML = "";
  for (const sp of speakers) {
    const opt = document.createElement("option");
    opt.value = sp; opt.textContent = sp;
    sel.appendChild(opt);
  }
  $("rename-to").value = "";
  $("rename-modal").showModal();
}
$("rename-ok").addEventListener("click", () => {
  const from = $("rename-from").value;
  const to = $("rename-to").value.trim();
  if (!from || !to) return;
  for (const seg of segments) if (seg.speaker === from) seg.speaker = to;
  renderTranscript();
});

// ---------- Save / diff / versions ----------

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
  await loadVersions();
});

$("btn-diff").addEventListener("click", () => {
  diffMode = !diffMode;
  $("btn-diff").textContent = diffMode ? "Edit" : "Diff";
  renderTranscript();
});

$("versions-dropdown").addEventListener("change", async (e) => {
  const fn = e.target.value;
  if (!fn) return;
  const r = await fetch(`/api/transcript/edit/${encodeURIComponent(fn)}`);
  if (!r.ok) { showToast("Load failed"); return; }
  const j = await r.json();
  segments = j.segments || [];
  renderTranscript();
});

async function loadVersions() {
  const r = await fetch("/api/session");
  if (!r.ok) return;
  const j = await r.json();
  const sel = $("versions-dropdown");
  sel.innerHTML = `<option value="">(versions: ${j.edits_on_disk.length})</option>`;
  for (const fn of j.edits_on_disk) {
    const opt = document.createElement("option");
    opt.value = fn; opt.textContent = fn;
    sel.appendChild(opt);
  }
}

// ---------- Init ----------

async function init() {
  // Load original
  const r = await fetch("/api/transcript");
  const j = await r.json();
  originalSegments = JSON.parse(JSON.stringify(j.segments || []));
  // If there are existing edits, load the latest as current state; else fresh copy of original.
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
  renderTranscript();
  drawSpectrogram();
}

init();
