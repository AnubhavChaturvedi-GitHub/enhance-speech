import { icons, dropzoneArt, emptyArt, bulkArt } from "./icons.js";
import { StemPlayer, computeGains } from "./player.js";

const $ = (selector, root = document) => root.querySelector(selector);
const view = $("#view");
const fileInput = $("#file-input");

const SUPPORTED = "Audio: wav, mp3, m4a, aac, flac, ogg\nVideo: mp4, mov, m4v";
const STATUS_TEXT = {
  processing: "Processing your file…",
  separating_background: "Separating background…",
  enhancing: "Enhancing speech…",
  removing_echo: "Removing echo…",
  finishing: "Finishing…",
};
const DEFAULTS = { speech: 50, background: 10, music: 10, speechMuted: false, backgroundMuted: false, musicMuted: false, enhanced: true };

const state = {
  route: "/",
  tracks: [],
  uploads: [], // { id, trackName, percent, modelVersion }
  selectedId: null,
  modelVersion: "v2",
  loaded: false,
};
const player = new StemPlayer();
let controls = { ...DEFAULTS };
let detailTrackId = null;
let renderedRoute = null;

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function formatTime(total) {
  if (!Number.isFinite(total)) return "0:00";
  const s = Math.max(0, Math.floor(total));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const seconds = String(s % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${seconds}` : `${minutes}:${seconds}`;
}

function toast(message) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

function dottedBorder(host, radius, dash) {
  // The reference tool draws its dashes as an SVG so they keep even spacing round the corners.
  let svg = host.querySelector(":scope > .dotted-border");
  if (!svg) {
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("dotted-border");
    svg.setAttribute("aria-hidden", "true");
    const rect = document.createElementNS(svg.namespaceURI, "rect");
    rect.setAttribute("stroke", "rgb(92, 92, 224)");
    rect.setAttribute("stroke-dasharray", `${dash} 8`);
    svg.appendChild(rect);
    host.prepend(svg);
    new ResizeObserver(() => size()).observe(host);
  }
  const rect = svg.firstChild;
  function size() {
    const { width, height } = host.getBoundingClientRect();
    rect.setAttribute("x", 1.5);
    rect.setAttribute("y", 1.5);
    rect.setAttribute("width", Math.max(0, width - 3));
    rect.setAttribute("height", Math.max(0, height - 3));
    rect.setAttribute("rx", radius);
  }
  size();
}

function acceptDrops(host) {
  let depth = 0;
  host.addEventListener("dragenter", (e) => {
    e.preventDefault();
    depth += 1;
    host.classList.add("drag-over");
  });
  host.addEventListener("dragover", (e) => e.preventDefault());
  host.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) host.classList.remove("drag-over");
  });
  host.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    host.classList.remove("drag-over");
    if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
  });
}

function pickFiles() {
  fileInput.click();
}
fileInput.addEventListener("change", () => {
  if (fileInput.files?.length) addFiles(fileInput.files);
  fileInput.value = "";
});

// ---------- Uploads ----------

function addFiles(fileList) {
  const files = Array.from(fileList);
  if (location.hash !== "#/enhance") location.hash = "#/enhance";
  // Sequential uploads keep the queue in the order the files were chosen.
  (async () => {
    for (const file of files) await uploadFile(file);
  })();
}

function uploadFile(file) {
  return new Promise((resolve) => {
    const upload = { id: `upload-${crypto.randomUUID()}`, trackName: file.name, percent: 0, modelVersion: state.modelVersion };
    state.uploads.unshift(upload);
    if (!state.selectedId) state.selectedId = upload.id;
    renderWorkspaceParts();
    const form = new FormData();
    form.append("file", file);
    form.append("modelVersion", state.modelVersion);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/tracks");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      upload.percent = e.loaded / e.total;
      updateTrackItem(upload.id);
    };
    xhr.onload = () => {
      state.uploads = state.uploads.filter((u) => u !== upload);
      if (xhr.status >= 200 && xhr.status < 300) {
        const track = JSON.parse(xhr.responseText);
        state.tracks = [track, ...state.tracks.filter((t) => t.id !== track.id)];
        if (state.selectedId === upload.id) state.selectedId = track.id;
      } else {
        let detail = "There was a problem uploading your file.";
        try {
          detail = JSON.parse(xhr.responseText).detail || detail;
        } catch {}
        toast(`${file.name}: ${detail}`);
        if (state.selectedId === upload.id) state.selectedId = null;
      }
      renderWorkspaceParts();
      resolve();
    };
    xhr.onerror = () => {
      state.uploads = state.uploads.filter((u) => u !== upload);
      if (state.selectedId === upload.id) state.selectedId = null;
      toast("The file failed to upload. Check your connection and try again.");
      renderWorkspaceParts();
      resolve();
    };
    xhr.send(form);
  });
}

// ---------- Polling ----------

async function refreshTracks() {
  try {
    const res = await fetch("/api/tracks");
    if (!res.ok) throw new Error(res.statusText);
    const tracks = await res.json();
    const before = JSON.stringify(state.tracks);
    state.tracks = tracks;
    const firstLoad = !state.loaded;
    state.loaded = true;
    if (firstLoad || before !== JSON.stringify(tracks)) {
      autoSelect();
      renderWorkspaceParts();
    }
    // Like a signed-in visit: anyone with files goes straight to them. The logo still reaches the landing page.
    if (firstLoad && !location.hash && tracks.length) location.hash = "#/enhance";
  } catch {
    // The server may be restarting; the next poll will catch up.
  }
}

function autoSelect() {
  const all = [...state.uploads, ...state.tracks];
  if (state.selectedId && all.some((t) => t.id === state.selectedId)) return;
  const usable = state.tracks.find((t) => t.state === "finished") || state.tracks.find((t) => t.state !== "failed");
  state.selectedId = usable?.id || null;
}

setInterval(refreshTracks, 1500);

// ---------- Landing ----------

function renderLanding() {
  player.dispose();
  detailTrackId = null;
  view.innerHTML = `
  <div class="landing">
    <section class="hero row" id="hero">
      <div class="hero-left">
        <h1 class="hero-title">Clean up audio for free with local speech AI</h1>
        <p class="hero-body">Enhance Speech makes voice recordings sound as if they were recorded in a professional podcasting studio.</p>
        <p class="hero-body">Hear a sample:</p>
        <div class="sample-player">
          <button class="circle-button" id="sample-play" style="width:52px;height:52px" aria-label="Play sample">${icons.play}</button>
          <div class="switch-row">
            <button class="big-switch" id="sample-switch" role="switch" aria-checked="false" aria-labelledby="sample-switch-label"></button>
            <label class="switch-label" id="sample-switch-label">Enhance Speech</label>
          </div>
        </div>
      </div>
      <div class="hero-right">
        <h2>Get started</h2>
        <p class="sub">Choose audio or video to enhance.</p>
        <button class="cta-primary" id="hero-choose">Choose files</button>
        <p class="hint">Or drop them anywhere in this box. <a href="#/enhance">Open workspace</a></p>
      </div>
    </section>

    <section class="upgrade-row row" id="features">
      <div class="upgrade-callout">
        <h2 class="upgrade-title">Every feature unlocked, running on your own computer</h2>
        <p class="upgrade-sub">No account, no upload to anyone's cloud, no daily limit.</p>
        <button class="white-pill" id="panel-choose">${icons.sparkle}Start enhancing</button>
        <p class="check-line">${icons.check}Files stay on this machine</p>
      </div>
      <div class="panel-inner">
        <div class="panel-head"><img src="assets/mark.svg" alt="">Included</div>
        <h3 class="panel-group">Enhance Speech</h3>
        <div class="panel-item">${icons.mic}Fixes audio problems after the fact, no perfect mic needed</div>
        <div class="panel-item">${icons.video}Video support for MP4, MOV, and more</div>
        <div class="panel-item">${icons.layers}Bulk upload files for enhancement</div>
        <div class="panel-item">${icons.sliders}Adjust speech, music, and background for a more natural sound</div>
        <div class="panel-item">${icons.infinity}Enhance as much as you like, at any length</div>
        <h3 class="panel-group">Stems and transcripts</h3>
        <div class="panel-item">${icons.download}Download speech, music and background separately</div>
        <div class="panel-item">${icons.text}Get a transcript of any file, also created locally</div>
        <h3 class="panel-group">Private by design</h3>
        <div class="panel-item">${icons.wifiOff}Works offline once the AI models have downloaded</div>
      </div>
    </section>

    <section class="bento-wrap">
      <h2 class="bento-title">Try these features</h2>
      <div class="bento">
        <div class="bento-col">
          <div class="bento-card browser-mock">
            <div class="browser-bar"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i><b>Enhance Speech</b></div>
            <div class="browser-url">localhost:5840/enhance</div>
            <img src="assets/workspace.jpg" alt="The Enhance Speech workspace with a queue of files and a video playing" loading="lazy">
          </div>
          <div class="bento-card">
            <div class="text">
              <h2>Adjust strength</h2>
              <p>Dial in the perfect balance of studio-quality and natural environment sound.</p>
              <div class="variant-enhance" style="margin-top:24px" id="demo-slider-host"></div>
            </div>
          </div>
        </div>
        <div class="bento-col">
          <div class="bento-card">
            <div class="bulk-art">${bulkArt}</div>
            <div class="text"><h2>Bulk Upload</h2><p>Upload multiple files to process in an automatic queue.</p></div>
          </div>
          <div class="bento-card">
            <div class="text"><h2>Enhance more</h2><p>No daily limit and no maximum file size. Your computer sets the pace.</p></div>
          </div>
          <div class="bento-card">
            <div class="text"><h2>Separate stems</h2><p>Lift out the speech, music or background on its own, and keep what you need.</p></div>
          </div>
        </div>
      </div>
    </section>

    <footer class="footer">
      <div class="footer-inner">
        <div class="brand-col">
          <a class="brand" href="#/"><img src="assets/mark.svg" alt="" width="40" height="40"><span>Enhance Speech</span></a>
          <span>Runs locally on your computer.</span>
        </div>
        <div>
          <h3>Use it</h3>
          <a href="#/enhance">Open workspace</a>
          <a href="#/features">What's included</a>
        </div>
        <div>
          <h3>Built with open models</h3>
          <a href="https://github.com/kwatcharasupat/bandit-v2" target="_blank" rel="noreferrer">BandIt v2</a>
          <a href="https://github.com/sarulab-speech/Sidon" target="_blank" rel="noreferrer">Sidon</a>
          <a href="https://github.com/Rikorose/DeepFilterNet" target="_blank" rel="noreferrer">DeepFilterNet</a>
          <a href="https://github.com/modelscope/ClearerVoice-Studio" target="_blank" rel="noreferrer">MossFormer2</a>
        </div>
      </div>
    </footer>
  </div>`;

  const hero = $("#hero");
  dottedBorder(hero, 32, 16);
  acceptDrops(hero);
  $("#hero-choose").addEventListener("click", pickFiles);
  $("#panel-choose").addEventListener("click", pickFiles);
  setupSample();
  const demoHost = $("#demo-slider-host");
  const demo = makeSlider({ value: 90, label: "Strength demo", onChange: () => {} });
  demoHost.appendChild(demo.el);
  if (state.route === "/features") $("#features").scrollIntoView({ block: "start" });
}

function setupSample() {
  const raw = new Audio("assets/sample/original.mp3");
  const clean = new Audio("assets/sample/enhanced.mp3");
  raw.preload = clean.preload = "auto";
  let playing = false;
  let enhanced = false;
  const button = $("#sample-play");
  const toggle = $("#sample-switch");
  const current = () => (enhanced ? clean : raw);
  const stop = () => {
    playing = false;
    button.innerHTML = icons.play;
    button.setAttribute("aria-label", "Play sample");
  };
  raw.addEventListener("ended", stop);
  clean.addEventListener("ended", stop);
  button.addEventListener("click", async () => {
    if (playing) {
      current().pause();
      stop();
      return;
    }
    playing = true;
    button.innerHTML = icons.pause;
    button.setAttribute("aria-label", "Pause sample");
    await current().play().catch(stop);
  });
  const flip = () => {
    const previous = current();
    enhanced = !enhanced;
    toggle.setAttribute("aria-checked", String(enhanced));
    const next = current();
    next.currentTime = previous.currentTime;
    if (playing) {
      previous.pause();
      next.play().catch(stop);
    }
  };
  toggle.addEventListener("click", flip);
  $("#sample-switch-label").addEventListener("click", flip);
  // Stops the sample if the page changes view.
  const end = () => {
    raw.pause();
    clean.pause();
    window.removeEventListener("hashchange", end);
  };
  window.addEventListener("hashchange", end);
}

// ---------- Slider ----------

function makeSlider({ value, label, onChange }) {
  const el = document.createElement("div");
  el.className = "slider";
  el.innerHTML = `<div class="slider-rail"></div><div class="slider-track"></div><button class="slider-handle" role="slider" aria-valuemin="0" aria-valuemax="100" aria-label="${escapeHtml(label)}"></button>`;
  const track = $(".slider-track", el);
  const handle = $(".slider-handle", el);
  const api = {
    el,
    value,
    disabled: false,
    set(next, emit = false) {
      const clamped = Math.round(Math.min(100, Math.max(0, next)));
      api.value = clamped;
      track.style.width = `${100 - clamped}%`;
      // The handle overshoots each end slightly, as in the reference slider.
      const reversed = 100 - clamped;
      let shift = 50;
      if (reversed < 50) shift = Math.max(16, reversed);
      else if (reversed > 50) shift = Math.min(85, reversed);
      handle.style.right = `${reversed}%`;
      handle.style.transform = `translateX(${shift}%)`;
      handle.setAttribute("aria-valuenow", String(clamped));
      handle.setAttribute("aria-valuetext", `${clamped}%`);
      if (emit) onChange(clamped);
    },
    setDisabled(flag) {
      api.disabled = flag;
      el.classList.toggle("is-disabled", flag);
      handle.disabled = flag;
    },
  };
  const fromEvent = (e) => {
    const box = el.getBoundingClientRect();
    return ((e.clientX - box.left) / box.width) * 100;
  };
  el.addEventListener("pointerdown", (e) => {
    if (api.disabled) return;
    el.setPointerCapture(e.pointerId);
    handle.focus({ preventScroll: true });
    api.set(fromEvent(e), true);
    const move = (ev) => api.set(fromEvent(ev), true);
    const up = () => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  });
  handle.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 }[e.key];
    if (step !== undefined) {
      e.preventDefault();
      api.set(api.value + step, true);
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      api.set(e.key === "Home" ? 0 : 100, true);
    }
  });
  api.set(value);
  return api;
}

// ---------- Workspace ----------

function renderWorkspaceShell() {
  view.innerHTML = `
  <div class="workspace-page">
    <div class="workspace" id="workspace">
      <div class="dropzone" id="dropzone">
        <div class="floating-options">
          <span class="icon-bare has-tip" tabindex="0" aria-label="Supported files">${icons.info}<span class="tooltip"><b>Supported files</b>\n${SUPPORTED}</span></span>
          <span class="version-picker has-tip">
            <select class="version-select" id="version" aria-label="Enhance Speech version">
              <option value="v2">v2</option>
              <option value="v1">v1</option>
            </select>
            <span class="tooltip">v2: full studio restoration with speech, music and background sliders.\nv1: quicker clean-up that blends with the original.</span>
          </span>
        </div>
        <div class="dropzone-content" id="dropzone-content" role="button" tabindex="0" aria-label="Choose files to upload">
          ${dropzoneArt}
          <h2 class="dropzone-title">Enhance</h2>
          <button class="outline-button" id="choose">Choose files</button>
        </div>
      </div>
      <div id="left-list" style="display:contents"></div>
      <div id="right-pane" style="display:contents"></div>
    </div>
  </div>`;
  const zone = $("#dropzone");
  dottedBorder(zone, 18, 16);
  acceptDrops(zone);
  const content = $("#dropzone-content");
  content.addEventListener("click", pickFiles);
  content.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      pickFiles();
    }
  });
  $("#choose").addEventListener("click", (e) => {
    e.stopPropagation();
    pickFiles();
  });
  const version = $("#version");
  version.value = state.modelVersion;
  version.addEventListener("change", () => {
    state.modelVersion = version.value;
  });
  renderWorkspaceParts();
}

function allItems() {
  const uploads = state.uploads.map((u) => ({ ...u, state: "uploading" }));
  return [...uploads, ...state.tracks];
}

function statusLine(item) {
  switch (item.state) {
    case "uploading":
      return "Uploading…";
    case "queued":
      return "Queued…";
    case "processing":
      return "Enhancing speech…";
    case "failed":
      return `Enhance failed <span class="icon-bare has-tip" style="width:18px;height:18px">${icons.info}<span class="tooltip" style="left:0;transform:none">${escapeHtml(item.error || "")}</span></span>`;
    default:
      return formatTime(item.duration);
  }
}

function ringSvg(fraction, spinning = false) {
  const r = 12;
  const c = 2 * Math.PI * r;
  const shown = spinning ? 0.25 : Math.max(0.02, fraction);
  return `<svg class="ring ${spinning ? "spin" : ""}" viewBox="0 0 28 28" aria-hidden="true"><circle class="track" cx="14" cy="14" r="${r}"/><circle class="value" cx="14" cy="14" r="${r}" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - shown)}"/></svg>`;
}

function trackItemHtml(item) {
  const selected = item.id === state.selectedId;
  const pending = item.state === "uploading" || item.state === "queued" || item.state === "failed";
  const classes = ["track-item", selected ? "selected" : "", pending && !selected ? "is-pending" : ""].join(" ");
  const busy = item.state === "uploading" || item.state === "processing";
  const fraction = item.state === "uploading" ? item.percent : item.progress;
  const ring = busy && !selected ? ringSvg(fraction || 0, item.state === "uploading" && !item.percent) : "";
  const deleteBtn = item.state === "uploading" ? "" : `<button class="delete-button" data-delete="${item.id}" aria-label="Delete">${icons.trash}</button>`;
  return `
  <div class="${classes}" data-track="${item.id}" tabindex="0" aria-selected="${selected}">
    <div class="track-info">
      <div class="title-group"><span class="track-title" title="${escapeHtml(item.trackName)}">${escapeHtml(item.trackName)}</span><span class="version-badge">${item.modelVersion || "v2"}</span></div>
      <div class="track-status">${statusLine(item)}</div>
    </div>
    <div class="track-actions">${ring}${deleteBtn}</div>
  </div>`;
}

function htmlToElement(html) {
  const holder = document.createElement("template");
  holder.innerHTML = html.trim();
  return holder.content.firstElementChild;
}

// Patches a row in place. Rebuilding it on every progress update could swallow a click on Delete.
function patchRow(row, fresh) {
  row.className = fresh.className;
  row.setAttribute("aria-selected", fresh.getAttribute("aria-selected"));
  const syncInner = (selector) => {
    const before = row.querySelector(selector);
    const after = fresh.querySelector(selector);
    if (before.innerHTML !== after.innerHTML) before.innerHTML = after.innerHTML;
  };
  syncInner(".title-group");
  syncInner(".track-status");
  const oldRing = row.querySelector(".ring");
  const newRing = fresh.querySelector(".ring");
  const actions = row.querySelector(".track-actions");
  if (oldRing && newRing) {
    oldRing.setAttribute("class", newRing.getAttribute("class"));
    oldRing.querySelector(".value").setAttribute("stroke-dashoffset", newRing.querySelector(".value").getAttribute("stroke-dashoffset"));
  } else if (newRing) {
    actions.prepend(newRing);
  } else if (oldRing) {
    oldRing.remove();
  }
  const oldDelete = row.querySelector(".delete-button");
  const newDelete = fresh.querySelector(".delete-button");
  if (newDelete && !oldDelete) actions.append(newDelete);
  else if (!newDelete && oldDelete) oldDelete.remove();
}

function updateTrackItem(id) {
  const row = view.querySelector(`[data-track="${id}"]`);
  const item = allItems().find((t) => t.id === id);
  if (!row || !item) return;
  patchRow(row, htmlToElement(trackItemHtml(item)));
  if (id === state.selectedId) renderRight();
}

function renderLeft() {
  const host = $("#left-list");
  if (!host) return;
  const items = allItems();
  const workspace = $("#workspace");
  workspace.classList.toggle("is-empty", items.length === 0);
  if (!items.length) {
    host.innerHTML = `<div class="null-tracks">${'<div class="null-track"></div>'.repeat(4)}<div class="null-overlay"></div></div>`;
    return;
  }
  let list = $(".track-list", host);
  if (!list) {
    host.innerHTML = `
    <div class="track-list-wrapper">
      <div class="track-list" id="track-list">
        <div class="track-footer">Files stay on this computer until you delete them.</div>
      </div>
    </div>`;
    list = $(".track-list", host);
  }
  const footer = $(".track-footer", list);
  const existing = new Map([...list.querySelectorAll("[data-track]")].map((row) => [row.dataset.track, row]));
  let cursor = list.firstElementChild;
  for (const item of items) {
    const fresh = htmlToElement(trackItemHtml(item));
    let row = existing.get(item.id);
    if (row) {
      patchRow(row, fresh);
      existing.delete(item.id);
    } else {
      row = fresh;
    }
    if (row === cursor) cursor = cursor.nextElementSibling;
    else list.insertBefore(row, cursor);
  }
  existing.forEach((row) => row.remove());
  if (footer.nextElementSibling) list.append(footer);
}

// Delegated, so re-rendering the list never drops a handler.
view.addEventListener("click", (e) => {
  const del = e.target.closest("[data-delete]");
  if (del) {
    e.stopPropagation();
    // The row below moves up as soon as this one goes, so a double-click's second click would delete it too.
    if (e.detail <= 1) removeTrack(del.dataset.delete);
    return;
  }
  const row = e.target.closest("[data-track]");
  if (row) selectTrack(row.dataset.track);
});
view.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && e.target.matches?.("[data-track]")) {
    e.preventDefault();
    selectTrack(e.target.dataset.track);
  }
});

function selectTrack(id) {
  if (state.selectedId === id) return;
  state.selectedId = id;
  renderWorkspaceParts();
}

async function removeTrack(id) {
  const response = await fetch(`/api/tracks/${id}`, { method: "DELETE" }).catch(() => null);
  if (!response?.ok) {
    toast("Failed to delete file.");
    return;
  }
  state.tracks = state.tracks.filter((t) => t.id !== id);
  if (state.selectedId === id) state.selectedId = null;
  autoSelect();
  renderWorkspaceParts();
}

function renderWorkspaceParts() {
  if (state.route !== "/enhance" || !$("#workspace")) return;
  renderLeft();
  renderRight();
}

function renderRight() {
  const host = $("#right-pane");
  if (!host) return;
  const item = allItems().find((t) => t.id === state.selectedId);
  const items = allItems();
  if (!item) {
    teardownDetail(host);
    host.innerHTML = items.length
      ? ""
      : `<div class="empty-state">${emptyArt}<h3>No enhanced files yet</h3><p>Fine tune your files here once they finish enhancing.</p></div>`;
    return;
  }
  if (item.state === "finished") {
    if (detailTrackId !== item.id) buildDetail(host, item);
    return;
  }
  teardownDetail(host);
  if (item.state === "failed") {
    host.innerHTML = `<div class="processing-detail"><p class="processing-text failed-detail">${escapeHtml(item.error || "There was an unknown problem enhancing the file.")}</p></div>`;
    return;
  }
  const fraction = item.state === "uploading" ? item.percent || 0 : item.progress || 0;
  const label = item.state === "uploading" ? "Uploading…" : item.state === "queued" ? "Queued…" : STATUS_TEXT[item.stage] || STATUS_TEXT.processing;
  let box = host.querySelector(".processing-detail[data-live]");
  if (!box) {
    host.innerHTML = `<div class="processing-detail" data-live role="status" aria-live="polite"><span class="ring-slot"></span><p class="processing-text"></p><p class="processing-percent"></p></div>`;
    box = host.querySelector(".processing-detail");
  }
  $(".ring-slot", box).innerHTML = ringSvg(fraction, item.state === "queued");
  $(".processing-text", box).textContent = label;
  $(".processing-percent", box).textContent = item.state === "queued" ? "" : `${Math.round(fraction * 100)}%`;
}

function teardownDetail(host) {
  if (detailTrackId !== null) {
    player.dispose();
    detailTrackId = null;
    host.innerHTML = "";
  }
  document.querySelector(".action-bar")?.remove();
}

function strengthCard(key, title, info, variant) {
  return `
  <div class="strength-card ${variant}" data-card="${key}">
    <div class="strength-head">
      <div class="strength-label">
        <h2 class="strength-title">${title}</h2>
        <span class="icon-bare has-tip" tabindex="0" aria-label="${escapeHtml(info)}">${icons.info}<span class="tooltip top">${escapeHtml(info)}</span></span>
      </div>
      <div class="strength-buttons">
        <button class="stem-button has-tip" data-mute="${key}" aria-label="Mute ${key}"><span class="tooltip top">Mute track</span></button>
        <button class="stem-button has-tip" data-stem="${key}" aria-label="Download ${key} track">${icons.download}<span class="tooltip top">Download track</span></button>
        <span class="strength-value"></span>
      </div>
    </div>
    <div class="slider-area" data-area="${key}"></div>
  </div>`;
}

function buildDetail(host, track) {
  detailTrackId = track.id;
  controls = { ...DEFAULTS };
  const v2 = track.modelVersion !== "v1";
  host.innerHTML = `
  <div class="track-detail" id="detail">
    ${track.isVideo ? '<div class="video-box" id="video-box"></div>' : ""}
    <div class="player-row">
      <button class="circle-button" id="play" style="width:48px;height:48px" aria-label="Play ${escapeHtml(track.trackName)}">${icons.play}</button>
      <span class="timecode" id="now">0:00</span>
      <div class="scrubber" id="scrubber">
        <div class="scrubber-rail"><div class="scrubber-buffer" id="buffer"></div></div>
        <div class="scrubber-track" id="progress"></div>
        <div class="scrubber-handle" id="handle" role="slider" tabindex="0" aria-label="Player current time" aria-valuemin="0"></div>
      </div>
      <span class="timecode" id="total">${formatTime(track.duration)}</span>
    </div>
    <div class="controls">
      <div class="control-toggle">
        <button class="big-switch" id="enhance-switch" role="switch" aria-checked="true" aria-labelledby="enhance-label"></button>
        <label class="switch-label" id="enhance-label">Enhance Speech</label>
      </div>
      ${strengthCard("speech", "Speech", "Adjust how much enhancement to apply to the isolated speech", "variant-enhance")}
      ${v2 ? strengthCard("music", "Music", "Adjust the loudness of the isolated music", "variant-music") : ""}
      ${v2 ? strengthCard("background", "Background", "Adjust the loudness of the isolated background noise", "variant-background") : ""}
    </div>
  </div>`;

  // Action bar lives at the page level because it is fixed to the viewport.
  document.querySelector(".action-bar")?.remove();
  const bar = document.createElement("div");
  bar.className = "action-bar";
  bar.innerHTML = `
    <div class="action-group">
      <button class="action-button transcript" id="get-transcript">${icons.text}Get transcript</button>
      <button class="action-button video" id="get-captions">${icons.captions}Get captions</button>
      <button class="action-button download" id="download">${icons.download}Download</button>
    </div>`;
  document.body.appendChild(bar);

  const sliders = {};
  for (const key of v2 ? ["speech", "music", "background"] : ["speech"]) {
    const area = host.querySelector(`[data-area="${key}"]`);
    const slider = makeSlider({
      value: controls[key],
      label: `${key} strength`,
      onChange: (value) => {
        controls[key] = value;
        syncControls();
      },
    });
    area.appendChild(slider.el);
    area.insertAdjacentHTML("beforeend", '<div class="strength-ends"><div>Less</div><div>More</div></div>');
    sliders[key] = slider;
  }

  function syncControls() {
    for (const key of Object.keys(sliders)) {
      const card = host.querySelector(`[data-card="${key}"]`);
      const muted = controls[`${key}Muted`];
      $(".strength-value", card).textContent = `${controls[key]}%`;
      $(".strength-title", card).classList.toggle("muted", muted);
      card.classList.toggle("disabled", !controls.enhanced);
      const area = host.querySelector(`[data-area="${key}"]`);
      area.classList.toggle("muted", muted);
      sliders[key].setDisabled(!controls.enhanced || muted);
      const muteBtn = card.querySelector("[data-mute]");
      if (!v2) muteBtn.hidden = true;
      muteBtn.classList.toggle("on", muted);
      muteBtn.innerHTML = `${muted ? icons.muted : icons.volume}<span class="tooltip top">${muted ? "Unmute track" : "Mute track"}</span>`;
      const stemBtn = card.querySelector("[data-stem]");
      if (!v2) stemBtn.hidden = true;
      stemBtn.disabled = stemBtn.dataset.busy === "1" || muted || !controls.enhanced;
    }
    player.applyGains(computeGains({ mode: track.modelVersion, ...controls }));
  }

  host.querySelectorAll("[data-mute]").forEach((button) =>
    button.addEventListener("click", () => {
      const key = `${button.dataset.mute}Muted`;
      controls[key] = !controls[key];
      syncControls();
    }),
  );
  host.querySelectorAll("[data-stem]").forEach((button) =>
    button.addEventListener("click", () => exportStem(track, button.dataset.stem, button)),
  );
  $("#enhance-switch").addEventListener("click", toggleEnhance);
  $("#enhance-label").addEventListener("click", toggleEnhance);
  function toggleEnhance() {
    controls.enhanced = !controls.enhanced;
    $("#enhance-switch").setAttribute("aria-checked", String(controls.enhanced));
    syncControls();
  }
  $("#download").addEventListener("click", () => exportFull(track));
  $("#get-transcript").addEventListener("click", (e) => exportTranscript(track, "txt", e.currentTarget));
  $("#get-captions").addEventListener("click", (e) => exportTranscript(track, "srt", e.currentTarget));

  const playButton = $("#play");
  const nowLabel = $("#now");
  const totalLabel = $("#total");
  const progressBar = $("#progress");
  const handle = $("#handle");
  const bufferBar = $("#buffer");
  const scrubber = $("#scrubber");
  let duration = track.duration || 0;
  let scrubbing = false;
  const paint = (time) => {
    const f = duration ? Math.min(1, time / duration) : 0;
    progressBar.style.width = `${f * 100}%`;
    handle.style.left = `${f * 100}%`;
    handle.setAttribute("aria-valuenow", String(Math.round(time)));
    handle.setAttribute("aria-valuetext", formatTime(time));
    nowLabel.textContent = formatTime(time);
  };
  const element = player.load(`/api/tracks/${track.id}/media`, track.isVideo, {
    onTime: (t) => {
      if (!scrubbing) paint(t);
    },
    onDuration: (d) => {
      if (Number.isFinite(d) && d > 0) {
        duration = d;
        totalLabel.textContent = formatTime(d);
        handle.setAttribute("aria-valuemax", String(Math.round(d)));
      }
    },
    onBuffered: (f) => {
      bufferBar.style.width = `${f * 100}%`;
    },
    onPlaying: (isPlaying) => {
      playButton.innerHTML = isPlaying ? icons.pause : icons.play;
      playButton.setAttribute("aria-label", `${isPlaying ? "Pause" : "Play"} ${track.trackName}`);
    },
    onError: () => toast("Preview failed to load. Download to listen to your enhanced file."),
  });
  if (track.isVideo) {
    const box = $("#video-box");
    box.appendChild(element);
    box.addEventListener("click", () => player.toggle());
  }
  playButton.addEventListener("click", () => player.toggle());
  const seekFrom = (e) => {
    const r = scrubber.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * duration;
  };
  scrubber.addEventListener("pointerdown", (e) => {
    scrubbing = true;
    scrubber.setPointerCapture(e.pointerId);
    paint(seekFrom(e));
    const move = (ev) => paint(seekFrom(ev));
    const up = (ev) => {
      player.seek(seekFrom(ev));
      scrubbing = false;
      scrubber.removeEventListener("pointermove", move);
      scrubber.removeEventListener("pointerup", up);
      scrubber.removeEventListener("pointercancel", up);
    };
    scrubber.addEventListener("pointermove", move);
    scrubber.addEventListener("pointerup", up);
    scrubber.addEventListener("pointercancel", up);
  });
  handle.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: -5, ArrowRight: 5 }[e.key];
    if (step && element) {
      e.preventDefault();
      player.seek(Math.max(0, Math.min(duration, element.currentTime + step)));
    }
  });
  paint(0);
  syncControls();
}

// ---------- Exports ----------

function exportQuery(extra = {}) {
  const params = new URLSearchParams({
    speech: controls.speech,
    speechMuted: controls.speechMuted,
    background: controls.background,
    backgroundMuted: controls.backgroundMuted,
    music: controls.music,
    musicMuted: controls.musicMuted,
    enhanced: controls.enhanced,
    ...extra,
  });
  return params.toString();
}

function saveLink(url, name) {
  const link = document.createElement("a");
  link.href = url;
  if (name) link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

// The server mixes the file first, then the browser's own download takes it straight to disk.
async function downloadExport(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("export failed");
  const { url: fileUrl } = await response.json();
  saveLink(fileUrl);
}

async function exportFull(track) {
  const button = $("#download");
  if (!button || button.disabled) return;
  button.disabled = true;
  const original = button.innerHTML;
  button.innerHTML = '<span class="btn-spinner"></span>Download';
  try {
    await downloadExport(`/api/tracks/${track.id}/export?${exportQuery()}`);
  } catch {
    toast("An error occurred while downloading the file. Please try again.");
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

async function exportTranscript(track, format, button) {
  if (button.disabled) return;
  button.disabled = true;
  const original = button.innerHTML;
  const label = button.textContent;
  button.innerHTML = `<span class="btn-spinner"></span>${escapeHtml(label)}`;
  try {
    // Whisper can take a while, so wait for the file here to keep the spinner honest.
    const url = `/api/tracks/${track.id}/transcript?format=${format}`;
    const response = await fetch(url, { method: "HEAD" });
    if (!response.ok) throw new Error("transcript failed");
    saveLink(url);
  } catch {
    toast("The transcript could not be created. Please try again.");
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

async function exportStem(track, component, button) {
  if (button.disabled) return;
  button.disabled = true;
  button.dataset.busy = "1";
  const original = button.innerHTML;
  button.innerHTML = '<span class="btn-spinner" style="width:18px;height:18px"></span>';
  try {
    await downloadExport(`/api/tracks/${track.id}/export?${exportQuery({ component })}`);
  } catch {
    toast("An error occurred while downloading the file. Please try again.");
  } finally {
    delete button.dataset.busy;
    button.disabled = controls[`${component}Muted`] || !controls.enhanced;
    button.innerHTML = original;
  }
}

// ---------- Router ----------

function route() {
  const path = location.hash.replace(/^#/, "") || "/";
  state.route = path === "/enhance" ? "/enhance" : path === "/features" ? "/features" : "/";
  document.querySelectorAll(".main-nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${state.route}`));
  $(".header-cta").hidden = state.route === "/enhance";
  const shown = state.route === "/enhance" ? "/enhance" : "/";
  document.body.style.overflow = shown === "/enhance" && innerWidth > 768 ? "hidden" : "";
  if (shown !== renderedRoute) {
    document.querySelector(".action-bar")?.remove();
    renderedRoute = shown;
    if (shown === "/enhance") {
      detailTrackId = null;
      renderWorkspaceShell();
    } else {
      renderLanding();
    }
    window.scrollTo(0, 0);
  } else if (state.route === "/features") {
    $("#features")?.scrollIntoView({ block: "start" });
  }
  autoSelect();
  renderWorkspaceParts();
}
window.addEventListener("hashchange", route);
// Dropping a file outside the zones must not navigate the browser away from the app.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

route();
refreshTracks();
