/* GISAXS pixel converter: UI, file upload, Plotly rendering, downloads.
 *
 * All analysis runs in Python (Pyodide) inside js/worker.js; this file only
 * sends commands, shows results and draws.  Display options (log scale,
 * colormap, colour range, point size) are applied here without calling
 * Python, so toggling them never re-runs the conversion.
 */
"use strict";

(function () {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const MINUS = "−";

function num(v, digits = 6) {
  if (v == null || !Number.isFinite(v)) return "–";
  let s = Number(v).toPrecision(digits);
  if (!s.includes("e") && s.includes(".")) s = s.replace(/\.?0+$/, "");
  return s.replace(/^-/, MINUS);
}
function fixed(v, d) {
  if (v == null || !Number.isFinite(v)) return "–";
  return v.toFixed(d).replace(/^-/, MINUS);
}
function signed(v, d) {
  if (v == null || !Number.isFinite(v)) return "–";
  return (v >= 0 ? "+" : MINUS) + Math.abs(v).toFixed(d);
}
function int(n) {
  return n == null ? "–" : Math.round(n).toLocaleString("en-US");
}
function pct(part, whole) {
  return whole ? `${(100 * part / whole).toFixed(1)} %` : "–";
}
function mb(bytes) {
  return bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} kB`;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function capitalise(s) {
  s = String(s || "");
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function storeGet(key) {
  try { return window.localStorage.getItem(key); } catch (_) { return null; }
}
function storeSet(key, value) {
  try { window.localStorage.setItem(key, value); } catch (_) { /* private mode */ }
}

const UNITS = {
  "nm-1": { factor: 1, html: "nm<sup>−1</sup>", suffix: "nm-1", digits: 4 },
  "A-1": { factor: 0.1, html: "Å<sup>−1</sup>", suffix: "A-1", digits: 5 },
};

const DATA_DIR = "/home/pyodide/data";

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const state = {
  ready: false,
  init: null,            // colours, colormaps, versions (from Python)
  files: { dat: null, pro: null },
  loadToken: 0,          // bumped for every new pair of files
  pro: null,             // load_pro result
  dat: null,             // load_dat result
  conv: null,            // convert result
  qUnit: "nm-1",
  version: 0,            // data version in Python (changes with masks)
  maskKind: "none",
  rod: null,             // "yes" | "no" | null
  spot: null,
  plotsUnlocked: false,
};

// ---------------------------------------------------------------------------
// status bar
// ---------------------------------------------------------------------------

const statusEl = $("#status");
const statusText = $("#status-text");
let busyCount = 0;

function setStatus(text, kind) {
  statusEl.dataset.state = kind;
  statusText.textContent = text;
}
function idle(text) {
  if (busyCount === 0) setStatus(text, "ready");
}

// ---------------------------------------------------------------------------
// the Python worker
// ---------------------------------------------------------------------------

const worker = new Worker("js/worker.js", { type: "module" });
const pending = new Map();
let nextId = 1;

worker.onmessage = ({ data: m }) => {
  switch (m.type) {
    case "progress":
      if (busyCount > 0 || !state.ready) setStatus(m.message, "busy");
      break;
    case "ready":
      onReady(m.meta, m.pyodide);
      break;
    case "fatal":
      onFatal(m.message);
      break;
    case "chunk": {
      const p = pending.get(m.id);
      if (p) p.chunks.push(m.data);
      break;
    }
    case "result": {
      const p = pending.get(m.id);
      pending.delete(m.id);
      if (p) p.resolve({ meta: m.meta, arrays: m.arrays, chunks: p.chunks });
      break;
    }
    case "error": {
      const p = pending.get(m.id);
      pending.delete(m.id);
      if (p) {
        const err = new Error(m.message);
        err.traceback = m.traceback;
        err.userError = m.userError;
        p.reject(err);
      }
      break;
    }
  }
};
worker.onerror = (event) => onFatal(event.message || "the Python worker failed to start");

/** Run a command in Python (py/bridge.py). files: [{path, file}] */
function py(cmd, args = {}, files = [], label = "Working…") {
  const id = nextId++;
  busyCount += 1;
  setStatus(state.ready ? label : statusText.textContent, "busy");
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, chunks: [] });
    worker.postMessage({ id, cmd, args, files });
  }).finally(() => {
    busyCount -= 1;
    if (busyCount === 0 && state.ready) setStatus("Ready.", "ready");
  });
}

function onReady(meta, pyodideVersion) {
  state.ready = true;
  state.init = meta;
  const v = meta.versions;
  $("#versions").innerHTML =
    `Python ${esc(v.python)} (Pyodide ${esc(pyodideVersion)}) with numpy ${esc(v.numpy)}, ` +
    `pandas ${esc(v.pandas)} and matplotlib ${esc(v.matplotlib)}, running in your browser; ` +
    `screen plots by Plotly ${esc(window.Plotly ? Plotly.version : "?")}.`;
  $("#rod-nsigma").value = meta.default_n_sigma;
  $("#spot-nsigma").value = meta.default_n_sigma;
  $("#spot-window").value = meta.default_window;
  if (busyCount === 0) {
    setStatus(state.files.dat || state.files.pro
      ? "Python ready." : "Python ready. Choose the .dat and .pro files.", "ready");
  }
}

function onFatal(message) {
  setStatus(`Python could not start: ${message}`, "error");
  showMsg(1, `Python could not start (${message}). The app needs to download Pyodide, numpy, ` +
    `pandas and matplotlib from cdn.jsdelivr.net the first time; check the connection and reload.`);
}

// ---------------------------------------------------------------------------
// steps, messages
// ---------------------------------------------------------------------------

function step(n) { return $(`#step-${n}`); }

function unlock(n, on = true) {
  step(n).classList.toggle("locked", !on);
  if (n === 2) $("#geometry").classList.toggle("locked", !on);
}
function markDone(n, on = true) {
  step(n).classList.toggle("done", on);
  if (on) step(n).classList.remove("skipped");
}
function markSkipped(n, on = true) {
  step(n).classList.toggle("skipped", on);
  if (on) step(n).classList.remove("done");
}
function lockFrom(n) {
  for (let i = n; i <= 14; i++) {
    unlock(i, false);
    markDone(i, false);
    markSkipped(i, false);
  }
}

function showMsg(n, text, kind = "error", traceback = null) {
  const el = $(`#msg-${n}`);
  el.className = `msg ${kind}`;
  el.textContent = text;
  if (traceback) {
    const details = document.createElement("details");
    details.innerHTML = "<summary>Python traceback</summary>";
    const pre = document.createElement("pre");
    pre.textContent = traceback;
    details.append(pre);
    el.append(details);
  }
  el.hidden = false;
}
function clearMsg(n) {
  const el = $(`#msg-${n}`);
  el.hidden = true;
  el.textContent = "";
}
function showError(n, err) {
  const message = capitalise(err.message || err);
  showMsg(n, message, "error", err.userError ? null : err.traceback);
  setStatus(`Error: ${message.split("\n")[0]}`, "error");
}

// "Geometry & equations": open the first time, then remember the choice.
const geometry = $("#geometry");
geometry.open = storeGet("gisaxs.geometry.open") !== "0";
geometry.addEventListener("toggle", () => storeSet("gisaxs.geometry.open", geometry.open ? "1" : "0"));

// ---------------------------------------------------------------------------
// 1. upload
// ---------------------------------------------------------------------------

const EXT = { dat: ".dat", pro: ".pro" };

function kindOf(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".dat")) return "dat";
  if (name.endsWith(".pro")) return "pro";
  return null;
}

function updatePickers() {
  for (const kind of ["dat", "pro"]) {
    const file = state.files[kind];
    $(`#name-${kind}`).textContent = file ? `${file.name} (${mb(file.size)})` : "no file chosen";
    $(`#name-${kind}`).closest(".picker").classList.toggle("has-file", !!file);
  }
}

function acceptFiles(list, expected = null) {
  clearMsg(1);
  const problems = [];
  let changed = false;
  for (const file of list) {
    const kind = kindOf(file);
    if (!kind || (expected && kind !== expected)) {
      const want = expected ? `a ${EXT[expected]}` : "a .dat or .pro";
      problems.push(`“${file.name}” is not ${want} file.`);
      continue;
    }
    state.files[kind] = file;
    changed = true;
  }
  updatePickers();
  if (problems.length) {
    showMsg(1, problems.join("\n") + "\nExpected a BornAgain 2D dataset (.dat) and a BornAgain 1.7 project file (.pro).");
  }
  if (!changed) return;
  if (state.files.dat && state.files.pro) {
    loadFiles();
  } else if (!problems.length) {
    const missing = state.files.dat ? ".pro instrument file" : ".dat data file";
    showMsg(1, `Now choose the ${missing}.`, "info");
  }
}

for (const kind of ["dat", "pro"]) {
  const input = $(`#file-${kind}`);
  input.addEventListener("change", () => {
    if (input.files.length) acceptFiles(input.files, kind);
    input.value = "";
  });
}

const dropzone = $("#dropzone");
window.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("over");
});
window.addEventListener("dragleave", (event) => {
  if (!event.relatedTarget) dropzone.classList.remove("over");
});
window.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("over");
  if (event.dataTransfer && event.dataTransfer.files.length) acceptFiles(event.dataTransfer.files);
});

async function loadFiles() {
  const token = ++state.loadToken;
  resetAnalysis();
  const { dat, pro } = state.files;
  try {
    const p = await py("load_pro", { path: `${DATA_DIR}/input.pro`, name: pro.name },
                       [{ path: `${DATA_DIR}/input.pro`, file: pro }], "Reading the .pro file…");
    if (token !== state.loadToken) return;
    state.pro = p.meta;
    renderPro();
    unlock(2);
    markDone(2);

    const d = await py("load_dat", { path: `${DATA_DIR}/input.dat`, name: dat.name },
                       [{ path: `${DATA_DIR}/input.dat`, file: dat }], `Reading ${dat.name}…`);
    if (token !== state.loadToken) return;
    state.dat = d.meta;
    state.version = 0;
    renderDat();
    markDone(1);
    unlock(3);
    markDone(3);
    unlock(4);
    unlock(5);
    $("#btn-convert").disabled = false;
    idle(`Read ${int(d.meta.pixels)} pixels in ${d.meta.seconds.toFixed(1)} s.`);
  } catch (err) {
    if (token === state.loadToken) showError(1, err);
  }
}

function resetAnalysis() {
  for (const card of cards.values()) card.destroy();
  cards.clear();
  $$("input[data-map], input[data-profile], #show-raw").forEach((box) => { box.checked = false; });
  $$('input[name="rod"], input[name="spot"]').forEach((r) => { r.checked = false; });
  $("#rod-options").hidden = true;
  $("#spot-options").hidden = true;
  $("#spot-skipped").hidden = true;
  $("#spot-question-wrap").hidden = false;
  for (const id of ["res-5", "res-6", "res-7", "res-8", "res-10", "res-11"]) $(`#${id}`).innerHTML = "";
  for (const n of [5, 9, 10, 11, 14]) clearMsg(n);
  Object.assign(state, { pro: null, dat: null, conv: null, version: 0, maskKind: "none",
                         rod: null, spot: null, plotsUnlocked: false });
  markDone(1, false);
  lockFrom(2);
  renderHub();
}

// ---------------------------------------------------------------------------
// 2. instrument table
// ---------------------------------------------------------------------------

function renderPro() {
  const s = state.pro.summary;
  const u = UNITS[state.qUnit];
  const pitch = Math.abs(s.pitch_x - s.pitch_y) < 1e-9
    ? num(s.pitch_x, 6) : `${num(s.pitch_x, 6)} (x), ${num(s.pitch_y, 6)} (y)`;
  const pixel = Math.abs(s.pixel_mdeg_x - s.pixel_mdeg_y) < 1e-9
    ? num(s.pixel_mdeg_x, 6) : `${num(s.pixel_mdeg_x, 6)} (x), ${num(s.pixel_mdeg_y, 6)} (y)`;
  const rows = [
    ["From the .pro file"],
    ["Detector bins", "n<sub>x</sub> × n<sub>y</sub>", `${s.nx} × ${s.ny}`, ""],
    ["Detector width × height", "(N<sub>bins</sub> − 1) × pitch", `${num(s.width, 8)} × ${num(s.height, 8)}`, "mm"],
    ["Detector normal (sample → detector)", "<b>N</b>", `(${s.N.map((v) => num(v, 12)).join(", ")})`, "mm"],
    ["Foot of the normal on the detector", "u<sub>0</sub>, v<sub>0</sub>", `${num(s.u0, 12)}, ${num(s.v0, 12)}`, "mm"],
    ["Wavelength", "λ", num(s.wavelength, 10), "nm"],
    ["Incidence angle", "α<sub>i</sub>", num(s.alpha_i, 15), "deg"],
    ["Derived"],
    ["Sample–detector distance", "L = |<b>N</b>|", num(s.L, 10), "mm"],
    ["Pixel pitch", "width / (n<sub>x</sub> − 1)", pitch, "mm"],
    ["Wavenumber", "k = 2π / λ", num(s.k * u.factor, 9), u.html],
    ["Angular size of one pixel", "pitch / L", pixel, "mdeg"],
    ["Sample horizon on the detector", "y = L tan α<sub>i</sub>", num(state.pro.horizon_y, 7), "mm"],
    ["Specular spot on the detector", "y = L tan 2α<sub>i</sub>", num(state.pro.specular_y, 7), "mm"],
  ];
  $("#pro-table").innerHTML =
    "<thead><tr><th>Quantity</th><th>Symbol</th><th>Value</th><th>Unit</th></tr></thead><tbody>" +
    rows.map((r) => r.length === 1
      ? `<tr class="group"><td colspan="4">${r[0]}</td></tr>`
      : `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("") +
    "</tbody>";

  const special = { beam: "x = 0, y = 0 mm",
                    horizon: `y = ${num(state.pro.horizon_y, 6)} mm`,
                    specular: `x = 0, y = ${num(state.pro.specular_y, 6)} mm` };
  $$("[data-special]").forEach((td) => { td.textContent = special[td.dataset.special]; });
}

// ---------------------------------------------------------------------------
// 3. raw data
// ---------------------------------------------------------------------------

function renderHead(table, head) {
  table.innerHTML =
    "<thead><tr><th></th>" + head.columns.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr></thead><tbody>" +
    head.rows.map((row, i) => `<tr><td>${head.index[i]}</td>` +
      row.map((v) => `<td>${esc(v.replace(/^-/, MINUS))}</td>`).join("") + "</tr>").join("") +
    "</tbody>";
}

function facts(list) {
  return list.map(([dt, dd, cls]) => `<div><dt>${dt}</dt><dd${cls ? ` class="${cls}"` : ""}>${dd}</dd></div>`).join("");
}

function renderDat() {
  const d = state.dat;
  const ordered = d.order.startsWith("row-major");
  const variance = { present: "present", "all zero": "present, all zero", absent: "absent" }[d.variance];
  $("#dat-facts").innerHTML = facts([
    ["Pixels", int(d.pixels)],
    ["Grid (from the header)", `${int(d.ny)} rows × ${int(d.nx)} columns`],
    ["Pixel order", ordered ? "✓ row-major <small>y ascending (bottom row first), x ascending in a row</small>"
                            : `re-sorted <small>${esc(d.order)}</small>`, ordered ? "ok" : ""],
    ["Dead pixels → NaN", `${int(d.dead)} <small>(${int(d.dead_minus_one)} written as −1)</small>`],
    ["Zero-count pixels", `${int(d.zeros)} <small>(${pct(d.zeros, d.pixels)})</small>`],
    ["Largest count", int(d.max)],
    ["Variance column", variance],
    ["x range", `${num(d.x_range[0], 6)} … ${num(d.x_range[1], 6)} mm`],
    ["y range", `${num(d.y_range[0], 6)} … ${num(d.y_range[1], 6)} mm`],
  ]);
  renderHead($("#dat-head"), d.head);
}

// ---------------------------------------------------------------------------
// 5-9. conversion
// ---------------------------------------------------------------------------

$("#btn-convert").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  clearMsg(5);
  const token = state.loadToken;
  try {
    const r = await py("convert", {}, [], "Converting…");
    if (token !== state.loadToken) return;
    state.conv = r.meta;
    state.version = r.meta.version;
    state.maskKind = "none";
    renderConversion();
    for (const n of [5, 6, 7, 8]) { unlock(n); markDone(n); }
    unlock(9);
    unlock(10);
    await refreshConvertedHead();
    markDone(9);
    const s8 = r.meta.steps["8"];
    idle(`Converted ${int(state.dat.pixels)} pixels; ${int(s8.pixels)} lie above the sample horizon.`);
  } catch (err) {
    button.disabled = false;
    showError(5, err);
  }
});

function rangeTable(rows) {
  return `<div class="table-wrap"><table class="plain"><thead><tr><th>column</th><th>quantity</th>` +
    `<th>minimum</th><th>maximum</th><th>unit</th></tr></thead><tbody>` +
    rows.map(([col, label, range, unit, factor = 1, digits = 7]) =>
      `<tr><td><code>${col}</code></td><td>${label}</td><td>${num(range[0] * factor, digits)}</td>` +
      `<td>${num(range[1] * factor, digits)}</td><td>${unit}</td></tr>`).join("") +
    "</tbody></table></div>";
}

function renderConversion() {
  const st = state.conv.steps;
  const u = UNITS[state.qUnit];
  $("#res-5").innerHTML = `<p class="note">Added 2 columns for all ${int(state.dat.pixels)} pixels.</p>` +
    rangeTable([["phi_f", "φ<sub>f</sub>", st["5"].phi_f, "deg"],
                ["alpha_f", "α<sub>f</sub>", st["5"].alpha_f, "deg"]]);
  $("#res-6").innerHTML = `<p class="note">Added 3 columns. Check: |<b>k</b><sub>f</sub>| = k = ` +
    `${num(st["6"].k * u.factor, 9)} ${u.html} for every pixel (largest relative deviation ` +
    `${st["6"].max_rel_dev.toExponential(1).replace("-", MINUS)}).</p>` +
    rangeTable(["kfx", "kfy", "kfz"].map((c, i) =>
      [c, `k<sub>f${"xyz"[i]}</sub>`, st["6"][c], u.html, u.factor]));
  $("#res-7").innerHTML = `<p class="note">Added 3 columns.</p>` +
    rangeTable(["qx", "qy", "qz"].map((c, i) =>
      [c, `q<sub>${"xyz"[i]}</sub>`, st["7"][c], u.html, u.factor]));
  const s8 = st["8"];
  $("#res-8").innerHTML = `<dl class="facts">` + facts([
    ["Sample horizon", `y = L tan α<sub>i</sub> = ${num(s8.horizon_y, 6)} mm`],
    ["Rows dropped", `${int(s8.rows_dropped)} <small>of ${int(s8.rows_before)}</small>`],
    ["Rows kept", int(s8.rows_kept)],
    ["Pixels kept", `${int(s8.rows_kept)} × ${int(s8.nx)} = ${int(s8.pixels)}`, "ok"],
    ["Lowest kept row", `y = ${num(s8.first_row_y, 6)} mm <small>α<sub>f,row</sub> = ${signed(s8.first_row_alpha, 5)}°</small>`],
    ["Highest dropped row", s8.last_dropped_y == null ? "none"
      : `y = ${num(s8.last_dropped_y, 6)} mm <small>α<sub>f,row</sub> = ${signed(s8.last_dropped_alpha, 5)}°</small>`],
  ]) + "</dl>";
}

async function refreshConvertedHead() {
  const r = await py("table_head", { which: "converted", q_unit: state.qUnit }, [], "Preparing the table…");
  renderHead($("#conv-head"), r.meta.head);
  $("#conv-summary").textContent =
    `${int(r.meta.pixels)} pixels (${int(r.meta.ny)} rows × ${int(r.meta.nx)} columns), 12 columns. First rows:`;
}

// ---------------------------------------------------------------------------
// downloads
// ---------------------------------------------------------------------------

function stem() {
  const name = state.files.dat ? state.files.dat.name : "gisaxs";
  return name.replace(/\.[^.]*$/, "");
}

function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-download-table]");
  if (!button) return;
  const which = button.dataset.downloadTable;
  const fmt = button.dataset.fmt;
  const msgStep = button.dataset.msg || "9";
  const unitTag = state.qUnit === "A-1" ? "_A-1" : "";
  const name = which === "converted"
    ? `${stem()}_converted${unitTag}.${fmt}`
    : `${stem()}_masked-${state.maskKind}${unitTag}.${fmt}`;
  button.disabled = true;
  clearMsg(msgStep);
  try {
    const r = await py("export_table", { which, fmt, q_unit: state.qUnit }, [], "Writing the table…");
    const blob = new Blob(r.chunks, { type: r.meta.mime });
    saveBlob(blob, name);
    idle(`Saved ${name} (${mb(blob.size)}).`);
  } catch (err) {
    showError(msgStep, err);
  } finally {
    button.disabled = false;
  }
});

function maskedDownloads(stepNo) {
  return `<div class="actions"><span>Masked table:</span>` +
    `<button type="button" data-download-table="masked" data-fmt="csv.gz" data-msg="${stepNo}">Download .csv.gz</button>` +
    `<button type="button" data-download-table="masked" data-fmt="csv" data-msg="${stepNo}">Download plain .csv</button>` +
    `<span class="muted">Same columns as step 9; masked pixels are <code>nan</code> and the header records the mask.</span></div>`;
}

// ---------------------------------------------------------------------------
// 10-11. masks
// ---------------------------------------------------------------------------

function readNumber(input, { min = null, optional = false, name }) {
  const text = input.value.trim();
  if (text === "") {
    if (optional) return null;
    throw new Error(`${name} is empty.`);
  }
  const value = Number(text);
  if (!Number.isFinite(value)) throw new Error(`${name} must be a number.`);
  if (min !== null && Number.isFinite(min) && value <= min) {
    throw new Error(`${name} must be greater than ${min}.`);
  }
  return value;
}

async function applyMask(kind, stepNo) {
  clearMsg(stepNo);
  let args = { kind };
  try {
    if (kind === "rod") {
      args.n_sigma = readNumber($("#rod-nsigma"), { min: 0, name: "nσ" });
      args.half_width = readNumber($("#rod-halfwidth"), { min: 0, optional: true, name: "The fixed half-width" });
    } else if (kind === "spot") {
      args.n_sigma = readNumber($("#spot-nsigma"), { min: 0, name: "nσ" });
      args.window = readNumber($("#spot-window"), { min: 2, name: "The window W" });
      args.follow_peak = $("#spot-follow").checked;
    }
  } catch (err) {
    showMsg(stepNo, err.message);
    return false;
  }
  const labels = { rod: "Masking the specular rod…", spot: "Masking the specular spot…", none: "Removing the mask…" };
  try {
    const r = await py("apply_mask", args, [], labels[kind]);
    state.version = r.meta.version;
    state.maskKind = kind;
    if (kind === "rod") renderRodReport(r.meta);
    if (kind === "spot") renderSpotReport(r.meta);
    if (kind !== "rod") $("#res-10").innerHTML = "";
    if (kind !== "spot") $("#res-11").innerHTML = "";
    refreshCards();
    idle(kind === "none" ? "No mask applied." : `Masked ${int(r.meta.n_masked)} pixels.`);
    return true;
  } catch (err) {
    showError(stepNo, err);
    if (kind === "rod") $("#res-10").innerHTML = "";
    if (kind === "spot") $("#res-11").innerHTML = "";
    return false;
  }
}

function reportTable(rows) {
  return `<div class="table-wrap"><table class="plain"><tbody>` +
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") + "</tbody></table></div>";
}

function renderRodReport(m) {
  const px = m.pixel;
  $("#res-10").innerHTML = reportTable([
    ["Rod centre φ<sub>f</sub>", `${signed(m.centre, 5)}° <span class="muted">(${signed(m.centre / px, 1)} pixels from φ<sub>f</sub> = 0; tallest column at ${signed(m.tallest, 5)}°)</span>`],
    ["FWHM", `${fixed(m.fwhm, 5)}° <span class="muted">(${fixed(m.fwhm / px, 1)} pixels of ${fixed(px * 1000, 3)} mdeg)</span>`],
    ["σ = FWHM / 2.3548", `${fixed(m.sigma, 5)}°`],
    ["Masked band", `${signed(m.band[0], 5)}° … ${signed(m.band[1], 5)}° <span class="muted">(±${fixed(m.half_width, 5)}° = ${m.fixed ? "fixed half-width" : `${num(m.n_sigma, 4)} σ`})</span>`],
    ["Pixels masked", `${int(m.n_masked)} <span class="muted">(${pct(m.n_masked, m.pixels)} of the frame; intensity and variance set to NaN)</span>`],
    ["Tallest column, summed counts", `${int(m.peak_height)} before → ${int(m.left_over)} after <span class="muted">(above the background of ${int(m.background)} per column)</span>`],
    ["Diffuse beside the mask", m.diffuse == null ? "–" : `${int(m.diffuse)} per column <span class="muted">(the leftover should be near this)</span>`],
  ]) + maskedDownloads(10);
}

function renderSpotReport(m) {
  const moved = m.follow_steps
    ? ` <span class="muted">(moved ${m.follow_steps} time${m.follow_steps > 1 ? "s" : ""} to follow the peak)</span>` : "";
  const hp = m.phi_profile, va = m.alpha_profile;
  const onePixel = (p) => p.fwhm_was_below_pixel ? ' <span class="muted">(narrower than a pixel: one pixel used)</span>' : "";
  $("#res-11").innerHTML = reportTable([
    ["Predicted position", `φ<sub>f</sub> = 0, α<sub>f</sub> = α<sub>i</sub> = ${fixed(m.predicted[1], 5)}°`],
    ["Window", `±${num(m.window_pixels, 4)} pixels = ±${fixed(m.half_width_phi, 5)}° (φ<sub>f</sub>) × ±${fixed(m.half_width_alpha, 5)}° (α<sub>f</sub>), centred on (${signed(m.window_centre[0], 5)}°, ${fixed(m.window_centre[1], 5)}°)${moved}`],
    ["Spot centre φ<sub>0</sub>, α<sub>0</sub>", `${signed(m.phi0, 5)}°, ${fixed(m.alpha0, 5)}° <span class="muted">(α<sub>0</sub> − α<sub>i</sub> = ${signed(m.alpha0 - m.predicted[1], 5)}° = ${signed((m.alpha0 - m.predicted[1]) / m.pixel_alpha, 1)} pixels)</span>`],
    ["FWHM along φ<sub>f</sub>, α<sub>f</sub>", `${fixed(m.fwhm_phi, 5)}°${onePixel(hp)}, ${fixed(m.fwhm_alpha, 5)}°${onePixel(va)}`],
    ["σ<sub>φ</sub>, σ<sub>α</sub>", `${fixed(m.sigma_phi, 5)}°, ${fixed(m.sigma_alpha, 5)}°`],
    ["Background, height (mean counts)", `φ<sub>f</sub> profile: ${num(hp.background, 4)}, ${num(hp.height, 5)}; α<sub>f</sub> profile: ${num(va.background, 4)}, ${num(va.height, 5)}`],
    ["Ellipse half-axes", `${fixed(m.half_axis_phi, 5)}° × ${fixed(m.half_axis_alpha, 5)}° <span class="muted">(${num(m.n_sigma, 4)} σ)</span>`],
    ["Pixels masked", `${int(m.n_masked)} <span class="muted">(${pct(m.n_masked, m.pixels)} of the frame; intensity and variance set to NaN)</span>`],
  ]) + maskedDownloads(11);
}

function unlockPlots() {
  if (state.plotsUnlocked) return;
  state.plotsUnlocked = true;
  for (const n of [12, 13, 14]) unlock(n);
}

async function onRodAnswer(value) {
  state.rod = value;
  clearMsg(10);
  $("#rod-options").hidden = value !== "yes";
  if (value === "yes") {
    const ok = await applyMask("rod", 10);
    if (!ok) { markDone(10, false); return; }
    markDone(10);
    unlock(11);
    markSkipped(11);
    $("#spot-skipped").hidden = false;
    $("#spot-question-wrap").hidden = true;
    $("#res-11").innerHTML = "";
    clearMsg(11);
    unlockPlots();
  } else {
    markDone(10);
    unlock(11);
    markSkipped(11, false);
    $("#spot-skipped").hidden = true;
    $("#spot-question-wrap").hidden = false;
    if (state.spot === "yes") {
      await onSpotAnswer("yes");
    } else {
      if (state.maskKind !== "none") await applyMask("none", 10);
      if (state.spot === "no") { markDone(11); unlockPlots(); }
    }
  }
}

async function onSpotAnswer(value) {
  state.spot = value;
  clearMsg(11);
  $("#spot-options").hidden = value !== "yes";
  if (value === "yes") {
    const ok = await applyMask("spot", 11);
    markDone(11, ok);
    if (ok) unlockPlots();
  } else {
    if (state.maskKind !== "none") await applyMask("none", 11);
    markDone(11);
    unlockPlots();
  }
}

$$('input[name="rod"]').forEach((r) => r.addEventListener("change", () => onRodAnswer(r.value)));
$$('input[name="spot"]').forEach((r) => r.addEventListener("change", () => onSpotAnswer(r.value)));
$("#rod-apply").addEventListener("click", () => onRodAnswer("yes"));
$("#spot-apply").addEventListener("click", () => onSpotAnswer("yes"));
for (const [id, handler] of [["rod-nsigma", () => onRodAnswer("yes")], ["rod-halfwidth", () => onRodAnswer("yes")],
                             ["spot-nsigma", () => onSpotAnswer("yes")], ["spot-window", () => onSpotAnswer("yes")],
                             ["spot-follow", () => onSpotAnswer("yes")]]) {
  $(`#${id}`).addEventListener("keydown", (e) => { if (e.key === "Enter") handler(); });
  if (id === "spot-follow") $(`#${id}`).addEventListener("change", handler);
}

// ---------------------------------------------------------------------------
// plots
// ---------------------------------------------------------------------------

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  showSendToCloud: false,                 // never offer to send data anywhere
  modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage", "autoScale2d"],
  doubleClick: "reset",
};
const FONT = { family: "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif", size: 12, color: "#1b2330" };
const INK = "#1b2330";
const ACCENT = "#1d4f91";

function axisDef(name) {
  const u = UNITS[state.qUnit];
  const defs = {
    x: { title: "x (mm)", label: "x", fmt: ".3f", unit: " mm" },
    y: { title: "y (mm)", label: "y", fmt: ".3f", unit: " mm" },
    phi_f: { title: "φ<sub>f</sub> (deg)", label: "φ<sub>f</sub>", fmt: ".4f", unit: "°" },
    alpha_f: { title: "α<sub>f</sub> (deg)", label: "α<sub>f</sub>", fmt: ".4f", unit: "°" },
    qy: { title: `q<sub>y</sub> (${u.html})`, label: "q<sub>y</sub>", fmt: `.${u.digits}f`, unit: ` ${u.html}` },
    qz: { title: `q<sub>z</sub> (${u.html})`, label: "q<sub>z</sub>", fmt: `.${u.digits}f`, unit: ` ${u.html}` },
  };
  return defs[name];
}

function axisStyle(title, extra = {}) {
  return Object.assign({
    title: { text: title, standoff: 8 },
    showline: true, linecolor: INK, linewidth: 1, mirror: "ticks",
    ticks: "inside", ticklen: 5, tickcolor: INK,
    zeroline: false, showgrid: false,
    exponentformat: "power", automargin: true,
  }, extra);
}

const MAPS = {
  raw: { title: "Raw intensity map: y vs x (full detector)", x: "x", y: "y", slug: "raw_y-vs-x", container: "#cards-raw" },
  qz_qy: { title: "q<sub>z</sub> vs q<sub>y</sub>", x: "qy", y: "qz", slug: "qz-vs-qy", container: "#cards-maps" },
  af_pf: { title: "α<sub>f</sub> vs φ<sub>f</sub>", x: "phi_f", y: "alpha_f", slug: "alphaf-vs-phif", container: "#cards-maps" },
  y_x: { title: "y vs x (above the horizon)", x: "x", y: "y", slug: "y-vs-x", container: "#cards-maps" },
};
const PROFILES = {
  phi_f: { title: "Mean counts per pixel vs φ<sub>f</sub>", axis: "phi_f", slug: "profile-vs-phif", along: "detector columns" },
  qy: { title: "Mean counts per pixel vs q<sub>y</sub>", axis: "qy", slug: "profile-vs-qy", along: "detector columns" },
  alpha_f: { title: "Mean counts per pixel vs α<sub>f</sub>", axis: "alpha_f", slug: "profile-vs-alphaf", along: "detector rows" },
  qz: { title: "Mean counts per pixel vs q<sub>z</sub>", axis: "qz", slug: "profile-vs-qz", along: "detector rows" },
};

const cards = new Map();

function maskLabel() {
  return { none: "no mask", rod: "specular rod masked", spot: "specular spot masked" }[state.maskKind];
}

class Card {
  constructor(key, template, container) {
    this.key = key;
    this.el = $(template).content.firstElementChild.cloneNode(true);
    this.plot = $(".plot", this.el);
    this.note = $(".card-note", this.el);
    this.dataKey = null;
    this.data = null;
    this.seq = 0;
    this.drawn = false;
    $$(".dl button", this.el).forEach((b) => b.addEventListener("click", () => exportCard(this, b.dataset.fmt, b)));
    $(container).append(this.el);
  }

  async refresh() {
    const seq = ++this.seq;
    const key = this.currentDataKey();
    try {
      if (key !== this.dataKey) {
        this.plot.classList.add("loading");
        const r = await this.fetch();
        if (seq !== this.seq) return;
        this.data = r;
        this.dataKey = key;
      }
      this.plot.classList.remove("loading");
      this.draw();
    } catch (err) {
      if (seq !== this.seq) return;
      this.plot.classList.remove("loading");
      this.note.textContent = `Could not draw: ${err.message}`;
      setStatus(`Error: ${err.message}`, "error");
    }
  }

  fileName(fmt) {
    return `${stem()}_${this.slug}.${fmt}`;
  }

  destroy() {
    this.seq += 1;
    if (window.Plotly) Plotly.purge(this.plot);
    this.el.remove();
  }
}

class MapCard extends Card {
  constructor(kind) {
    const def = MAPS[kind];
    super(kind, "#tpl-map", def.container);
    this.kind = kind;
    this.def = def;
    this.slug = def.slug;
    this.isProfile = false;
    this.usesQ = def.x === "qy";
    this.label = def.title;
    $(".card-title", this.el).innerHTML = def.title;
    const cmap = $(".opt-cmap", this.el);
    cmap.innerHTML = state.init.colormaps.map((c) => `<option value="${c}">${c}</option>`).join("");
    for (const input of $$(".card-options input, .card-options select", this.el)) {
      input.addEventListener("change", () => { if (this.data) this.draw(); });
    }
    this.sizeListener = null;
  }

  currentDataKey() {
    return this.kind === "raw" ? `raw|${state.loadToken}`
      : `${state.loadToken}|${state.version}|${this.usesQ ? state.qUnit : ""}`;
  }

  fetch() {
    return py("map_data", { kind: this.kind, q_unit: state.qUnit }, [], "Preparing the map…");
  }

  options() {
    const log = $(".opt-log", this.el).checked;
    const vmin = readNumber($(".opt-vmin", this.el), { min: log ? 0 : -Infinity, optional: true, name: "vmin" });
    const vmax = readNumber($(".opt-vmax", this.el), { min: -Infinity, optional: true, name: "vmax" });
    const size = readNumber($(".opt-size", this.el), { min: 0, optional: true, name: "Point size" });
    const lo = vmin == null ? 1 : vmin;
    const hi = vmax == null ? (this.data && this.data.meta.vmax) || 10 : vmax;
    if (hi <= lo) throw new Error("vmax must be larger than vmin.");
    return { log, cmap: $(".opt-cmap", this.el).value, zero: $(".opt-zero", this.el).value,
             vmin, vmax, lo, hi, size };
  }

  exportArgs() {
    const o = this.options();
    return { log: o.log, cmap: o.cmap, vmin: o.vmin, vmax: o.vmax, zero_colour: o.zero };
  }

  autoSize() {
    const fl = this.plot._fullLayout;
    const m = this.data.meta;
    if (!fl || !fl.xaxis || !fl.xaxis._length) {
      // Before the first layout: estimate from the card width.
      const width = Math.max(200, this.plot.clientWidth - 140);
      const span = (m.x_range[1] - m.x_range[0]) || 1;
      return Math.min(40, Math.max(1, 1.15 * m.dx * width / span));
    }
    const xa = fl.xaxis, ya = fl.yaxis;
    const px = Math.abs(xa._length / (xa.range[1] - xa.range[0]));
    const py_ = Math.abs(ya._length / (ya.range[1] - ya.range[0]));
    return Math.min(40, Math.max(1, 1.15 * Math.max(m.dx * px, m.dy * py_)));
  }

  draw() {
    let o;
    try {
      o = this.options();
    } catch (err) {
      this.note.textContent = err.message;
      return;
    }
    const { meta, arrays } = this.data;
    const { x, y, c } = arrays;
    const colours = state.init.colours;
    const n = c.length;

    // Split the pixels: below vmin (incl. the measured zeros), NaN, and the rest.
    let nUnder = 0, nBad = 0;
    for (let i = 0; i < n; i++) {
      const v = c[i];
      if (v !== v) nBad++;
      else if (v < o.lo) nUnder++;
    }
    const nMain = n - nUnder - nBad;
    const ux = new Float32Array(nUnder), uy = new Float32Array(nUnder), uc = new Float32Array(nUnder);
    const bx = new Float32Array(nBad), by = new Float32Array(nBad);
    const idx = new Uint32Array(nMain);
    for (let i = 0, iu = 0, ib = 0, im = 0; i < n; i++) {
      const v = c[i];
      if (v !== v) { bx[ib] = x[i]; by[ib] = y[i]; ib++; }
      else if (v < o.lo) { ux[iu] = x[i]; uy[iu] = y[i]; uc[iu] = v; iu++; }
      else idx[im++] = i;
    }
    // Brightest last, so a bright pixel is never hidden under a dim neighbour.
    idx.sort((a, b) => c[a] - c[b]);
    const mx = new Float32Array(nMain), my = new Float32Array(nMain);
    const mc = new Float32Array(nMain), md = new Float32Array(nMain);
    for (let k = 0; k < nMain; k++) {
      const i = idx[k];
      mx[k] = x[i]; my[k] = y[i]; md[k] = c[i];
      mc[k] = o.log ? Math.log10(c[i]) : c[i];      // colour only; data stay linear
    }

    const ax = axisDef(this.def.x), ay = axisDef(this.def.y);
    const where = `${ax.label} = %{x:${ax.fmt}}${ax.unit}<br>${ay.label} = %{y:${ay.fmt}}${ay.unit}`;
    const size = o.size || this.autoSize();
    const marker = (extra) => Object.assign({ size, symbol: "square", line: { width: 0 } }, extra);

    let cmin = o.lo, cmax = o.hi, tickvals, ticktext;
    if (o.log) {
      cmin = Math.log10(o.lo);
      cmax = Math.log10(o.hi);
      tickvals = [];
      ticktext = [];
      for (let k = Math.ceil(cmin - 1e-9); k <= Math.floor(cmax + 1e-9); k++) {
        tickvals.push(k);
        ticktext.push(k === 0 ? "1" : k === 1 ? "10" : `10<sup>${String(k).replace("-", MINUS)}</sup>`);
      }
      if (tickvals.length < 2) {
        tickvals = [cmin, cmax];
        ticktext = [num(o.lo, 3), num(o.hi, 3)];
      }
    }
    const underName = o.lo === 1 ? "0 counts (below the scale)" : `< ${num(o.lo, 4)} counts (below the scale)`;

    const traces = [];
    if (nUnder) {
      traces.push({
        type: "scattergl", mode: "markers", name: underName, x: ux, y: uy, customdata: uc,
        showlegend: false, legendgroup: "under",
        marker: marker({ color: colours.under[o.zero] }),
        hovertemplate: `${where}<br>I = %{customdata:.6~r} cts<extra></extra>`,
      });
      // Its legend entry: an outlined square, so the default white swatch
      // stays visible on the white background.  Clicking it toggles the group.
      traces.push({
        type: "scatter", mode: "markers", name: underName, x: [null], y: [null],
        legendgroup: "under", legendrank: 1, hoverinfo: "skip",
        marker: { color: colours.under[o.zero], size: 10, symbol: "square",
                  line: { color: "#8a94a3", width: 1 } },
      });
    }
    if (nBad) {
      traces.push({
        type: "scattergl", mode: "markers", name: "masked or dead (NaN)", x: bx, y: by,
        legendrank: 2,
        marker: marker({ color: colours.bad }),
        hovertemplate: `${where}<br>masked or dead pixel<extra></extra>`,
      });
    }
    traces.push({
      type: "scattergl", mode: "markers", name: "counts", showlegend: false,
      x: mx, y: my, customdata: md,
      marker: marker({
        color: mc, colorscale: colours.scales[o.cmap], cmin, cmax, showscale: true,
        colorbar: {
          title: { text: "Counts", side: "right" }, thickness: 14, len: 0.9, outlinewidth: 0,
          ticks: "outside", ticklen: 4, tickvals, ticktext,
        },
      }),
      hovertemplate: `${where}<br>I = %{customdata:.6~r} cts<extra></extra>`,
    });

    const pad = (r, d) => [r[0] - d / 2, r[1] + d / 2];
    const layout = {
      font: FONT, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
      margin: { l: 10, r: 10, t: 36, b: 10 },
      xaxis: axisStyle(ax.title, { range: pad(meta.x_range, meta.dx), constrain: "domain" }),
      yaxis: axisStyle(ay.title, { range: pad(meta.y_range, meta.dy), constrain: "domain",
                                   scaleanchor: "x", scaleratio: 1 }),
      legend: { orientation: "h", x: 0, xanchor: "left", y: 1.0, yanchor: "bottom",
                itemsizing: "constant", font: { size: 11 } },
      hovermode: "closest",
      hoverlabel: { font: { family: FONT.family, size: 12 } },
      uirevision: `${this.kind}|${this.usesQ ? state.qUnit : ""}|${state.loadToken}`,
    };

    Plotly.react(this.plot, traces, layout, PLOT_CONFIG).then(() => {
      if (!this.sizeListener) {
        this.sizeListener = () => this.resize();
        this.plot.on("plotly_relayout", this.sizeListener);
      }
      this.resize();
    });

    $(".card-meta", this.el).textContent = this.kind === "raw"
      ? `${int(n)} pixels, before conversion` : `${int(n)} pixels, ${maskLabel()}`;
    this.note.textContent =
      `${o.log ? "log₁₀ colour scale" : "linear colour scale"}, ${num(o.lo, 4)} – ${int(o.hi)} counts · ` +
      `${int(nUnder)} below the scale (${pct(nUnder, n)}) · ${int(nBad)} NaN · ` +
      `point size ${o.size ? num(o.size, 3) : "auto"}`;
  }

  resize() {
    // Auto point size: one detector pixel on screen, recomputed after a zoom.
    let o;
    try { o = this.options(); } catch (_) { return; }
    if (o.size || !this.data) return;
    const size = this.autoSize();
    // Only the WebGL pixel layers; the legend swatch keeps its own size.
    const layers = (this.plot.data || []).map((t, i) => (t.type === "scattergl" ? i : -1)).filter((i) => i >= 0);
    if (!layers.length) return;
    const current = this.plot.data[layers[0]].marker.size;
    if (current && Math.abs(current - size) < 0.25) return;
    Plotly.restyle(this.plot, { "marker.size": size }, layers);
  }
}

class ProfileCard extends Card {
  constructor(kind) {
    const def = PROFILES[kind];
    super(`p:${kind}`, "#tpl-profile", "#cards-profiles");
    this.kind = kind;
    this.def = def;
    this.slug = def.slug;
    this.isProfile = true;
    this.usesQ = kind === "qy" || kind === "qz";
    this.label = def.title;
    $(".card-title", this.el).innerHTML = def.title;
    $(".opt-log", this.el).addEventListener("change", () => { if (this.data) this.draw(); });
    const smooth = $(".opt-smooth", this.el);
    smooth.value = String(state.init.default_smoothing);
    // Smoothing is done in Python (gisaxs.smooth_profile), the same code the
    // exported figure uses, so a change re-fetches this one small profile.
    smooth.addEventListener("change", () => this.refresh());
  }

  smoothing() {
    return Number($(".opt-smooth", this.el).value);
  }

  currentDataKey() {
    return `${state.loadToken}|${state.version}|${this.usesQ ? state.qUnit : ""}|${this.smoothing()}`;
  }

  fetch() {
    return py("profile_data", { kind: this.kind, q_unit: state.qUnit, smooth: this.smoothing() },
              [], "Computing the profile…");
  }

  exportArgs() {
    return { log: $(".opt-log", this.el).checked, smooth: this.smoothing() };
  }

  draw() {
    const { meta, arrays } = this.data;
    const log = $(".opt-log", this.el).checked;
    const ax = axisDef(this.def.axis);
    const spikes = { showspikes: true, spikemode: "across", spikesnap: "data", spikethickness: 1,
                     spikedash: "dot", spikecolor: "#6b7585" };
    const trace = {
      type: "scatter", mode: "lines", x: arrays.x, y: arrays.y, connectgaps: false,
      line: { color: ACCENT, width: 1.5 }, name: "",
      hovertemplate: `${ax.label} = %{x:${ax.fmt}}${ax.unit}, I = %{y:.4~r} cts<extra></extra>`,
    };
    const layout = {
      font: FONT, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
      margin: { l: 10, r: 16, t: 16, b: 10 },
      xaxis: axisStyle(ax.title, spikes),
      yaxis: axisStyle("Mean counts per pixel", Object.assign({ type: log ? "log" : "linear" }, spikes)),
      hovermode: "closest", hoverdistance: -1, spikedistance: -1, showlegend: false,
      hoverlabel: { font: { family: FONT.family, size: 12 } },
      uirevision: `${this.kind}|${this.usesQ ? state.qUnit : ""}|${state.loadToken}|${log}`,
    };
    Plotly.react(this.plot, [trace], layout, PLOT_CONFIG);
    $(".card-meta", this.el).textContent = `${int(meta.n)} ${this.def.along}, ${maskLabel()}`;
    const smoothing = meta.smooth > 1 ? `smoothed: centred running mean of ${meta.smooth} points` : "not smoothed";
    this.note.textContent = `${smoothing} · ` + (meta.n_gaps
      ? `${int(meta.n_gaps)} fully masked ${this.def.along.replace("detector ", "")} drawn as gaps`
      : "no fully masked columns or rows");
  }
}

function toggleCard(key, make, on) {
  if (on && !cards.has(key)) {
    const card = make();
    cards.set(key, card);
    card.refresh();
  } else if (!on && cards.has(key)) {
    cards.get(key).destroy();
    cards.delete(key);
  }
  renderHub();
}

$("#show-raw").addEventListener("change", (e) => toggleCard("raw", () => new MapCard("raw"), e.target.checked));
$$("input[data-map]").forEach((box) => box.addEventListener("change", () =>
  toggleCard(box.dataset.map, () => new MapCard(box.dataset.map), box.checked)));
$$("input[data-profile]").forEach((box) => box.addEventListener("change", () =>
  toggleCard(`p:${box.dataset.profile}`, () => new ProfileCard(box.dataset.profile), box.checked)));

async function refreshCards() {
  for (const card of Array.from(cards.values())) {
    if (card.kind !== "raw") await card.refresh();
  }
  renderHub();
}

// ---------------------------------------------------------------------------
// 14. figure export (matplotlib in Python)
// ---------------------------------------------------------------------------

function exportSettings() {
  return { preset: $("#export-preset").value, columns: Number($("#export-columns").value) };
}

async function exportCard(card, fmt, button) {
  let extra;
  try {
    extra = card.exportArgs();
  } catch (err) {
    showMsg(14, `${card.label.replace(/<[^>]+>/g, "")}: ${err.message}`);
    return;
  }
  const name = card.fileName(fmt);
  const buttons = $$(`[data-card="${card.key}"][data-fmt="${fmt}"]`).concat([button]);
  buttons.forEach((b) => { b.disabled = true; });
  clearMsg(14);
  try {
    const r = await py("export_figure", Object.assign({
      kind: card.kind, fmt, profile: card.isProfile, q_unit: state.qUnit,
    }, exportSettings(), extra), [], `Drawing ${name}…`);
    saveBlob(new Blob([r.arrays.file], { type: r.meta.mime }), name);
    idle(`Saved ${name} (${mb(r.meta.bytes)}, drawn in ${r.meta.seconds.toFixed(1)} s).`);
  } catch (err) {
    showError(14, err);
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
  }
}

function renderHub() {
  const hub = $("#download-hub");
  const list = Array.from(cards.values());
  if (!list.length) {
    hub.innerHTML = '<li class="muted">Tick a plot in steps 4, 12 or 13 to list it here.</li>';
    return;
  }
  hub.innerHTML = list.map((card) =>
    `<li><span><span class="hub-name">${card.label}</span> <code>${esc(card.fileName("pdf").replace(/\.pdf$/, ".png / .pdf"))}</code></span>` +
    `<span class="dl"><button type="button" data-card="${card.key}" data-fmt="png">PNG</button>` +
    `<button type="button" data-card="${card.key}" data-fmt="pdf">PDF</button></span></li>`).join("");
}

$("#download-hub").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-card]");
  if (!button) return;
  const card = cards.get(button.dataset.card);
  if (card) exportCard(card, button.dataset.fmt, button);
});

function syncExportControls() {
  $("#export-columns-tool").hidden = $("#export-preset").value === "presentation";
}
$("#export-preset").addEventListener("change", syncExportControls);
syncExportControls();

// ---------------------------------------------------------------------------
// q unit
// ---------------------------------------------------------------------------

$$(".segmented button[data-unit]").forEach((button) => button.addEventListener("click", () => {
  const unit = button.dataset.unit;
  if (unit === state.qUnit) return;
  state.qUnit = unit;
  $$(".segmented button[data-unit]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.unit === unit)));
  if (state.pro) renderPro();
  if (state.conv) {
    renderConversion();
    refreshConvertedHead().catch((err) => showError(9, err));
  }
  for (const card of cards.values()) if (card.usesQ) card.refresh();
  renderHub();
}));

// ---------------------------------------------------------------------------
// start
// ---------------------------------------------------------------------------

if (window.renderMathInElement) {
  renderMathInElement(document.body, {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }],
    throwOnError: false,
  });
}
if (!window.Plotly) {
  setStatus("Plotly could not be loaded from cdn.jsdelivr.net; plots will not work. Check the connection and reload.", "error");
}

})();
