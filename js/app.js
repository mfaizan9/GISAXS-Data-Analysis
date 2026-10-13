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

// ---- MathJax: symbols and numbers on the page are TeX, written \( ... \) ----

let mathQueue = Promise.resolve();

/** Typeset the maths in these elements (one MathJax call at a time). */
function typeset(...nodes) {
  const els = nodes.filter(Boolean);
  mathQueue = mathQueue.then(async () => {
    const mj = window.MathJax;
    if (!mj || !mj.startup || !mj.startup.promise || !mj.typesetPromise) return;
    await mj.startup.promise;
    await mj.typesetPromise(els);
  }).catch((err) => console.warn("MathJax:", err));
  return mathQueue;
}

/** Replace an element's HTML and typeset the maths in it. */
function setMath(el, html) {
  const mj = window.MathJax;
  if (mj && mj.typesetClear) mj.typesetClear([el]);
  el.innerHTML = html;
  typeset(el);
}

/** Inline TeX. */
const tex = (s) => `\\(${s}\\)`;

/** A formatted number (from num, fixed, int, ...) as TeX: 1{,}091{,}736, -38.23.
    Digits are grouped only from five on (1028, but 12{,}345), as in print. */
function texNum(text) {
  return String(text).replace(new RegExp(MINUS, "g"), "-")
    .replace(/^(-?)(\d),(\d{3})(?=$|\.)/, "$1$2$3")
    .replace(/,/g, "{,}");
}

const UNITS = {
  "nm-1": { factor: 1, html: "nm<sup>−1</sup>", tex: String.raw`\mathrm{nm}^{-1}`, suffix: "nm-1", digits: 4 },
  "A-1": { factor: 0.1, html: "Å<sup>−1</sup>", tex: String.raw`\text{Å}^{-1}`, suffix: "A-1", digits: 5 },
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
  alphaCheck: null,      // check_alpha result: where the specular spot sits
  alphaFix: null,        // "yes" | "no": correct alpha_i from the spot
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
  for (let i = n; i <= 13; i++) {
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
    idle(`Read ${int(d.meta.pixels)} pixels in ${d.meta.seconds.toFixed(1)} s.`);
    await checkAlpha(token);
  } catch (err) {
    if (token === state.loadToken) showError(1, err);
  }
}

function resetAnalysis() {
  for (const card of cards.values()) card.destroy();
  cards.clear();
  $("#show-raw").checked = false;
  resetConversion();
  Object.assign(state, { pro: null, dat: null, version: 0, alphaCheck: null, alphaFix: null });
  $("#alpha-check").hidden = true;
  $("#alpha-check").innerHTML = "";
  markDone(1, false);
  lockFrom(2);
}

/** Forget steps 5 on (the conversion, masks and their plots); the raw data stay. */
function resetConversion() {
  for (const [key, card] of cards) {
    if (key !== "raw") { card.destroy(); cards.delete(key); }
  }
  $$("input[data-map], input[data-profile]").forEach((box) => { box.checked = false; });
  $$('input[name="rod"], input[name="spot"]').forEach((r) => { r.checked = false; });
  $("#rod-options").hidden = true;
  $("#spot-options").hidden = true;
  $("#spot-skipped").hidden = true;
  $("#spot-question-wrap").hidden = false;
  for (const id of ["res-5", "res-6", "res-7", "res-8", "res-10", "res-11"]) $(`#${id}`).innerHTML = "";
  for (const n of [5, 9, 10, 11]) clearMsg(n);
  Object.assign(state, { conv: null, maskKind: "none", rod: null, spot: null, plotsUnlocked: false });
  lockFrom(5);
}

// ---------------------------------------------------------------------------
// 2. instrument table
// ---------------------------------------------------------------------------

function renderPro() {
  const s = state.pro.summary;
  const rows = [
    ["Detector bins", String.raw`n_x \times n_y`, String.raw`${s.nx} \times ${s.ny}`, ""],
    ["Wavelength", String.raw`\lambda`, texNum(num(s.wavelength, 10)), String.raw`\mathrm{nm}`],
    ["Incidence angle", String.raw`\alpha_i`, texNum(num(s.alpha_i, 6)), String.raw`\mathrm{deg}`],
    ["Sample–detector distance", String.raw`L = |\mathbf N|`, String(Math.round(s.L)), String.raw`\mathrm{mm}`],
  ];
  setMath($("#pro-table"),
    "<thead><tr><th>Quantity</th><th>Symbol</th><th>Value</th><th>Unit</th></tr></thead><tbody>" +
    rows.map(([q, sym, val, unit]) =>
      `<tr><td>${q}</td><td>${tex(sym)}</td><td>${tex(val)}</td><td>${unit ? tex(unit) : ""}</td></tr>`).join("") +
    "</tbody>");
}

// ---------------------------------------------------------------------------
// 3. raw data
// ---------------------------------------------------------------------------

/** A df.head() table: plain text, as pandas prints it. */
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

/** [lo, hi] mm, in TeX. */
function texRange(range, digits = 6) {
  return tex(String.raw`[${texNum(num(range[0], digits))},\ ${texNum(num(range[1], digits))}]\ \mathrm{mm}`);
}

function renderDat() {
  const d = state.dat;
  setMath($("#dat-facts"), facts([
    ["Pixels", tex(texNum(int(d.pixels)))],
    ["x range", texRange(d.x_range)],
    ["y range", texRange(d.y_range)],
  ]));
  renderHead($("#dat-head"), d.head);
}

// ---------------------------------------------------------------------------
// 5-9. conversion
// ---------------------------------------------------------------------------

/** Before converting: where the specular spot sits with the .pro alpha_i.
    Off by more than a pixel, the user is asked whether to correct alpha_i
    from the spot; the answer runs the conversion. */
async function checkAlpha(token) {
  const box = $("#alpha-check");
  const button = $("#btn-convert");
  button.disabled = true;
  try {
    const r = await py("check_alpha", {}, [], "Finding the specular spot…");
    if (token !== state.loadToken) return;
    state.alphaCheck = r.meta;
  } catch (err) {
    if (token !== state.loadToken) return;
    state.alphaCheck = null;
    box.hidden = false;
    box.innerHTML = `<p class="muted">${esc(capitalise(err.message))}. The incidence angle from the .pro file is used.</p>`;
    button.disabled = false;
    return;
  }
  const c = state.alphaCheck;
  const deg = (v) => tex(String.raw`${texNum(fixed(v, 4))}^\circ`);
  let html = `<p>${tex(String.raw`\alpha_i`)} = ${deg(c.alpha_i)} (<code>.pro</code>); the specular spot is at ` +
    `${tex(String.raw`\alpha_f`)} = ${deg(c.spot.alpha_f)}.</p>`;
  if (c.needs_correction) {
    html += `<fieldset class="question"><legend>Correct ${tex(String.raw`\alpha_i`)} from the specular spot?</legend>` +
      `<label class="radio"><input type="radio" name="alpha-fix" value="yes"> Yes, ${tex(String.raw`\alpha_i`)} = ${deg(c.alpha_i_corrected)}</label>` +
      `<label class="radio"><input type="radio" name="alpha-fix" value="no"> No, keep ${deg(c.alpha_i)}</label></fieldset>`;
  } else {
    html += `<p class="muted">That is ${tex(String.raw`\alpha_i`)} within one pixel: no correction needed.</p>`;
  }
  box.hidden = false;
  setMath(box, html);
  button.disabled = c.needs_correction;           // the answer converts
}

$("#alpha-check").addEventListener("change", (event) => {
  if (event.target.name !== "alpha-fix") return;
  state.alphaFix = event.target.value;
  if (state.conv) {
    resetConversion();
    unlock(5);
  }
  convert();
});

$("#btn-convert").addEventListener("click", () => convert());

async function convert() {
  const button = $("#btn-convert");
  const answers = $$('input[name="alpha-fix"]');
  clearMsg(5);
  button.disabled = true;
  answers.forEach((r) => { r.disabled = true; });
  const token = state.loadToken;
  try {
    // The detector x, y are used as they are in the .dat file: the direct
    // beam is at the origin.  alpha_i is the .pro value, or the one the
    // specular spot gives if the user chose the correction.
    const r = await py("convert", { correct_offset: false, correct_alpha: state.alphaFix === "yes" },
                       [], "Converting…");
    if (token !== state.loadToken) return;
    state.conv = r.meta;
    state.version = r.meta.version;
    state.maskKind = "none";
    if (r.meta.offset_error) {
      showMsg(5, capitalise(r.meta.offset_error), "warn");
    }
    renderConversion();
    for (const n of [5, 6, 7, 8]) { unlock(n); markDone(n); }
    unlock(9);
    unlock(10);
    await refreshConvertedHead();
    markDone(9);
    idle(`Converted ${int(state.dat.pixels)} pixels; ${int(r.meta.steps["8"].pixels)} lie above the sample horizon.`);
  } catch (err) {
    button.disabled = !!(state.alphaCheck && state.alphaCheck.needs_correction && !state.alphaFix);
    showError(5, err);
  } finally {
    answers.forEach((r) => { r.disabled = false; });
  }
}

function rangeTable(rows) {
  return `<div class="table-wrap"><table class="plain"><thead><tr><th>column</th><th>quantity</th>` +
    `<th>minimum</th><th>maximum</th><th>unit</th></tr></thead><tbody>` +
    rows.map(([col, label, range, unit, factor = 1, digits = 7]) =>
      `<tr><td><code>${col}</code></td><td>${tex(label)}</td><td>${tex(texNum(num(range[0] * factor, digits)))}</td>` +
      `<td>${tex(texNum(num(range[1] * factor, digits)))}</td><td>${tex(unit)}</td></tr>`).join("") +
    "</tbody></table></div>";
}

function renderConversion() {
  const st = state.conv.steps;
  const u = UNITS[state.qUnit];
  const deg = String.raw`\mathrm{deg}`;
  const fix = state.conv.alpha_fix;
  const used = fix ? ` with ${tex(String.raw`\alpha_i = ${texNum(fixed(fix.alpha_i, 4))}^\circ`)}, corrected from the specular spot` : "";
  setMath($("#res-5"),
    `<p class="note">Added 2 columns for all ${tex(texNum(int(state.dat.pixels)))} pixels${used}.</p>` +
    rangeTable([["phi_f", String.raw`\varphi_f`, st["5"].phi_f, deg],
                ["alpha_f", String.raw`\alpha_f`, st["5"].alpha_f, deg]]));
  const dev = st["6"].max_rel_dev.toExponential(1).split("e");
  setMath($("#res-6"), `<p class="note">Added 3 columns. Check: ${tex(String.raw`|\mathbf k_f| = k = ` +
      String.raw`${texNum(num(st["6"].k * u.factor, 9))}\ ${u.tex}`)} for every pixel (largest relative deviation ` +
    `${tex(String.raw`${dev[0]} \times 10^{${Number(dev[1])}}`)}).</p>` +
    rangeTable(["kfx", "kfy", "kfz"].map((c, i) => [c, `k_{f${"xyz"[i]}}`, st["6"][c], u.tex, u.factor])));
  setMath($("#res-7"), `<p class="note">Added 3 columns.</p>` +
    rangeTable(["qx", "qy", "qz"].map((c, i) => [c, `q_${"xyz"[i]}`, st["7"][c], u.tex, u.factor])));
  const s8 = st["8"];
  setMath($("#res-8"), `<dl class="facts">` + facts([
    [`Sample horizon, ${tex(String.raw`y = L\tan\alpha_i`)}`, tex(String.raw`${texNum(num(s8.horizon_y, 6))}\ \mathrm{mm}`)],
    ["Rows dropped", tex(String.raw`${texNum(int(s8.rows_dropped))}\ \text{of}\ ${texNum(int(s8.rows_before))}`)],
    ["Rows kept", tex(texNum(int(s8.rows_kept)))],
    ["Pixels kept", tex(String.raw`${texNum(int(s8.rows_kept))} \times ${texNum(int(s8.nx))} = ${texNum(int(s8.pixels))}`)],
  ]) + "</dl>");
}

async function refreshConvertedHead() {
  const r = await py("table_head", { which: "converted", q_unit: state.qUnit }, [], "Preparing the table…");
  renderHead($("#conv-head"), r.meta.head);
  setMath($("#conv-summary"),
    `${tex(texNum(int(r.meta.pixels)))} pixels (${tex(String.raw`${texNum(int(r.meta.ny))} \times ${texNum(int(r.meta.nx))}`)}), ` +
    `12 columns. First few rows:`);
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

function maskedDownloads(stepNo, nMasked) {
  return `<div class="actions"><span>${int(nMasked)} pixels masked. Masked table:</span>` +
    `<button type="button" data-download-table="masked" data-fmt="csv.gz" data-msg="${stepNo}">Download .csv.gz</button>` +
    `<button type="button" data-download-table="masked" data-fmt="csv" data-msg="${stepNo}">Download plain .csv</button>` +
    `</div>`;
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
      args.follow_peak = true;          // always: the point is to mask the spot
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

function renderRodReport(m) {
  $("#res-10").innerHTML = maskedDownloads(10, m.n_masked);
}

function renderSpotReport(m) {
  $("#res-11").innerHTML = maskedDownloads(11, m.n_masked);
}

function unlockPlots() {
  if (state.plotsUnlocked) return;
  state.plotsUnlocked = true;
  for (const n of [12, 13]) unlock(n);
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
                             ["spot-nsigma", () => onSpotAnswer("yes")], ["spot-window", () => onSpotAnswer("yes")]]) {
  $(`#${id}`).addEventListener("keydown", (e) => { if (e.key === "Enter") handler(); });
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
const FONT = { family: "'Source Sans 3', 'Segoe UI', system-ui, Arial, sans-serif", size: 15, color: "#18202c" };
const INK = "#1b2330";

/* An axis: its Plotly title (TeX between $...$, rendered by MathJax), its
   label on the page (TeX), and its hover label (HTML: Plotly hover text
   cannot hold TeX). */
function axisDef(name) {
  const u = UNITS[state.qUnit];
  const defs = {
    x: { tex: "x", unit: String.raw`\mathrm{mm}`, label: "x", fmt: ".3f", hunit: " mm" },
    y: { tex: "y", unit: String.raw`\mathrm{mm}`, label: "y", fmt: ".3f", hunit: " mm" },
    phi_f: { tex: String.raw`\varphi_f`, unit: String.raw`\mathrm{deg}`, label: "φ<sub>f</sub>", fmt: ".4f", hunit: "°" },
    alpha_f: { tex: String.raw`\alpha_f`, unit: String.raw`\mathrm{deg}`, label: "α<sub>f</sub>", fmt: ".4f", hunit: "°" },
    qy: { tex: "q_y", unit: u.tex, label: "q<sub>y</sub>", fmt: `.${u.digits}f`, hunit: ` ${u.html}` },
    qz: { tex: "q_z", unit: u.tex, label: "q<sub>z</sub>", fmt: `.${u.digits}f`, hunit: ` ${u.html}` },
  };
  const d = defs[name];
  const full = String.raw`${d.tex}\ (${d.unit})`;
  // after a number on a profile label: 0.280°, 0.52 nm⁻¹
  const markUnit = d.unit === String.raw`\mathrm{deg}` ? String.raw`^\circ` : String.raw`\ ${d.unit}`;
  return Object.assign(d, { title: `$${full}$`, page: tex(full), markUnit });
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
  raw: { title: String.raw`Raw intensity map: \(y\) vs \(x\) (full detector)`, plain: "raw y vs x",
         x: "x", y: "y", slug: "raw_y-vs-x", container: "#cards-raw" },
  qz_qy: { title: String.raw`\(q_z\) vs \(q_y\)`, plain: "qz vs qy",
           x: "qy", y: "qz", slug: "qz-vs-qy", container: "#cards-maps" },
  af_pf: { title: String.raw`\(\alpha_f\) vs \(\varphi_f\)`, plain: "alpha_f vs phi_f",
           x: "phi_f", y: "alpha_f", slug: "alphaf-vs-phif", container: "#cards-maps" },
  y_x: { title: String.raw`\(y\) vs \(x\) (above the horizon)`, plain: "y vs x",
         x: "x", y: "y", slug: "y-vs-x", container: "#cards-maps" },
};
const PROFILES = {
  phi_f: { title: String.raw`Mean counts per pixel vs \(\varphi_f\)`, plain: "profile vs phi_f",
           axis: "phi_f", slug: "profile-vs-phif", along: "detector columns" },
  qy: { title: String.raw`Mean counts per pixel vs \(q_y\)`, plain: "profile vs qy",
        axis: "qy", slug: "profile-vs-qy", along: "detector columns" },
  alpha_f: { title: String.raw`Mean counts per pixel vs \(\alpha_f\)`, plain: "profile vs alpha_f",
             axis: "alpha_f", slug: "profile-vs-alphaf", along: "detector rows" },
  qz: { title: String.raw`Mean counts per pixel vs \(q_z\)`, plain: "profile vs qz",
        axis: "qz", slug: "profile-vs-qz", along: "detector rows" },
};

/** For each profile label line at xs, the side its text goes: towards the
    wider gap to the next line or to the axis edge, so neighbouring labels
    part.  The same rule as figures.mark_sides() for the exported figure. */
function markSides(xs, lo, hi) {
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const sides = new Array(xs.length);
  order.forEach((i, k) => {
    const left = xs[i] - (k > 0 ? xs[order[k - 1]] : lo);
    const right = (k + 1 < order.length ? xs[order[k + 1]] : hi) - xs[i];
    sides[i] = right >= left ? "right" : "left";
  });
  return sides;
}

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
    this.limitAxes = $(".opt-ymin", this.el) ? ["x", "y"] : ["x"];
    $$(".dl button", this.el).forEach((b) => b.addEventListener("click", () => exportCard(this, b.dataset.fmt, b)));
    $(".opt-full", this.el).addEventListener("click", () => {
      this.clearLimits();
      if (this.data) this.draw();
    });
    $(container).append(this.el);
    typeset($(".card-options", this.el));
  }

  /** Axis limits typed in the card: {x: [lo, hi], y: [lo, hi]}, null = the data's own. */
  limits() {
    const out = {};
    for (const axis of this.limitAxes) {
      const lo = readNumber($(`.opt-${axis}min`, this.el), { optional: true, name: `The ${axis}-axis "from"` });
      const hi = readNumber($(`.opt-${axis}max`, this.el), { optional: true, name: `The ${axis}-axis "to"` });
      if (lo != null && hi != null && lo >= hi) throw new Error(`${axis}-axis limits: "from" must be smaller than "to".`);
      out[axis] = [lo, hi];
    }
    return out;
  }

  clearLimits() {
    for (const axis of this.limitAxes) {
      $(`.opt-${axis}min`, this.el).value = "";
      $(`.opt-${axis}max`, this.el).value = "";
    }
  }

  /** A mouse zoom fills in the limit boxes, so the download shows what the
      screen shows; zooming back out to the full range empties them. */
  syncLimits(event) {
    for (const axis of this.limitAxes) {
      const key = `${axis}axis`;
      let range = event[`${key}.range`] || (event[`${key}.range[0]`] !== undefined
        ? [event[`${key}.range[0]`], event[`${key}.range[1]`]] : null);
      if (event[`${key}.autorange`]) range = "auto";
      if (!range) continue;
      const lo = $(`.opt-${axis}min`, this.el), hi = $(`.opt-${axis}max`, this.el);
      const full = this.fullRange(axis);
      const tol = full ? 1e-6 * Math.abs(full[1] - full[0]) : 0;
      if (range === "auto" || (full && Math.abs(range[0] - full[0]) <= tol && Math.abs(range[1] - full[1]) <= tol)) {
        lo.value = "";
        hi.value = "";
      } else {
        lo.value = String(Number(Number(range[0]).toPrecision(6)));
        hi.value = String(Number(Number(range[1]).toPrecision(6)));
      }
    }
  }

  /** Limits typed in one q unit, converted when the unit changes. */
  rescaleLimits(factor) {
    for (const axis of this.qAxes || []) {
      for (const end of ["min", "max"]) {
        const input = $(`.opt-${axis}${end}`, this.el);
        if (input.value.trim() !== "" && Number.isFinite(Number(input.value))) {
          input.value = String(Number((Number(input.value) * factor).toPrecision(6)));
        }
      }
    }
  }

  async refresh() {
    const seq = ++this.seq;
    try {
      const key = this.currentDataKey();
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

  /** A TeX label in the card, typeset again only when it changes. */
  setLabel(selector, html) {
    const el = $(selector, this.el);
    if (el.dataset.html === html) return;
    el.dataset.html = html;
    setMath(el, html);
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
    this.qAxes = this.usesQ ? ["x", "y"] : [];
    this.label = def.plain;
    setMath($(".card-title", this.el), def.title);
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
    const lim = this.limits();
    return { log: o.log, cmap: o.cmap, vmin: o.vmin, vmax: o.vmax, zero_colour: o.zero,
             xlim: lim.x, ylim: lim.y };
  }

  /** The axis range shown with no limits: the data plus half a pixel. */
  fullRange(axis) {
    if (!this.data) return null;
    const m = this.data.meta;
    return axis === "x" ? [m.x_range[0] - m.dx / 2, m.x_range[1] + m.dx / 2]
                        : [m.y_range[0] - m.dy / 2, m.y_range[1] + m.dy / 2];
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
    let o, lim;
    try {
      o = this.options();
      lim = this.limits();
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
    const where = `${ax.label} = %{x:${ax.fmt}}${ax.hunit}<br>${ay.label} = %{y:${ay.fmt}}${ay.hunit}`;
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

    this.setLabel(".lim-x", ax.page);
    this.setLabel(".lim-y", ay.page);
    const fullX = this.fullRange("x"), fullY = this.fullRange("y");
    const range = (l, full) => [l[0] == null ? full[0] : l[0], l[1] == null ? full[1] : l[1]];
    const layout = {
      font: FONT, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
      margin: { l: 10, r: 10, t: 36, b: 10 },
      xaxis: axisStyle(ax.title, { range: range(lim.x, fullX), constrain: "domain" }),
      yaxis: axisStyle(ay.title, { range: range(lim.y, fullY), constrain: "domain",
                                   scaleanchor: "x", scaleratio: 1 }),
      legend: { orientation: "h", x: 0, xanchor: "left", y: 1.0, yanchor: "bottom",
                itemsizing: "constant", font: { size: 13 } },
      hovermode: "closest",
      hoverlabel: { font: { family: FONT.family, size: 12 } },
      // A new typed limit replaces any mouse zoom; otherwise the zoom stays.
      uirevision: `${this.kind}|${this.usesQ ? state.qUnit : ""}|${state.loadToken}|${JSON.stringify(lim)}`,
    };

    Plotly.react(this.plot, traces, layout, PLOT_CONFIG).then(() => {
      if (!this.sizeListener) {
        this.sizeListener = (event) => {
          this.syncLimits(event);
          this.resize();
        };
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
    this.qAxes = this.usesQ ? ["x"] : [];
    this.label = def.plain;
    setMath($(".card-title", this.el), def.title);
    for (const cls of [".opt-log", ".opt-xmin", ".opt-xmax"]) {
      $(cls, this.el).addEventListener("change", () => { if (this.data) this.draw(); });
    }
    this.zoomListener = null;
    const smooth = $(".opt-smooth", this.el);
    smooth.value = String(state.init.default_smoothing);
    // Smoothing is done in Python (gisaxs.smooth_profile), the same code the
    // exported figure uses, so a change re-fetches this one small profile.
    smooth.addEventListener("change", () => {
      try {
        this.smoothing();
      } catch (err) {
        this.note.textContent = err.message;
        return;
      }
      this.refresh();
    });

    // Labelled vertical lines: [{value, text (as typed), style (index)}].
    this.marks = [];
    const ax = axisDef(def.axis);
    this.setLabel(".mark-sym", tex(`${ax.tex} =`));
    this.setLabel(".mark-unit", tex(ax.markUnit));
    const mark = $(".opt-mark", this.el);
    $(".opt-mark-add", this.el).addEventListener("click", () => this.addMark());
    mark.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); this.addMark(); }
    });
    $(".mark-list", this.el).addEventListener("click", (event) => {
      const button = event.target.closest("button[data-mark]");
      if (!button) return;
      this.marks.splice(Number(button.dataset.mark), 1);
      this.renderMarks();
      if (this.data) this.draw();
    });
  }

  /** The label text for a value, in TeX: "\alpha_f = 0.280^\circ". */
  markTex(text) {
    const ax = axisDef(this.def.axis);
    return String.raw`${ax.tex} = ${texNum(text)}${ax.markUnit}`;
  }

  addMark() {
    const input = $(".opt-mark", this.el);
    const text = input.value.trim();
    const value = Number(text);
    if (text === "" || !Number.isFinite(value)) {
      this.note.textContent = "Type a number in the label box first.";
      return;
    }
    const used = new Set(this.marks.map((m) => m.style));
    let style = 0;
    while (used.has(style)) style += 1;
    this.marks.push({ value, text, style });
    input.value = "";
    this.renderMarks();
    if (this.data) this.draw();
  }

  /** The labels as chips under the options, each with its line style and a remove button. */
  renderMarks() {
    const styles = state.init.colours.marks;
    const dashes = { dot: "2 2.5", dash: "5 3", dashdot: "5 2 1.5 2", longdash: "8 3" };
    setMath($(".mark-list", this.el), this.marks.map((m, i) => {
      const s = styles[m.style % styles.length];
      return `<span class="mark-chip" style="--c:${s.color}">` +
        `<svg width="6" height="20" aria-hidden="true"><line x1="3" y1="1" x2="3" y2="19" stroke="${s.color}" ` +
        `stroke-width="2" stroke-dasharray="${dashes[s.dash]}"/></svg>${tex(this.markTex(m.text))}` +
        `<button type="button" data-mark="${i}" title="Remove this label" aria-label="Remove this label">×</button></span>`;
    }).join(""));
  }

  /** Limits and labels typed in one q unit, converted when the unit changes;
      a label keeps the number of significant digits it was typed with. */
  rescaleLimits(factor) {
    super.rescaleLimits(factor);
    if (!this.usesQ || !this.marks.length) return;
    for (const m of this.marks) {
      const digits = Math.max(1, m.text.replace(/^[+-]/, "").replace(/e.*$/i, "")
        .replace(".", "").replace(/^0+/, "").length);
      m.value *= factor;
      const text = m.value.toPrecision(Math.min(digits, 21));
      m.text = /e/i.test(text) ? String(Number(text)) : text;
    }
    this.renderMarks();
  }

  /** The smoothing window typed in the card: an odd whole number, 1 = none. */
  smoothing() {
    const text = $(".opt-smooth", this.el).value.trim();
    const n = Number(text);
    if (text === "" || !Number.isInteger(n) || n < 1 || n % 2 === 0) {
      throw new Error("Smoothing must be an odd whole number of points (1, 3, 5, 7, …; 1 = no smoothing), " +
                      "so the running mean stays centred.");
    }
    return n;
  }

  currentDataKey() {
    let smooth;
    try { smooth = this.smoothing(); } catch (_) { smooth = "invalid"; }
    return `${state.loadToken}|${state.version}|${this.usesQ ? state.qUnit : ""}|${smooth}`;
  }

  fetch() {
    return py("profile_data", { kind: this.kind, q_unit: state.qUnit, smooth: this.smoothing() },
              [], "Computing the profile…");
  }

  exportArgs() {
    return { log: $(".opt-log", this.el).checked, smooth: this.smoothing(), xlim: this.limits().x,
             marks: this.marks.map((m) => ({ x: m.value, text: m.text, style: m.style })) };
  }

  /** The x range shown with no limits: the data's own. */
  fullRange() {
    if (!this.data) return null;
    const x = this.data.arrays.x;
    return [x[0], x[x.length - 1]];
  }

  draw() {
    let lim;
    try {
      lim = this.limits();
    } catch (err) {
      this.note.textContent = err.message;
      return;
    }
    const { meta, arrays } = this.data;
    const log = $(".opt-log", this.el).checked;
    const ax = axisDef(this.def.axis);
    const colours = state.init.colours.profile;
    this.setLabel(".lim-x", ax.page);

    // Only the points inside the x limits are drawn, so the y-axis fits them
    // (the export does the same).  Smoothing ran on the whole profile first.
    const lo = lim.x[0] == null ? -Infinity : lim.x[0];
    const hi = lim.x[1] == null ? Infinity : lim.x[1];
    const keep = [];
    for (let i = 0; i < arrays.x.length; i++) {
      if (arrays.x[i] >= lo && arrays.x[i] <= hi) keep.push(i);
    }
    if (keep.length < 2) {
      this.note.textContent = "Fewer than 2 points lie inside the x-axis limits.";
      return;
    }
    const pick = (a) => Float64Array.from(keep, (i) => a[i]);
    const x = pick(arrays.x), measured = pick(arrays.raw), smoothed = pick(arrays.y);
    const smooth = meta.smooth > 1;
    const where = `${ax.label} = %{x:${ax.fmt}}${ax.hunit}`;
    const spikes = { showspikes: true, spikemode: "across", spikesnap: "data", spikethickness: 1,
                     spikedash: "dot", spikecolor: "#6b7585" };

    // The measured curve faded underneath, the smoothed one on top.
    const traces = smooth ? [
      { type: "scatter", mode: "lines", name: "measured", x, y: measured, connectgaps: false,
        line: { color: colours.measured, width: 1 }, hoverinfo: "skip" },
      { type: "scatter", mode: "lines", name: `smoothed (${meta.smooth}-point running mean)`,
        x, y: smoothed, customdata: measured, connectgaps: false,
        line: { color: colours.smoothed, width: 2 },
        hovertemplate: `${where}, I = %{y:.4~r} cts (measured %{customdata:.4~r})<extra></extra>` },
    ] : [
      { type: "scatter", mode: "lines", name: "measured", x, y: measured, connectgaps: false,
        line: { color: colours.smoothed, width: 1.5 },
        hovertemplate: `${where}, I = %{y:.4~r} cts<extra></extra>` },
    ];
    const limited = lim.x[0] != null || lim.x[1] != null;
    const xlo = lim.x[0] == null ? x[0] : lim.x[0];
    const xhi = lim.x[1] == null ? x[x.length - 1] : lim.x[1];
    const xRange = limited ? { range: [xlo, xhi], autorange: false } : { autorange: true };

    // Labelled vertical lines, the text at the top on the side with more
    // room, and head room above the data for it (as figures._draw_marks).
    this.setLabel(".mark-sym", tex(`${ax.tex} =`));
    this.setLabel(".mark-unit", tex(ax.markUnit));
    const markStyles = state.init.colours.marks;
    const shown = this.marks.filter((m) => m.value >= xlo && m.value <= xhi);
    const sides = markSides(shown.map((m) => m.value), xlo, xhi);
    const shapes = [], annotations = [];
    shown.forEach((m, i) => {
      const s = markStyles[m.style % markStyles.length];
      const right = sides[i] === "right";
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: m.value, x1: m.value, y0: 0, y1: 1,
                    line: { color: s.color, width: 2, dash: s.dash } });
      annotations.push({ xref: "x", yref: "paper", x: m.value, y: 1, xanchor: right ? "left" : "right",
                         yanchor: "top", xshift: right ? 5 : -5, yshift: -4, showarrow: false,
                         text: `$${this.markTex(m.text)}$`, font: { color: s.color, size: 17 } });
    });
    let yRange = { autorange: true };
    if (shown.length) {
      let ylo = Infinity, yhi = -Infinity;
      for (const series of smooth ? [measured, smoothed] : [measured]) {
        for (const v of series) {
          if (!Number.isFinite(v) || (log && v <= 0)) continue;
          const t = log ? Math.log10(v) : v;
          if (t < ylo) ylo = t;
          if (t > yhi) yhi = t;
        }
      }
      if (ylo < yhi) {
        const pad = 0.05 * (yhi - ylo);          // matplotlib's default margins
        ylo -= pad;
        yhi += pad;
        yRange = { range: [ylo, ylo + (yhi - ylo) / (1 - state.init.colours.mark_headroom)], autorange: false };
      }
    }

    const layout = {
      font: FONT, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
      margin: { l: 10, r: 16, t: smooth ? 30 : 16, b: 10 },
      xaxis: axisStyle(ax.title, Object.assign(xRange, spikes)),
      yaxis: axisStyle("Mean counts per pixel", Object.assign({ type: log ? "log" : "linear" }, yRange, spikes)),
      shapes, annotations,
      hovermode: "closest", hoverdistance: -1, spikedistance: -1,
      showlegend: smooth,
      legend: { orientation: "h", x: 0, xanchor: "left", y: 1.0, yanchor: "bottom", font: { size: 13 } },
      hoverlabel: { font: { family: FONT.family, size: 12 } },
      uirevision: `${this.kind}|${this.usesQ ? state.qUnit : ""}|${state.loadToken}|${log}|${meta.smooth}|` +
                  `${JSON.stringify(lim)}|${shown.map((m) => m.value).join(",")}`,
    };
    Plotly.react(this.plot, traces, layout, PLOT_CONFIG).then(() => {
      if (!this.zoomListener) {
        this.zoomListener = (event) => this.syncLimits(event);
        this.plot.on("plotly_relayout", this.zoomListener);
      }
    });
    $(".card-meta", this.el).textContent = `${int(meta.n)} ${this.def.along}, ${maskLabel()}`;
    const smoothing = meta.smooth > 1 ? `smoothed: centred running mean of ${meta.smooth} points` : "not smoothed";
    const hidden = this.marks.length - shown.length;
    this.note.textContent = `${smoothing} · ` + (meta.n_gaps
      ? `${int(meta.n_gaps)} fully masked ${this.def.along.replace("detector ", "")} drawn as gaps`
      : "no fully masked columns or rows") +
      (hidden ? ` · ${hidden === 1 ? "1 label lies" : `${hidden} labels lie`} outside the plotted range` : "");
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
}

// ---------------------------------------------------------------------------
// figure export (matplotlib in Python), from the buttons under each plot
// ---------------------------------------------------------------------------

function exportSettings() {
  return { preset: $("#export-preset").value, columns: Number($("#export-columns").value) };
}

async function exportCard(card, fmt, button) {
  let extra;
  try {
    extra = card.exportArgs();
  } catch (err) {
    card.note.textContent = err.message;
    return;
  }
  const name = card.fileName(fmt);
  button.disabled = true;
  try {
    const r = await py("export_figure", Object.assign({
      kind: card.kind, fmt, profile: card.isProfile, q_unit: state.qUnit,
    }, exportSettings(), extra), [], `Drawing ${name}…`);
    saveBlob(new Blob([r.arrays.file], { type: r.meta.mime }), name);
    idle(`Saved ${name} (${mb(r.meta.bytes)}, drawn in ${r.meta.seconds.toFixed(1)} s).`);
  } catch (err) {
    const message = capitalise(err.message || err);
    card.note.textContent = `Could not save ${name}: ${message}`;
    setStatus(`Error: ${message.split("\n")[0]}`, "error");
  } finally {
    button.disabled = false;
  }
}

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
  const factor = UNITS[unit].factor / UNITS[state.qUnit].factor;
  state.qUnit = unit;
  for (const card of cards.values()) card.rescaleLimits(factor);
  $$(".segmented button[data-unit]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.unit === unit)));
  if (state.pro) renderPro();
  if (state.conv) {
    renderConversion();
    refreshConvertedHead().catch((err) => showError(9, err));
  }
  for (const card of cards.values()) if (card.usesQ) card.refresh();
}));

// ---------------------------------------------------------------------------
// start
// ---------------------------------------------------------------------------

if (!window.Plotly) {
  setStatus("Plotly could not be loaded from cdn.jsdelivr.net; plots will not work. Check the connection and reload.", "error");
}

})();
