/* Pyodide in a (module) Web Worker: all the analysis runs here, in Python.
 *
 * A module worker loads Pyodide with import(); some embedded browsers refuse
 * cross-origin importScripts() in classic workers, but allow this.
 *
 * Messages in:   {id, cmd, args, files: [{path, file}]}
 * Messages out:  {type: "progress", message}
 *                {type: "chunk", id, data}                 (streamed downloads)
 *                {type: "result", id, meta, arrays}        (typed arrays transferred)
 *                {type: "error", id, message, traceback}
 *                {type: "ready", meta} / {type: "fatal", message}
 *
 * Python commands live in py/bridge.py; each returns
 * {"meta": JSON text, "arrays": {name: ndarray}, "stream": generator of bytes}.
 */

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PACKAGES = ["numpy", "pandas", "matplotlib"];
// Provided modules first (unchanged), then the app's own.
const PY_FILES = ["read_pro.py", "mask.py", "plotstyle.py",
                  "gisaxs.py", "spot.py", "figures.py", "bridge.py"];
const PY_DIR = "/home/pyodide/app";
const DATA_DIR = "/home/pyodide/data";

function progress(message) {
  self.postMessage({ type: "progress", message: String(message) });
}
// Called from Python (bridge.progress) via `from js import reportProgress`.
self.reportProgress = progress;

let pyodide = null;
let dispatch = null;

async function start() {
  progress(`Loading Python (Pyodide ${PYODIDE_VERSION})…`);
  const { loadPyodide } = await import(PYODIDE_URL + "pyodide.mjs");
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  progress("Loading numpy, pandas and matplotlib…");
  await pyodide.loadPackage(PACKAGES, { messageCallback: () => {} });

  progress("Loading the analysis modules…");
  pyodide.FS.mkdirTree(PY_DIR);
  pyodide.FS.mkdirTree(DATA_DIR);
  await Promise.all(PY_FILES.map(async (name) => {
    // Relative to this script (js/worker.js), so it works under /<repo>/.
    const url = new URL(`../py/${name}`, self.location.href);
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) throw new Error(`could not fetch py/${name} (${response.status})`);
    pyodide.FS.writeFile(`${PY_DIR}/${name}`, await response.text());
  }));

  progress("Starting Python…");
  pyodide.runPython(`
import os, sys
os.environ["MPLBACKEND"] = "AGG"
sys.path.insert(0, "${PY_DIR}")
import bridge
`);
  dispatch = pyodide.globals.get("bridge").dispatch;
  const { meta } = call("init", {});
  if (meta.error) throw new Error(meta.error);
  return meta;
}

/* Copy a Python buffer (ndarray / bytes) out of the WASM heap. */
function takeBuffer(proxy) {
  const buffer = proxy.getBuffer();
  try {
    return buffer.data.slice();
  } finally {
    buffer.release();
    proxy.destroy();
  }
}

/* Run one bridge command and unpack its result. */
function call(cmd, args, id) {
  const result = dispatch(cmd, JSON.stringify(args || {}));
  try {
    const meta = JSON.parse(result.get("meta"));
    const arrays = {};
    const transfer = [];
    const pyArrays = result.get("arrays");
    if (pyArrays) {
      for (const key of pyArrays) {
        const typed = takeBuffer(pyArrays.get(key));
        arrays[key] = typed;
        transfer.push(typed.buffer);
      }
      pyArrays.destroy();
    }
    const stream = result.get("stream");
    if (stream) {
      try {
        let bytes = 0;
        for (const piece of stream) {
          const data = takeBuffer(piece);
          bytes += data.byteLength;
          self.postMessage({ type: "chunk", id, data }, [data.buffer]);
        }
        meta.bytes = bytes;
      } finally {
        stream.destroy();
      }
    }
    return { meta, arrays, transfer };
  } finally {
    result.destroy();
  }
}

const ready = start().then(
  (meta) => {
    self.postMessage({ type: "ready", meta, pyodide: PYODIDE_VERSION });
    return true;
  },
  (err) => {
    self.postMessage({ type: "fatal", message: String(err && err.message || err) });
    return false;
  });

/* One command at a time: Python is single-threaded. */
let queue = Promise.resolve();
self.onmessage = (event) => {
  queue = queue.then(() => handle(event.data));
};

async function handle({ id, cmd, args, files }) {
  const written = [];
  try {
    if (!(await ready)) throw new Error("Python did not start; reload the page.");
    for (const { path, file } of files || []) {
      progress(`Reading ${file.name} (${(file.size / 1e6).toFixed(1)} MB)…`);
      pyodide.FS.writeFile(path, new Uint8Array(await file.arrayBuffer()));
      written.push(path);
    }
    const { meta, arrays, transfer } = call(cmd, args, id);
    if (meta && meta.error) {
      self.postMessage({ type: "error", id, message: meta.error,
                         traceback: meta.traceback, userError: !!meta.user_error });
    } else {
      self.postMessage({ type: "result", id, meta, arrays }, transfer);
    }
  } catch (err) {
    // Errors raised while streaming (inside a Python generator) land here.
    const text = String(err && err.message || err);
    const lines = text.trim().split("\n");
    self.postMessage({ type: "error", id, message: lines[lines.length - 1], traceback: text });
  } finally {
    // The uploaded files are parsed; free the memory they take in the virtual FS.
    for (const path of written) {
      try { pyodide.FS.unlink(path); } catch (_) { /* already gone */ }
    }
  }
}
