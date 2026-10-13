"""The browser side of the app: session state and the commands js/worker.js calls.

Every command takes keyword arguments (sent as JSON) and returns a dict

    {"meta": <JSON text>, "arrays": {name: numpy array}, "stream": generator}

"arrays" are sent to JavaScript as typed arrays (float32 for the maps) and
"stream" yields bytes for large downloads.  Errors come back as
{"meta": '{"error": ...}'} so the page can show them.

The analysis itself lives in gisaxs.py, mask.py, spot.py and figures.py; this
module only keeps the frames between steps, so changing a display option never
re-runs the conversion.
"""

import dataclasses
import json
import math
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

import figures
import gisaxs
import mask
import read_pro
import spot
from gisaxs import InputError

try:                                    # in the Web Worker
    from js import reportProgress as _report
except ImportError:                     # plain CPython
    def _report(message):
        print(message, file=sys.stderr)


def progress(message):
    _report(str(message))


class Session:
    """What the app has loaded and computed so far."""

    def __init__(self):
        # The reference frame (direct beam at alpha_i = 0) outlives reset():
        # it serves every measurement of the series.
        self.ref = None          # spot.find_reference_beam() + "name"
        self.reset()

    def reset(self):
        self.pro = None          # read_pro() dict
        self.pro_name = None
        self.summary = None      # gisaxs.instrument_summary()
        self.geom = None         # from the .pro file
        self.geom_used = None    # what the conversion used (alpha_i may be fixed)
        self.offset = None       # the direct-beam offset (spot.find_offset)
        self.calibration = None  # with a reference frame: origin and alpha_i
        self.alpha_check = None  # cmd_check_alpha(): where the specular spot is
        self.alpha_fix = None    # the alpha_i correction the conversion used
        self.dat_name = None
        self.raw = None          # x, y, intensity[, variance]; full detector
        self.cut = None          # converted, rows below the horizon dropped
        self.data = None         # cut, or cut with the rod / spot masked
        self.mask_kind = "none"
        self.mask_text = ""
        self.version = 0         # bumped whenever self.data changes


S = Session()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean(value):
    """JSON-safe copy: numpy scalars -> Python, NaN/inf -> None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _pack(meta=None, arrays=None, stream=None):
    out = {"meta": json.dumps(_clean(meta or {}))}
    if arrays:
        out["arrays"] = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    if stream is not None:
        out["stream"] = stream
    return out


def _f32(values):
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32))


def _fmt(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NaN"
    return "%.8g" % value


def _head(df, n=5):
    """df.head() as {columns, index, rows} of strings."""
    head = df.head(n)
    return {
        "columns": [str(c) for c in head.columns],
        "index": [int(i) for i in head.index],
        "rows": [[_fmt(v) for v in row] for row in head.to_numpy(dtype=float)],
    }


def _range(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None, None]
    return [float(finite.min()), float(finite.max())]


def _require(what):
    checks = {
        "pro": (S.pro, "Load a .pro instrument file first."),
        "raw": (S.raw, "Load a .dat data file first."),
        "cut": (S.cut, "Run the conversion (step 5) first."),
    }
    value, message = checks[what]
    if value is None:
        raise InputError(message)


def _pitch(X, Y):
    """Median spacing of neighbouring pixels along a row and down a column."""
    dx = float(np.median(np.abs(np.diff(X, axis=1)))) if X.shape[1] > 1 else 0.0
    dy = float(np.median(np.abs(np.diff(Y, axis=0)))) if Y.shape[0] > 1 else 0.0
    return dx, dy


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_init():
    import matplotlib
    import pandas
    return _pack({
        "colours": figures.screen_colours(),
        "colormaps": figures.COLORMAPS,
        "default_n_sigma": mask.DEFAULT_N_SIGMA,
        "default_window": spot.DEFAULT_WINDOW,
        "default_smoothing": gisaxs.DEFAULT_SMOOTHING,
        "zero_colours": list(figures.ZERO_COLOURS),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__,
        },
    })


def cmd_load_pro(path, name):
    S.reset()
    try:
        pro = read_pro.read_pro(path)
    except ET.ParseError as err:
        raise InputError(f"“{name}” is not a valid XML file ({err}). "
                         f"Expected a BornAgain 1.7 .pro project file.") from None
    except KeyError as err:
        what = str(err).strip("'\"").split(": ", 1)[-1]
        raise InputError(f"“{name}” is missing an instrument parameter "
                         f"({what}). Expected a BornAgain 1.7 project with a "
                         f"rectangular detector, a wavelength and an "
                         f"inclination angle.") from None
    except (TypeError, ValueError, AttributeError) as err:
        raise InputError(f"“{name}”: an instrument parameter could not be read "
                         f"as a number ({err})") from None
    summary = gisaxs.instrument_summary(pro)
    S.pro, S.pro_name, S.summary = pro, name, summary
    S.geom = gisaxs.Geometry.from_pro(pro)
    return _pack({"summary": summary,
                  "horizon_y": gisaxs.horizon_y(S.geom),
                  "specular_y": S.geom.L * np.tan(np.radians(2 * S.geom.alpha_i))})


def cmd_load_dat(path, name):
    _require("pro")
    S.dat_name = name
    S.raw = S.cut = S.data = None
    S.geom_used = S.offset = S.calibration = None
    S.alpha_check = S.alpha_fix = None
    S.mask_kind, S.mask_text = "none", ""
    t0 = time.time()
    try:
        df = gisaxs.read_dat(path, nx=S.summary["nx"], ny=S.summary["ny"],
                             progress=progress)
    except UnicodeDecodeError:
        raise InputError(f"“{name}” is not a text file. Expected a BornAgain "
                         f".dat 2D dataset.") from None
    S.raw = df
    S.version += 1
    counts = df["intensity"].to_numpy()
    finite = np.isfinite(counts)
    variance = None
    if "variance" in df.columns:
        v = df["variance"].to_numpy()
        variance = "all zero" if np.all(v[np.isfinite(v)] == 0) else "present"
    return _pack({
        "head": _head(df),
        "pixels": len(df),
        "nx": df.attrs["nx"], "ny": df.attrs["ny"],
        "order": df.attrs["order"],
        "dead": int((~finite).sum()),
        "dead_minus_one": df.attrs["dead_minus_one"],
        "zeros": int((counts == 0).sum()),
        "max": float(np.nanmax(counts)) if finite.any() else None,
        "total": float(np.nansum(counts)),
        "variance": variance or "absent",
        "x_range": _range(df["x"].to_numpy()),
        "y_range": _range(df["y"].to_numpy()),
        "seconds": time.time() - t0,
    })


def _find_offset():
    """The direct-beam offset of the loaded frame (spot.find_offset)."""
    progress("Finding the specular spot and the direct beam…")
    try:
        return spot.find_offset(S.raw, S.geom)
    except spot.SpotError as err:
        raise InputError(f"could not find the specular spot, so the direct-beam "
                         f"offset cannot be corrected: {err}") from None


def cmd_load_ref(path, name):
    """Read the reference frame (direct beam at alpha_i = 0, beamstop out) and
    measure the beam: its position is the origin for every measurement."""
    try:
        df = gisaxs.read_dat(path, progress=progress)
    except UnicodeDecodeError:
        raise InputError(f"“{name}” is not a text file. Expected a BornAgain "
                         f".dat 2D dataset.") from None
    progress("Measuring the direct beam…")
    try:
        beam = spot.find_reference_beam(df)
    except spot.SpotError as err:
        raise InputError(f"could not measure the direct beam in “{name}”: {err}") from None
    S.ref = dict(beam, name=name)
    _drop_conversion()
    return _pack(dict(S.ref, version=S.version))


def cmd_clear_ref():
    S.ref = None
    _drop_conversion()
    return _pack({"version": S.version})


def _drop_conversion():
    """A new reference changes the conversion: forget the converted frame."""
    if S.cut is not None:
        S.cut = S.data = S.geom_used = S.offset = None
        S.mask_kind, S.mask_text = "none", ""
        S.version += 1


def _calibrate():
    """With a reference frame: shift the frame so the direct beam is at the
    origin, find the specular spot, and take alpha_i from where it sits.
    Returns (shifted frame, geometry, report, warning or None)."""
    ref = S.ref
    base = gisaxs.shift_detector(S.raw, ref["x0"], ref["y0"])
    report = {"name": ref["name"], "x0": ref["x0"], "y0": ref["y0"],
              "half_cut": ref["half_cut"], "alpha_i_pro": S.geom.alpha_i}
    progress("Finding the specular spot…")
    try:
        # brightest row along x ~ 0, above half the nominal horizon height:
        # clear of the direct beam, below any plausible spot
        found = spot.find_on_detector(base, S.geom.L, S.geom.alpha_i, scan_column=True,
                                      above=0.5 * gisaxs.horizon_y(S.geom))
    except spot.SpotError as err:
        warning = (f"Could not find the specular spot, so alpha_i is the .pro value "
                   f"({S.geom.alpha_i:.4f} deg): {err}")
        report.update(alpha_i=S.geom.alpha_i, spot=None)
        return base, S.geom, report, warning
    alpha_i = gisaxs.alpha_i_from_specular(found["x0"], found["y0"], S.geom.L)
    geom = dataclasses.replace(S.geom, alpha_i=alpha_i)
    phi, alpha = gisaxs.exit_angles(found["x0"], found["y0"], geom)
    report.update(alpha_i=alpha_i, delta=alpha_i - S.geom.alpha_i,
                  spot={"x": found["x0"], "y": found["y0"],
                        "fwhm_x": found["fwhm_x"], "fwhm_y": found["fwhm_y"],
                        "phi_f": float(phi), "alpha_f": float(alpha)})
    return base, geom, report, None


# The specular spot counts as being at alpha_f = alpha_i when it is within
# this many pixels of it; then no correction is offered.
ALPHA_TOLERANCE_PIXELS = 1.0


def cmd_check_alpha():
    """Where the specular spot sits, before converting: alpha_f of the spot
    with the .pro alpha_i, and the alpha_i the spot itself gives.

    The direct beam is at the detector origin, so the spot is 2 alpha_i above
    it: with a zero error, a spot at alpha_f = alpha_i + delta means the true
    incidence angle is about alpha_i + delta / 2, exactly
    gisaxs.alpha_i_from_specular(x_s, y_s, L).  needs_correction is False when
    the spot is within ALPHA_TOLERANCE_PIXELS pixels of alpha_i.
    """
    _require("raw")
    geom = S.geom
    progress("Finding the specular spot…")
    try:
        # brightest row along x ~ 0, above half the horizon height (clear of
        # the direct beam and the beamstop)
        found = spot.find_on_detector(S.raw, geom.L, geom.alpha_i, scan_column=True,
                                      above=0.5 * gisaxs.horizon_y(geom))
    except spot.SpotError as err:
        S.alpha_check = None
        raise InputError(f"could not find the specular spot: {err}") from None
    phi, alpha = (float(v) for v in gisaxs.exit_angles(found["x0"], found["y0"], geom))
    pixel = float(np.degrees(found["pixel_y"] / geom.L))
    delta = alpha - geom.alpha_i
    S.alpha_check = {
        "alpha_i": geom.alpha_i,
        "spot": {"x": found["x0"], "y": found["y0"], "alpha_f": alpha, "phi_f": phi},
        "delta": delta, "pixel_alpha": pixel, "pixels_off": delta / pixel,
        "tolerance_pixels": ALPHA_TOLERANCE_PIXELS,
        "needs_correction": bool(abs(delta) > ALPHA_TOLERANCE_PIXELS * pixel),
        "alpha_i_corrected": gisaxs.alpha_i_from_specular(found["x0"], found["y0"], geom.L),
    }
    return _pack(S.alpha_check)


def cmd_convert(correct_offset=True, use_reference=True, correct_alpha=False):
    """Steps 5-8.

    With a reference frame loaded (cmd_load_ref) and use_reference: every pixel
    is moved so the direct beam measured there is at the origin, and alpha_i
    is taken from the specular spot (gisaxs.alpha_i_from_specular), not from
    the .pro file.  Otherwise, with correct_offset, every pixel is moved by the
    offset spot.find_offset finds from the spot alone, keeping the .pro
    alpha_i.  Otherwise the frame is converted as it is.  If the specular spot
    cannot be found, offset_error says so and why.

    correct_alpha: convert with the alpha_i the specular spot gives
    (cmd_check_alpha) instead of the .pro value; the pixels are not moved."""
    _require("raw")
    geom, offset, x0, y0, offset_error = S.geom, None, 0.0, 0.0, None
    reference = None
    if use_reference and S.ref is not None:
        base, geom, reference, offset_error = _calibrate()
        x0, y0 = S.ref["x0"], S.ref["y0"]
    elif correct_offset:
        try:
            offset = _find_offset()
            x0, y0 = offset["x0"], offset["y0"]
        except InputError as err:
            offset_error = str(err)
    if reference is None:
        base = gisaxs.shift_detector(S.raw, x0, y0) if (x0 or y0) else S.raw
    alpha_fix = None
    if correct_alpha and reference is None:
        if S.alpha_check is None:
            cmd_check_alpha()
        check = S.alpha_check
        geom = dataclasses.replace(geom, alpha_i=check["alpha_i_corrected"])
        alpha_fix = dict(check, alpha_i_pro=check["alpha_i"], alpha_i=geom.alpha_i)
    n = len(base)
    df = base.copy()
    steps = {}

    progress(f"Step 5: exit angles for {n:,} pixels…")
    gisaxs.add_exit_angles(df, geom)
    steps["5"] = {"phi_f": _range(df["phi_f"].to_numpy()),
                  "alpha_f": _range(df["alpha_f"].to_numpy())}

    progress(f"Step 6: final wavevector for {n:,} pixels…")
    gisaxs.add_final_wavevector(df, geom)
    kf = np.sqrt(df["kfx"] ** 2 + df["kfy"] ** 2 + df["kfz"] ** 2).to_numpy()
    steps["6"] = {c: _range(df[c].to_numpy()) for c in gisaxs.KF_COLUMNS}
    steps["6"]["k"] = geom.k
    steps["6"]["max_rel_dev"] = float(np.max(np.abs(kf - geom.k)) / geom.k)

    progress(f"Step 7: scattering vector for {n:,} pixels…")
    gisaxs.add_scattering_vector(df, geom)
    steps["7"] = {c: _range(df[c].to_numpy()) for c in gisaxs.Q_COLUMNS}

    progress("Step 8: dropping rows below the sample horizon…")
    alpha_row = gisaxs.row_alpha_f(df, geom)
    cut = gisaxs.horizon_cut(df, geom)
    del df
    y_row = mask.as_image(base, "y").mean(axis=1)
    kept = alpha_row >= 0
    steps["8"] = {
        "rows_before": int(base.attrs["ny"]),
        "rows_dropped": cut.attrs["rows_dropped"],
        "rows_kept": cut.attrs["ny"],
        "nx": cut.attrs["nx"],
        "pixels": len(cut),
        "horizon_y": gisaxs.horizon_y(geom),
        "first_row_y": float(y_row[kept][0]),
        "first_row_alpha": float(alpha_row[kept][0]),
        "last_dropped_y": float(y_row[~kept][-1]) if (~kept).any() else None,
        "last_dropped_alpha": float(alpha_row[~kept][-1]) if (~kept).any() else None,
    }
    S.cut = cut
    S.data = cut
    S.geom_used, S.offset, S.calibration, S.alpha_fix = geom, offset, reference, alpha_fix
    S.mask_kind, S.mask_text = "none", ""
    S.version += 1
    return _pack({"steps": steps, "version": S.version,
                  "alpha_i": geom.alpha_i, "x0": x0, "y0": y0, "offset": offset,
                  "reference": reference, "alpha_fix": alpha_fix,
                  "offset_error": offset_error})


def cmd_table_head(which="converted", q_unit="nm-1"):
    _require("cut")
    df = S.cut if which == "converted" else S.data
    return _pack({"head": _head(gisaxs.export_frame(df, q_unit)),
                  "pixels": len(df), "nx": df.attrs["nx"], "ny": df.attrs["ny"]})


def _export_meta(df, mask_text=""):
    s = S.summary
    shift = ""
    o = S.offset
    cal = S.calibration
    alpha = ""
    fix = S.alpha_fix
    if fix:
        sp = fix["spot"]
        alpha = (f"alpha_i = {fix['alpha_i']:.6f} deg from the specular spot at x = "
                 f"{sp['x']:.4f} mm, y = {sp['y']:.4f} mm (the direct beam at the origin), "
                 f"alpha_i = atan(sqrt(x^2 + y^2) / L) / 2; with the .pro value "
                 f"{fix['alpha_i_pro']:.6f} deg the spot was at alpha_f = "
                 f"{sp['alpha_f']:.6f} deg")
    if cal:
        edge = ("the lower edge of the half-cut beam, i.e. the sample surface"
                if cal["half_cut"] else "its centroid")
        shift = (f"x_mm = x - {cal['x0']:.4f} mm and y_mm = y - {cal['y0']:.4f} mm, with "
                 f"x, y as in the .dat file: the direct beam measured in the reference "
                 f"frame {cal['name']} ({edge})")
        if cal.get("spot"):
            sp = cal["spot"]
            alpha = (f"alpha_i = {cal['alpha_i']:.6f} deg from the specular spot at "
                     f"x = {sp['x']:.4f} mm, y = {sp['y']:.4f} mm measured from the "
                     f"direct beam, alpha_i = atan(sqrt(x^2 + y^2) / L) / 2; the .pro "
                     f"value {cal['alpha_i_pro']:.6f} deg is not used")
    elif o:
        beam = "the direct beam seen in the frame"
        if o["method_x"] == o["method_y"] == "direct":
            how = f"x0 and y0 from {beam}"
        elif o["method_x"] == o["method_y"]:
            how = "x0 and y0 from the specular spot: x0 = x_s, y0 = y_s - L tan(2 alpha_i)"
        else:
            how = ("x0 from " + (beam if o["method_x"] == "direct" else "the specular spot, x0 = x_s")
                   + ", y0 from " + (beam if o["method_y"] == "direct"
                                     else "the specular spot, y0 = y_s - L tan(2 alpha_i)"))
        shift = (f"x_mm = x - {o['x0']:.4f} mm and y_mm = y - {o['y0']:.4f} mm, with x, y "
                 f"as in the .dat file, so the direct beam is at the origin ({how}); "
                 f"the specular spot, measured at x = {o['spot']['x']:.4f} mm, "
                 f"y = {o['spot']['y']:.4f} mm in the file, is then at "
                 f"phi_f = {o['phi_f_spot']:.6f} deg, alpha_f = {o['alpha_f_spot']:.6f} deg")
    return {
        "shift_text": shift, "alpha_text": alpha,
        "dat_name": S.dat_name, "pro_name": S.pro_name,
        "L": s["L"], "alpha_i": S.geom_used.alpha_i,
        "wavelength": s["wavelength"],
        "k": s["k"], "pitch_x": s["pitch_x"], "pitch_y": s["pitch_y"],
        "nx": df.attrs["nx"], "ny": df.attrs["ny"],
        "rows_dropped": S.cut.attrs["rows_dropped"],
        "mask": mask_text,
    }


def cmd_export_table(which="converted", fmt="csv.gz", q_unit="nm-1"):
    _require("cut")
    if which == "converted":
        df, mask_text = S.cut, ""
    else:
        df, mask_text = S.data, S.mask_text
    header = gisaxs.export_header(_export_meta(df, mask_text), q_unit)
    gz = fmt == "csv.gz"
    n = len(df)

    def report(done, total):
        progress(f"Writing {'compressed ' if gz else ''}CSV: "
                 f"{done:,} of {total:,} rows ({100 * done / total:.0f} %)…")

    stream = gisaxs.iter_csv_bytes(df, header, q_unit,
                                   gzip_level=6 if gz else None,
                                   progress=report)
    mime = "application/gzip" if gz else "text/csv"
    return _pack({"mime": mime, "rows": n}, stream=stream)


def cmd_apply_mask(kind="none", n_sigma=None, half_width=None, window=None,
                   follow_peak=True):
    _require("cut")
    cut = S.cut
    report = {"kind": kind}
    if kind == "none":
        S.data, S.mask_kind, S.mask_text = cut, "none", ""
    elif kind == "rod":
        n_sigma = float(n_sigma if n_sigma is not None else mask.DEFAULT_N_SIGMA)
        progress("Finding the specular rod…")
        try:
            rod = mask.find_rod(cut)
        except ValueError as err:
            raise InputError(f"could not find the specular rod: {err}") from None
        hw = float(half_width) if half_width else n_sigma * rod["sigma"]
        progress("Masking the specular rod…")
        masked, n = mask.mask_rod(cut, rod["centre"], hw)

        # mask.py's step 5: is the rod actually gone?
        after = mask.column_profile(masked)
        after_phi = after.index.to_series()
        left_over = float(after.max() - rod["background"])
        outside = after[(after_phi - rod["centre"]).abs().between(hw, 3.0 * hw)]
        diffuse = float(outside.median() - rod["background"]) if len(outside) else None

        report.update(rod)
        report.update(half_width=hw, n_sigma=None if half_width else n_sigma,
                      fixed=bool(half_width), n_masked=n, pixels=len(cut),
                      left_over=left_over, diffuse=diffuse,
                      band=[rod["centre"] - hw, rod["centre"] + hw])
        how = (f"fixed half-width {hw:.6f} deg" if half_width
               else f"{n_sigma:g} sigma, sigma = {rod['sigma']:.6f} deg")
        S.mask_text = (f"specular rod, |phi_f - {rod['centre']:.6f}| <= "
                       f"{hw:.6f} deg ({how}); {n} pixels set to nan")
        S.data, S.mask_kind = masked, "rod"
    elif kind == "spot":
        n_sigma = float(n_sigma if n_sigma is not None else mask.DEFAULT_N_SIGMA)
        window = float(window if window is not None else spot.DEFAULT_WINDOW)
        progress("Measuring the specular spot…")
        try:
            # The search starts at the brightest row along phi_f ~ 0, so a
            # spot well away from alpha_f = alpha_i is still the one masked.
            masked, info = spot.find_and_mask(cut, S.geom_used.alpha_i,
                                              n_sigma=n_sigma, window=window,
                                              follow_peak=bool(follow_peak),
                                              scan_column=True)
        except spot.SpotError as err:
            raise InputError(f"could not measure the specular spot: {err}") from None
        report.update(info)
        report.update(pixels=len(cut))
        S.mask_text = (f"specular spot, ellipse centre phi_f = {info['phi0']:.6f} "
                       f"deg, alpha_f = {info['alpha0']:.6f} deg, half-axes "
                       f"{info['half_axis_phi']:.6f} x {info['half_axis_alpha']:.6f} "
                       f"deg ({n_sigma:g} sigma); {info['n_masked']} pixels set to nan")
        S.data, S.mask_kind = masked, "spot"
    else:
        raise ValueError(f"unknown mask kind {kind!r}")
    S.version += 1
    report["version"] = S.version
    report["mask_text"] = S.mask_text
    return _pack(report)


# --- plots ----------------------------------------------------------------

MAPS = {
    # kind: (frame, x column, y column)
    "raw": ("raw", "x", "y"),
    "y_x": ("data", "x", "y"),
    "af_pf": ("data", "phi_f", "alpha_f"),
    "qz_qy": ("data", "qy", "qz"),
}

PROFILES = {
    # kind: (function, label column)
    "phi_f": (gisaxs.column_mean_profile, "phi_f"),
    "qy": (gisaxs.column_mean_profile, "qy"),
    "alpha_f": (gisaxs.row_profile, "alpha_f"),
    "qz": (gisaxs.row_profile, "qz"),
}


def _map_arrays(kind, q_unit):
    frame, xc, yc = MAPS[kind]
    if frame == "raw":
        _require("raw")
        df = S.raw
    else:
        _require("cut")
        df = S.data
    factor = gisaxs.q_unit_info(q_unit)[0] if xc in gisaxs.Q_COLUMNS else 1.0
    X = mask.as_image(df, xc) * factor
    Y = mask.as_image(df, yc) * factor
    return df, X, Y, df["intensity"].to_numpy()


def cmd_map_data(kind, q_unit="nm-1"):
    df, X, Y, counts = _map_arrays(kind, q_unit)
    dx, dy = _pitch(X, Y)
    finite = np.isfinite(counts)
    return _pack(
        {"kind": kind, "n": len(df), "nx": df.attrs["nx"], "ny": df.attrs["ny"],
         "vmax": float(np.nanmax(counts)) if finite.any() else None,
         "n_nan": int((~finite).sum()), "n_zero": int((counts == 0).sum()),
         "dx": dx, "dy": dy, "x_range": _range(X.ravel()),
         "y_range": _range(Y.ravel()), "q_unit": q_unit,
         "version": S.version if kind != "raw" else 0},
        arrays={"x": _f32(X.ravel()), "y": _f32(Y.ravel()), "c": _f32(counts)})


def _profile_series(kind, q_unit, smooth=gisaxs.DEFAULT_SMOOTHING):
    """(x, measured, smoothed) mean counts of one profile; smoothed is a
    running mean of `smooth` points (1 = as measured)."""
    _require("cut")
    func, label = PROFILES[kind]
    measured = func(S.data, label)
    try:
        smoothed = gisaxs.smooth_profile(measured, smooth)
    except ValueError as err:
        raise InputError(str(err)) from None
    x = measured.index.to_numpy(dtype=float)
    if label in gisaxs.Q_COLUMNS:
        x = x * gisaxs.q_unit_info(q_unit)[0]
    return x, measured.to_numpy(dtype=float), smoothed.to_numpy(dtype=float)


def cmd_profile_data(kind, q_unit="nm-1", smooth=gisaxs.DEFAULT_SMOOTHING):
    x, raw, y = _profile_series(kind, q_unit, smooth)
    return _pack({"kind": kind, "n": len(x), "n_gaps": int(np.isnan(y).sum()),
                  "smooth": int(smooth), "q_unit": q_unit, "version": S.version},
                 arrays={"x": np.ascontiguousarray(x), "y": np.ascontiguousarray(y),
                         "raw": np.ascontiguousarray(raw)})


def cmd_export_figure(kind, fmt="png", log=True, cmap="viridis", vmin=None,
                      vmax=None, zero_colour="white", preset="publication",
                      columns=1, q_unit="nm-1", profile=False,
                      smooth=gisaxs.DEFAULT_SMOOTHING, xlim=None, ylim=None,
                      marks=None):
    try:
        data, t0 = _draw_figure(kind, fmt, log, cmap, vmin, vmax, zero_colour,
                                preset, columns, q_unit, profile, smooth,
                                xlim, ylim, marks)
    except figures.FigureError as err:
        raise InputError(str(err)) from None
    mime = "application/pdf" if fmt == "pdf" else "image/png"
    return _pack({"mime": mime, "bytes": len(data), "seconds": time.time() - t0},
                 arrays={"file": np.frombuffer(data, dtype=np.uint8)})


_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def _marks(kind, marks, q_unit):
    """The profile labels sent by the page, [{"x", "text", "style"}], checked;
    text is the number as typed, shown as it is on the label."""
    out = []
    for i, m in enumerate(marks or []):
        try:
            x = float(m["x"])
            style = int(m.get("style", i))
        except (KeyError, TypeError, ValueError):
            raise InputError("every label needs a number") from None
        if not np.isfinite(x):
            raise InputError("every label needs a finite number")
        text = str(m.get("text", "")).strip()
        if not _NUMBER.fullmatch(text):
            text = f"{x:g}"
        out.append({"x": x, "style": style,
                    "label": figures.mark_label(kind, text, q_unit)})
    return out


def _draw_figure(kind, fmt, log, cmap, vmin, vmax, zero_colour, preset, columns,
                 q_unit, profile, smooth, xlim, ylim, marks=None):
    t0 = time.time()
    if profile:
        progress(f"Drawing the {fmt.upper()} figure…")
        x, raw, smoothed = _profile_series(kind, q_unit, smooth)
        data = figures.profile(x, raw, figures.axis_label(kind, q_unit),
                               smoothed=smoothed if int(smooth) > 1 else None,
                               log=log, preset=preset, columns=int(columns),
                               fmt=fmt, xlim=xlim, marks=_marks(kind, marks, q_unit))
    else:
        df, X, Y, counts = _map_arrays(kind, q_unit)
        progress(f"Drawing {len(counts):,} pixels into the {fmt.upper()} "
                 f"figure…")
        _, xc, yc = MAPS[kind]
        data = figures.intensity_map(
            X.ravel(), Y.ravel(), counts,
            figures.axis_label(xc, q_unit), figures.axis_label(yc, q_unit),
            log=log, cmap=cmap, vmin=vmin, vmax=vmax, zero_colour=zero_colour,
            preset=preset, columns=int(columns), fmt=fmt, pixel=_pitch(X, Y),
            xlim=xlim, ylim=ylim)
    return data, t0


COMMANDS = {
    "init": cmd_init,
    "load_pro": cmd_load_pro,
    "load_dat": cmd_load_dat,
    "check_alpha": cmd_check_alpha,
    "load_ref": cmd_load_ref,
    "clear_ref": cmd_clear_ref,
    "convert": cmd_convert,
    "table_head": cmd_table_head,
    "export_table": cmd_export_table,
    "apply_mask": cmd_apply_mask,
    "map_data": cmd_map_data,
    "profile_data": cmd_profile_data,
    "export_figure": cmd_export_figure,
}


def dispatch(command, args_json="{}"):
    """Run one command; never raises, so every error reaches the page."""
    try:
        func = COMMANDS[command]
        return func(**json.loads(args_json or "{}"))
    except InputError as err:
        return _pack({"error": str(err), "user_error": True})
    except MemoryError:
        return _pack({"error": "the browser ran out of memory for Python. "
                               "Close other tabs or reload the page and try "
                               "again."})
    except Exception as err:  # noqa: BLE001 - reported to the page
        return _pack({"error": f"{type(err).__name__}: {err}",
                      "traceback": traceback.format_exc()})
