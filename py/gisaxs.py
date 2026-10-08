"""Convert GISAXS detector pixels to exit angles, final wavevector and q.

Every function here is plain numpy/pandas, so the same code runs in the browser
(Pyodide) and under normal CPython for the tests.

Frames (see assets/geometry.svg):

    lab frame     X, Y, Z   X along the incident beam, Z vertical up, Y = Z x X
    sample frame  u, v, n   u in the surface along the beam, v = Y, n = normal
    detector      x, y      mm on the detector, measured from the direct beam;
                            x along Y, y along Z (up)

The sample is tilted by alpha_i about Y.  A pixel at (x, y) sits at
r = L X + x Y + y Z, whose sample-frame components are

    a = r.u = L cos(alpha_i) + y sin(alpha_i)
    b = r.v = x
    c = r.n = y cos(alpha_i) - L sin(alpha_i)

and everything else follows from a/R, b/R, c/R with R = |r|.  All results
(phi_f, alpha_f, k_f, q) are in the sample frame.  Angles are returned in
degrees; k and q in nm^-1 (the wavelength is in nm).
"""

import datetime
import re
import warnings
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mask import as_image

# Internal column names.  x, y, intensity, variance are what mask.py expects.
DATA_COLUMNS = ["x", "y", "intensity", "variance"]
ANGLE_COLUMNS = ["phi_f", "alpha_f"]
KF_COLUMNS = ["kfx", "kfy", "kfz"]
Q_COLUMNS = ["qx", "qy", "qz"]

# Columns that carry a wavevector and scale with the q unit.
WAVEVECTOR_COLUMNS = KF_COLUMNS + Q_COLUMNS

# q unit -> (factor applied to values in nm^-1, suffix used in column names,
# label used in the UI).
Q_UNITS = {
    "nm-1": (1.0, "nm-1", "nm⁻¹"),
    "A-1": (0.1, "A-1", "Å⁻¹"),
}

# A counting detector writes -1 for a dead pixel.
DEAD_PIXEL = -1

# 1D profiles are smoothed with a centred running mean of this many points;
# the UI offers SMOOTHING_WINDOWS (1 = no smoothing).
DEFAULT_SMOOTHING = 7
SMOOTHING_WINDOWS = (7, 5, 1)


class InputError(ValueError):
    """A problem with the user's files, worded for the user."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """What the conversion needs: distance, incidence angle and wavelength."""

    L: float            # sample-detector distance [mm]
    alpha_i: float      # incidence angle [deg]
    wavelength: float   # [nm]

    @property
    def k(self):
        """Wavenumber 2 pi / lambda [nm^-1]."""
        return 2.0 * np.pi / self.wavelength

    @classmethod
    def from_pro(cls, pro):
        """Build from read_pro.read_pro(); only the length of N is used."""
        return cls(L=float(np.linalg.norm(pro["N"])),
                   alpha_i=float(pro["alpha_i"]),
                   wavelength=float(pro["wavelength"]))


def instrument_summary(pro):
    """The .pro values plus the quantities derived from them (section 2.1)."""
    geom = Geometry.from_pro(pro)
    for key in ("nx", "ny", "width", "height", "wavelength", "alpha_i"):
        if not np.isfinite(pro[key]) or pro[key] <= 0:
            raise InputError(f"the .pro file gives {key} = {pro[key]}, "
                             f"which is not a positive number")
    if geom.L <= 0:
        raise InputError("the .pro detector normal vector N has zero length, "
                         "so the sample-detector distance is unknown")
    pitch_x = pro["width"] / (pro["nx"] - 1)
    pitch_y = pro["height"] / (pro["ny"] - 1)
    return {
        "nx": int(pro["nx"]),
        "ny": int(pro["ny"]),
        "width": float(pro["width"]),
        "height": float(pro["height"]),
        "N": [float(v) for v in pro["N"]],
        "u0": float(pro["u0"]),
        "v0": float(pro["v0"]),
        "wavelength": float(pro["wavelength"]),
        "alpha_i": float(pro["alpha_i"]),
        "L": geom.L,
        "pitch_x": float(pitch_x),
        "pitch_y": float(pitch_y),
        "k": geom.k,
        "pixel_mdeg_x": float(np.degrees(pitch_x / geom.L) * 1000.0),
        "pixel_mdeg_y": float(np.degrees(pitch_y / geom.L) * 1000.0),
    }


def sample_frame(x, y, L, alpha_i):
    """Components (a, b, c) of the pixel vector r along u, v, n, and |r|.

    x, y in mm (from the direct beam), L in mm, alpha_i in degrees.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ai = np.radians(alpha_i)
    sin_ai, cos_ai = np.sin(ai), np.cos(ai)
    a = L * cos_ai + y * sin_ai
    b = x
    c = y * cos_ai - L * sin_ai
    R = np.sqrt(L * L + x * x + y * y)
    return a, b, c, R


def exit_angles(x, y, geom):
    """phi_f, alpha_f [deg] of each pixel (section 3.2)."""
    a, b, c, _ = sample_frame(x, y, geom.L, geom.alpha_i)
    phi_f = np.degrees(np.arctan2(b, a))
    alpha_f = np.degrees(np.arctan2(c, np.hypot(a, b)))
    return phi_f, alpha_f


def final_wavevector(x, y, geom):
    """k_f = (k/R)(a, b, c) in the sample frame [nm^-1] (section 3.3)."""
    a, b, c, R = sample_frame(x, y, geom.L, geom.alpha_i)
    s = geom.k / R
    return s * a, s * b, s * c


def incident_wavevector(geom):
    """k_i = k (cos alpha_i, 0, -sin alpha_i) in the sample frame (3.4)."""
    ai = np.radians(geom.alpha_i)
    return np.array([geom.k * np.cos(ai), 0.0, -geom.k * np.sin(ai)])


def scattering_vector(x, y, geom):
    """q = k_f - k_i in the sample frame [nm^-1] (section 3.5)."""
    kfx, kfy, kfz = final_wavevector(x, y, geom)
    kix, kiy, kiz = incident_wavevector(geom)
    return kfx - kix, kfy - kiy, kfz - kiz


def pixel(x, y, geom):
    """Every converted quantity for one or more pixels, as a dict (for tests)."""
    phi_f, alpha_f = exit_angles(x, y, geom)
    kfx, kfy, kfz = final_wavevector(x, y, geom)
    qx, qy, qz = scattering_vector(x, y, geom)
    return dict(phi_f=phi_f, alpha_f=alpha_f, kfx=kfx, kfy=kfy, kfz=kfz,
                qx=qx, qy=qy, qz=qz)


# --- the three workflow steps, each adding its columns to the frame --------

def add_exit_angles(df, geom):
    """Step 5: add phi_f, alpha_f [deg]."""
    df["phi_f"], df["alpha_f"] = exit_angles(df["x"].to_numpy(),
                                             df["y"].to_numpy(), geom)
    return df


def add_final_wavevector(df, geom):
    """Step 6: add kfx, kfy, kfz [nm^-1]."""
    df["kfx"], df["kfy"], df["kfz"] = final_wavevector(df["x"].to_numpy(),
                                                       df["y"].to_numpy(), geom)
    return df


def add_scattering_vector(df, geom):
    """Step 7: add qx, qy, qz [nm^-1]."""
    df["qx"], df["qy"], df["qz"] = scattering_vector(df["x"].to_numpy(),
                                                     df["y"].to_numpy(), geom)
    return df


def convert(df, geom):
    """Steps 5-7 on a copy of df."""
    out = df.copy()
    add_exit_angles(out, geom)
    add_final_wavevector(out, geom)
    add_scattering_vector(out, geom)
    return out


# ---------------------------------------------------------------------------
# Horizon cut (step 8)
# ---------------------------------------------------------------------------

def row_alpha_f(df, geom):
    """alpha_f at x = 0 for every detector row [deg]: atan2(c, a) with b = 0."""
    y_row = as_image(df, "y").mean(axis=1)
    a, _, c, _ = sample_frame(0.0, y_row, geom.L, geom.alpha_i)
    return np.degrees(np.arctan2(c, a))


def horizon_y(geom):
    """Detector height of the sample horizon, y = L tan(alpha_i) [mm]."""
    return geom.L * np.tan(np.radians(geom.alpha_i))


def horizon_cut(df, geom):
    """Drop every whole detector row whose alpha_f at x = 0 is below 0.

    Whole rows are kept or dropped, so the result is still a rectangular image;
    df.attrs["nx"] and ["ny"] are set for mask.as_image().
    """
    ny, nx = as_image(df, "y").shape
    keep_row = row_alpha_f(df, geom) >= 0.0
    if not keep_row.any():
        raise InputError("every detector row lies below the sample horizon "
                         "(alpha_f < 0); check alpha_i and the y coordinates")
    out = df.loc[np.repeat(keep_row, nx)].reset_index(drop=True)
    out.attrs = dict(df.attrs)
    out.attrs["nx"] = int(nx)
    out.attrs["ny"] = int(keep_row.sum())
    out.attrs["rows_dropped"] = int(ny - keep_row.sum())
    return out


# ---------------------------------------------------------------------------
# 1D profiles (section 7)
# ---------------------------------------------------------------------------

def row_profile(df, label="alpha_f"):
    """Mean counts per pixel in each detector row, against the row mean of label.

    mask.py's row_profile, generalised to any label column (alpha_f, qz, ...).
    """
    intensity = as_image(df, "intensity")
    coord = as_image(df, label)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A fully masked row/column is NaN (a gap in the plot), not a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(intensity, axis=1)
    return pd.Series(means, index=coord.mean(axis=1))


def column_mean_profile(df, label="phi_f"):
    """Mean counts per pixel in each detector column, against the column mean.

    Named so it does not shadow mask.py's sum-based column_profile, which
    mask.find_rod() relies on.
    """
    intensity = as_image(df, "intensity")
    coord = as_image(df, label)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A fully masked row/column is NaN (a gap in the plot), not a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(intensity, axis=0)
    return pd.Series(means, index=coord.mean(axis=0))


def smooth_profile(profile, window=DEFAULT_SMOOTHING):
    """Centred running mean of `window` neighbouring points of a profile.

        smooth = profile.rolling(window, center=True, min_periods=1).mean()

    min_periods=1 keeps the points at both ends (their windows are shorter).
    A point that was NaN before (a fully masked row or column) is NaN again
    afterwards, so a masked region stays a gap instead of being filled in from
    its edges.  window = 1 returns the profile unchanged.
    """
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("the smoothing window must be an odd number of points")
    if window == 1:
        return profile.copy()
    smooth = profile.rolling(window, center=True, min_periods=1).mean()
    return smooth.where(profile.notna())


# ---------------------------------------------------------------------------
# Reading the .dat file
# ---------------------------------------------------------------------------

def read_header(path):
    """The '#' comment lines at the top of a .dat file and the grid they state.

    Returns dict(rows=..., columns=..., lines=[...]); rows/columns are None if
    the header does not give them.
    """
    lines = []
    found = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            lines.append(line.rstrip("\r\n"))
            key, sep, value = line[1:].partition(":")
            if not sep:
                continue
            match = re.search(r"\d[\d,]*", value)
            if match:
                found[key.strip().lower()] = int(match.group().replace(",", ""))
    return {"rows": found.get("rows"), "columns": found.get("columns"),
            "lines": lines}


def _grid_is_row_major(x, y, nx, ny):
    """True if y is constant along each row, x rises along a row, y rises
    from row to row: bottom detector row first, x ascending within a row."""
    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    pitch = (np.nanmax(y) - np.nanmin(y)) / max(ny - 1, 1)
    row_flat = np.all(np.abs(Y - Y[:, :1]) <= 1e-3 * pitch)
    x_rises = nx == 1 or np.all(np.diff(X, axis=1) > 0)
    y_rises = ny == 1 or np.all(np.diff(Y[:, 0]) > 0)
    return bool(row_flat and x_rises and y_rises)


def order_grid(df, nx, ny):
    """Put the pixels in row-major order (y ascending, x ascending in a row).

    Returns (df, description).  The expected order is only verified; any other
    regular order (e.g. column-major or y descending) is sorted into it.
    """
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    if _grid_is_row_major(x, y, nx, ny):
        return df, "row-major: y ascending (bottom row first), x ascending within a row"

    # Sort on integer grid indices so tiny float noise in x or y cannot
    # interleave rows.
    def index(v, n):
        span = np.nanmax(v) - np.nanmin(v)
        step = span / max(n - 1, 1) if span > 0 else 1.0
        return np.rint((v - np.nanmin(v)) / step).astype(np.int64)

    order = np.lexsort((index(x, nx), index(y, ny)))
    sorted_df = df.iloc[order].reset_index(drop=True)
    sorted_df.attrs = dict(df.attrs)
    if not _grid_is_row_major(sorted_df["x"].to_numpy(),
                              sorted_df["y"].to_numpy(), nx, ny):
        raise InputError(
            f"the (x, y) positions in the .dat file do not form a regular "
            f"{ny} x {nx} grid, so the pixels cannot be arranged into a "
            f"detector image")
    return sorted_df, ("not row-major in the file; re-sorted to y ascending "
                       "(bottom row first), x ascending within a row")


def read_dat(path, nx=None, ny=None, progress=None):
    """Read a .dat detector file into a DataFrame x, y, intensity[, variance].

    nx, ny : the grid the .pro file declares; if given, the header must match.

    Counts of -1 (dead pixels) become NaN, and so does their variance.  The
    detector grid is kept in df.attrs["nx"], ["ny"].
    """
    header = read_header(path)
    rows, cols = header["rows"], header["columns"]
    if rows is None or cols is None:
        raise InputError("the .dat header has no '# Rows:' and '# Columns:' "
                         "lines, so the detector grid is unknown. Is this a "
                         "BornAgain 2D dataset?")
    if progress:
        progress(f"Parsing {rows * cols:,} pixels…")
    try:
        df = pd.read_csv(path, comment="#", header=None, dtype=np.float64,
                         skipinitialspace=True)
    except (ValueError, pd.errors.ParserError) as err:
        raise InputError(f"the .dat file could not be read as comma-separated "
                         f"numbers ({err})") from None
    except pd.errors.EmptyDataError:
        raise InputError("the .dat file contains no data lines") from None

    if df.shape[1] not in (3, 4):
        raise InputError(f"the .dat file has {df.shape[1]} columns; expected "
                         f"3 (x, y, counts) or 4 (x, y, counts, variance)")
    df.columns = DATA_COLUMNS[:df.shape[1]]

    if len(df) != rows * cols:
        raise InputError(f"the header says {rows:,} rows × {cols:,} columns "
                         f"= {rows * cols:,} pixels, but the file has "
                         f"{len(df):,} data lines")
    if nx is not None and ny is not None and (cols != nx or rows != ny):
        raise InputError(f"the .dat grid is {rows:,} rows × {cols:,} columns, "
                         f"but the .pro detector is {ny:,} × {nx:,} "
                         f"(ny × nx). Are the two files from the same "
                         f"measurement?")
    if not (np.isfinite(df["x"]).all() and np.isfinite(df["y"]).all()):
        raise InputError("some x or y positions in the .dat file are not "
                         "numbers")

    dead = df["intensity"] == DEAD_PIXEL
    blank = [c for c in ("intensity", "variance") if c in df.columns]
    df.loc[dead, blank] = np.nan

    df.attrs["nx"] = int(cols)
    df.attrs["ny"] = int(rows)
    df, order = order_grid(df, cols, rows)
    df.attrs["nx"] = int(cols)
    df.attrs["ny"] = int(rows)
    df.attrs["order"] = order
    df.attrs["dead_minus_one"] = int(dead.sum())
    return df


# ---------------------------------------------------------------------------
# Export (step 9)
# ---------------------------------------------------------------------------

def q_unit_info(q_unit):
    try:
        return Q_UNITS[q_unit]
    except KeyError:
        raise ValueError(f"q unit must be one of {list(Q_UNITS)}") from None


def export_columns(df, q_unit="nm-1"):
    """[(internal name, exported name)] for the columns df actually has."""
    suffix = q_unit_info(q_unit)[1]
    names = {
        "x": "x_mm", "y": "y_mm", "intensity": "counts", "variance": "variance",
        "phi_f": "phi_f_deg", "alpha_f": "alpha_f_deg",
    }
    names.update({c: f"{c}_{suffix}" for c in WAVEVECTOR_COLUMNS})
    order = DATA_COLUMNS + ANGLE_COLUMNS + KF_COLUMNS + Q_COLUMNS
    return [(c, names[c]) for c in order if c in df.columns]


def export_frame(df, q_unit="nm-1"):
    """The frame as it is written to disk: unit-scaled, renamed columns."""
    factor = q_unit_info(q_unit)[0]
    out = pd.DataFrame(index=df.index)
    for src, dst in export_columns(df, q_unit):
        values = df[src].to_numpy()
        out[dst] = values * factor if src in WAVEVECTOR_COLUMNS else values
    return out


def export_header(meta, q_unit="nm-1", when=None):
    """'#' comment lines written before the column names.

    meta: dict with dat_name, pro_name, L, alpha_i, wavelength, k, pitch_x,
    pitch_y, nx, ny, rows_dropped, and optionally mask (a text line).
    """
    factor, _, _ = q_unit_info(q_unit)
    unit_text = "nm^-1" if q_unit == "nm-1" else "A^-1 (1 A^-1 = 10 nm^-1)"
    when = when or datetime.datetime.now().astimezone()
    lines = [
        "# GISAXS pixel table: detector (x, y) -> exit angles, k_f and q",
        f"# Data file: {meta['dat_name']}",
        f"# Instrument file: {meta['pro_name']}",
        f"# Sample-detector distance L = |N| = {meta['L']:.6f} mm",
        f"# Incidence angle alpha_i = {meta['alpha_i']:.9g} deg",
        f"# Wavelength lambda = {meta['wavelength']:.9g} nm",
        f"# k = 2 pi / lambda = {meta['k'] * factor:.9g} {unit_text.split(' ')[0]}",
        f"# Pixel pitch = {meta['pitch_x']:.6g} mm (x), {meta['pitch_y']:.6g} mm (y)",
        f"# Grid: {meta['ny']} rows x {meta['nx']} columns = "
        f"{meta['ny'] * meta['nx']} pixels (row-major, bottom row first)",
        f"# Rows below the sample horizon dropped: {meta['rows_dropped']} "
        f"(whole detector rows with alpha_f < 0 at x = 0)",
        f"# q and k unit: {unit_text}",
        "# Frame: phi_f, alpha_f, k_f and q are in the sample frame "
        "(x along u, y along v, z along the surface normal n)",
        "# x_mm, y_mm: position on the detector measured from the direct beam",
        "# nan counts = dead or masked pixel; 0 = a measured zero",
    ]
    if meta.get("mask"):
        lines.append(f"# Mask: {meta['mask']}")
    lines.append(f"# Written: {when.isoformat(timespec='seconds')}")
    return lines


def iter_csv(df, header_lines, q_unit="nm-1", chunk_rows=50_000,
             progress=None):
    """Yield the CSV text in pieces: header, column names, then data chunks.

    Numbers are written with 8 significant digits ('%.8g'); NaN as 'nan'.
    Reads back with pandas.read_csv(path, comment='#').
    """
    factor = q_unit_info(q_unit)[0]
    columns = export_columns(df, q_unit)
    yield ("\n".join(header_lines) + "\n"
           + ",".join(dst for _, dst in columns) + "\n")
    # Column arrays, not a copy of the whole table: the chunks are built from
    # views, so memory stays flat however large the frame is.
    arrays = [(df[src].to_numpy(), factor if src in WAVEVECTOR_COLUMNS else None)
              for src, _ in columns]
    row_fmt = ",".join(["%.8g"] * len(columns)) + "\n"
    n = len(df)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = np.empty((stop - start, len(columns)))
        for j, (values, scale) in enumerate(arrays):
            block[:, j] = values[start:stop]
            if scale is not None:
                block[:, j] *= scale
        if progress:
            progress(start, n)
        yield (row_fmt * len(block)) % tuple(block.ravel().tolist())
    if progress:
        progress(n, n)


def iter_csv_bytes(df, header_lines, q_unit="nm-1", gzip_level=None,
                   progress=None):
    """iter_csv() encoded as UTF-8, gzip-compressed if gzip_level is given."""
    compressor = None
    if gzip_level is not None:
        # wbits=31: a gzip container (header + CRC), as gzip.open() writes.
        compressor = zlib.compressobj(gzip_level, zlib.DEFLATED, 31)
    for text in iter_csv(df, header_lines, q_unit, progress=progress):
        data = text.encode("utf-8")
        if compressor is None:
            yield data
        else:
            packed = compressor.compress(data)
            if packed:
                yield packed
    if compressor is not None:
        yield compressor.flush()


def write_csv(df, path, header_lines, q_unit="nm-1", gzip_level=None):
    """Write the export to a file (used by the tests and for scripting)."""
    with open(path, "wb") as f:
        for piece in iter_csv_bytes(df, header_lines, q_unit, gzip_level):
            f.write(piece)
