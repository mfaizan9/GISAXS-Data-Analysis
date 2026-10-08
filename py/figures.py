"""PNG / PDF figures of the maps and profiles, drawn with matplotlib + plotstyle.

Two presets:

    publication   plotstyle defaults, 1 or 2 journal columns wide
    presentation  font_size=16, linewidth=1.5, lines 2 pt, 8 x 6 in

Every figure is drawn inside mpl.rc_context(), so a preset never leaks into
the next figure.  LaTeX is not available in the browser, so engine="stix" is
used; Times New Roman is not installed there either, and plotstyle falls back
to STIXGeneral, which is intended.
"""

import io
import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, to_hex  # noqa: E402

import plotstyle  # noqa: E402

# "Times New Roman not found" is expected (see above); do not log it per figure.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Projector-friendly and colour-blind-safe first; no red-green maps.
COLORMAPS = ["viridis", "cividis", "inferno", "magma", "turbo"]
# Measured zeros (the "under" colour); the first is the default, so the maps
# sit on a white background.
ZERO_COLOURS = {"white": "white", "black": "black"}
PRESETS = ("publication", "presentation")
PRESENTATION_SIZE = (8.0, 6.0)

# Marker side as a multiple of the pixel pitch, so neighbouring squares
# touch without leaving hairline gaps in the raster.
MARKER_OVERLAP = 1.25


def q_unit_tex(q_unit):
    # \mathrm keeps the 1 upright: mathtext.default is "it" (plotstyle gotcha).
    return (r"nm$^{-\mathrm{1}}$" if q_unit == "nm-1"
            else "Å" + r"$^{-\mathrm{1}}$")


def axis_label(name, q_unit="nm-1"):
    """Axis label with units; numbers kept outside $...$ (plotstyle gotcha)."""
    unit = q_unit_tex(q_unit)
    labels = {
        "x": r"$x$ (mm)",
        "y": r"$y$ (mm)",
        "phi_f": r"$\varphi_f$ (deg)",
        "alpha_f": r"$\alpha_f$ (deg)",
        "qy": rf"$q_y$ ({unit})",
        "qz": rf"$q_z$ ({unit})",
        "counts": "Counts",
        "mean": "Mean counts per pixel",
    }
    return labels[name]


def apply_style(preset="publication"):
    """Set rcParams for a preset (call inside mpl.rc_context())."""
    if preset == "publication":
        plotstyle.use_publication_style(engine="stix")
    elif preset == "presentation":
        plotstyle.use_publication_style(font_size=16, linewidth=1.5,
                                        engine="stix")
        mpl.rcParams.update({
            "lines.linewidth": 2.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "xtick.minor.size": 3.5,
            "ytick.minor.size": 3.5,
        })
    else:
        raise ValueError(f"preset must be one of {PRESETS}")


def figure_size(preset="publication", columns=1, ratio=0.75):
    if preset == "presentation":
        return PRESENTATION_SIZE
    return plotstyle.figure_size(columns=columns, ratio=ratio)


def _to_bytes(fig, fmt):
    if fmt not in ("png", "pdf"):
        raise ValueError("format must be png or pdf")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt)
    plt.close(fig)
    return buf.getvalue()


def colour_limits(counts, vmin=None, vmax=None):
    """Default range: vmin = 1 count, vmax = max of the unmasked data."""
    counts = np.asarray(counts, dtype=float)
    if vmin is None:
        vmin = 1.0
    if vmax is None:
        finite = counts[np.isfinite(counts)]
        vmax = float(finite.max()) if finite.size else 10.0
    if vmax <= vmin:
        vmax = vmin * 10.0
    return float(vmin), float(vmax)


def intensity_map(x, y, counts, xlabel, ylabel, *, log=True, cmap="viridis",
                  vmin=None, vmax=None, zero_colour="white",
                  preset="publication", columns=1, fmt="png", pixel=None,
                  equal_aspect=True):
    """A scatter plot of every pixel coloured by counts; returns PNG/PDF bytes.

    pixel : (dx, dy) pixel pitch in data units, used to size the markers so
            they tile the plane.  The scatter layer is rasterized; the axes,
            labels and colour bar stay vector.
    """
    if cmap not in COLORMAPS:
        raise ValueError(f"colormap must be one of {COLORMAPS}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(counts, dtype=float)
    vmin, vmax = colour_limits(c, vmin, vmax)
    if log and vmin <= 0:
        raise ValueError("on a log scale vmin must be > 0")

    with mpl.rc_context():
        apply_style(preset)
        colours = plotstyle.intensity_cmap(cmap,
                                           zero_colour=ZERO_COLOURS[zero_colour])
        if log:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            # A log scale has no place for 0: nudge measured zeros just under
            # the scale so they take the "under" colour, not the "bad" one.
            shown = np.where(c <= 0, 0.5 * vmin, c)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            shown = c

        fig, ax = plt.subplots(figsize=figure_size(preset, columns, ratio=0.85))
        dx, dy = pixel if pixel is not None else (0.0, 0.0)
        ax.set_xlim(np.min(x) - dx / 2, np.max(x) + dx / 2)
        ax.set_ylim(np.min(y) - dy / 2, np.max(y) + dy / 2)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        finite = c[np.isfinite(c)]
        extend = "min"
        if finite.size and finite.max() > vmax:
            extend = "both"
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colours)
        fig.colorbar(mappable, ax=ax, extend=extend, label=axis_label("counts"),
                     fraction=0.046, pad=0.03)

        # Size the squares from the final axes box (after the colour bar and
        # the equal aspect have taken their share of the figure).
        ax.apply_aspect()
        box = ax.get_position()
        w_pt = box.width * fig.get_figwidth() * 72.0
        h_pt = box.height * fig.get_figheight() * 72.0
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        side = 0.5
        if dx > 0 and dy > 0:
            side = MARKER_OVERLAP * max(dx / (x1 - x0) * w_pt,
                                        dy / (y1 - y0) * h_pt)

        # Empty and masked pixels first, brightest last, so a bright pixel is
        # never hidden under a neighbour.
        order = np.argsort(np.where(np.isfinite(c), c, -np.inf), kind="stable")
        ax.scatter(x[order], y[order], c=shown[order], s=side ** 2, marker="s",
                   cmap=colours, norm=norm, linewidths=0, edgecolors="none",
                   plotnonfinite=True, rasterized=True)
        return _to_bytes(fig, fmt)


def profile(x, y, xlabel, *, log=True, preset="publication", columns=1,
            fmt="png", ylabel=None):
    """Mean counts per pixel against one coordinate; NaN draws as a gap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with mpl.rc_context():
        apply_style(preset)
        fig, ax = plt.subplots(figsize=figure_size(preset, columns, ratio=0.75))
        ax.plot(x, y, "-")
        if log:
            ax.set_yscale("log")
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or axis_label("mean"))
        return _to_bytes(fig, fmt)


def screen_colours(n=64):
    """Plotly colour scales sampled from the same matplotlib colormaps, plus
    the "under" (zero) and "bad" (NaN) colours of plotstyle.intensity_cmap()."""
    scales = {}
    for name in COLORMAPS:
        cm = plt.get_cmap(name)
        scales[name] = [[i / (n - 1), to_hex(cm(i / (n - 1)))] for i in range(n)]
    under = {}
    for key, colour in ZERO_COLOURS.items():
        under[key] = to_hex(plotstyle.intensity_cmap(
            "viridis", zero_colour=colour).get_under())
    bad = to_hex(plotstyle.intensity_cmap("viridis").get_bad())
    return {"scales": scales, "under": under, "bad": bad}
