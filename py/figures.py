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
# Every downloaded figure has the proportions of an A4 sheet in landscape,
# 297 x 210 mm: height = width / sqrt(2).  The width is the journal column
# (publication) or 8 in (presentation).
A4_RATIO = 210.0 / 297.0
PRESENTATION_SIZE = (8.0, 8.0 * A4_RATIO)


class FigureError(ValueError):
    """A figure cannot be drawn with the options given (worded for the user)."""

# Marker side as a multiple of the pixel pitch, so neighbouring squares
# touch without leaving hairline gaps in the raster.
MARKER_OVERLAP = 1.25

# 1D profiles: the measured curve faded underneath, the smoothed one on top,
# with line widths relative to the preset's lines.linewidth.
PROFILE_MEASURED = {"color": "0.78", "width": 0.75}
PROFILE_SMOOTHED = {"color": "0.15", "width": 1.5}

# Labelled vertical lines on the 1D profiles ("Add label"): colour and line
# style in the order the labels are added, the same on screen (Plotly dash
# names; js/app.js gets them from screen_colours()).
MARK_STYLES = [
    {"color": "#b2182b", "dash": "dot"},
    {"color": "#4682b4", "dash": "dash"},
    {"color": "#1b7837", "dash": "dashdot"},
    {"color": "#7b3294", "dash": "longdash"},
    {"color": "#d95f02", "dash": "dot"},
    {"color": "#404040", "dash": "dash"},
]
MPL_DASH = {"dot": (0, (1, 1.5)), "dash": (0, (4, 2)),
            "dashdot": (0, (4, 1.5, 1, 1.5)), "longdash": (0, (8, 2.5))}
# With labels, the top of the plot is kept free for them: the data use the
# bottom (1 - MARK_HEADROOM) of the y-axis (in log units on a log axis).
MARK_HEADROOM = 0.14
MINUS = "−"


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


def mark_label(name, text, q_unit="nm-1"):
    """The text of a profile label, e.g. "$\alpha_f$ = 0.280°".

    text is the number as typed; it is kept as it is, so the label shows the
    digits that were entered.  Numbers stay outside $...$ (plotstyle gotcha).
    """
    symbol = {"x": r"$x$", "y": r"$y$", "phi_f": r"$\varphi_f$",
              "alpha_f": r"$\alpha_f$", "qy": r"$q_y$", "qz": r"$q_z$"}[name]
    unit = {"phi_f": "°", "alpha_f": "°", "x": " mm", "y": " mm"}.get(
        name, " " + q_unit_tex(q_unit))
    return f"{symbol} = {text.replace('-', MINUS)}{unit}"


def mark_sides(xs, lo, hi):
    """For each label line at xs, which side its text goes: towards the wider
    gap to the next line or to the axis edge, so neighbouring labels part.
    js/app.js (markSides) does the same for the screen."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sides = [None] * len(xs)
    for k, i in enumerate(order):
        left = xs[i] - (xs[order[k - 1]] if k > 0 else lo)
        right = (xs[order[k + 1]] if k + 1 < len(order) else hi) - xs[i]
        sides[i] = "right" if right >= left else "left"
    return sides


def _draw_marks(ax, marks, lo, hi, log, width):
    """Vertical lines with their labels at the top, and head room for them."""
    shown = [m for m in marks if lo <= m["x"] <= hi]
    if not shown:
        return
    sides = mark_sides([m["x"] for m in shown], lo, hi)
    for m, side in zip(shown, sides):
        style = MARK_STYLES[int(m["style"]) % len(MARK_STYLES)]
        ax.axvline(m["x"], color=style["color"], linestyle=MPL_DASH[style["dash"]],
                   linewidth=1.3 * width, zorder=3)
        pad = 3.0
        ax.annotate(m["label"], xy=(m["x"], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(pad if side == "right" else -pad, -pad),
                    textcoords="offset points", ha="left" if side == "right" else "right",
                    va="top", color=style["color"],
                    fontsize=mpl.rcParams["axes.labelsize"], zorder=4)
    y0, y1 = ax.get_ylim()
    if log:
        a, b = np.log10(y0), np.log10(y1)
        ax.set_ylim(y0, 10.0 ** (a + (b - a) / (1.0 - MARK_HEADROOM)))
    else:
        ax.set_ylim(y0, y0 + (y1 - y0) / (1.0 - MARK_HEADROOM))


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
    # Save the whole figure, not a box cropped to its contents, so the file
    # keeps the A4 proportions; the constrained layout fits the labels in.
    mpl.rcParams["savefig.bbox"] = "standard"


def figure_size(preset="publication", columns=1):
    """(width, height) in inches, in A4 proportions."""
    if preset == "presentation":
        return PRESENTATION_SIZE
    return plotstyle.figure_size(columns=columns, ratio=A4_RATIO)


def _centre_on_page(fig, ax):
    """Move the axes and their colour bar so the drawing sits in the middle of
    the page: an equal-aspect map narrower (or flatter) than the A4 page would
    otherwise hug the colour bar and leave the spare room on one side."""
    fig.draw_without_rendering()
    box = fig.get_tightbbox()                     # everything drawn, in inches
    w, h = fig.get_figwidth(), fig.get_figheight()
    dx = ((w - box.x1) - box.x0) / 2.0 / w
    dy = ((h - box.y1) - box.y0) / 2.0 / h
    for a in fig.axes:
        pos = a.get_position(original=True)
        a.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height], which="both")
    ax.apply_aspect()


def _to_bytes(fig, fmt):
    if fmt not in ("png", "pdf"):
        raise ValueError("format must be png or pdf")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt)
    plt.close(fig)
    return buf.getvalue()


def axis_limits(limits, data_lo, data_hi, name="x"):
    """(lo, hi) from [lo or None, hi or None]; a missing end is the data's."""
    lo, hi = limits if limits is not None else (None, None)
    lo = float(data_lo) if lo is None else float(lo)
    hi = float(data_hi) if hi is None else float(hi)
    if not lo < hi:
        raise FigureError(f"{name}-axis: the lower limit ({lo:g}) must be smaller "
                         f"than the upper limit ({hi:g})")
    return lo, hi


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
                  equal_aspect=True, xlim=None, ylim=None):
    """A scatter plot of every pixel coloured by counts; returns PNG/PDF bytes.

    pixel      : (dx, dy) pixel pitch in data units, used to size the markers
                 so they tile the plane.  The scatter layer is rasterized; the
                 axes, labels and colour bar stay vector.
    xlim, ylim : [lo, hi] axis limits, either end None for the data's own.
                 The default colour range still comes from the whole frame,
                 so a zoomed figure keeps the colours of the full one.
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

        finite = c[np.isfinite(c)]
        dx, dy = pixel if pixel is not None else (0.0, 0.0)
        x_lo, x_hi = axis_limits(xlim, np.min(x) - dx / 2, np.max(x) + dx / 2, "x")
        y_lo, y_hi = axis_limits(ylim, np.min(y) - dy / 2, np.max(y) + dy / 2, "y")
        # Draw only the pixels in view (with one pixel of margin at the edges).
        view = ((x >= x_lo - dx) & (x <= x_hi + dx)
                & (y >= y_lo - dy) & (y <= y_hi + dy))
        if not view.any():
            raise FigureError("no pixels lie inside the chosen axis limits")
        x, y, c, shown = x[view], y[view], c[view], shown[view]

        fig, ax = plt.subplots(figsize=figure_size(preset, columns), layout="constrained")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        extend = "min"
        if finite.size and finite.max() > vmax:
            extend = "both"
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colours)
        fig.colorbar(mappable, ax=ax, extend=extend, label=axis_label("counts"),
                     fraction=0.046, pad=0.03)

        # Size the squares from the final axes box (after the layout, the
        # colour bar and the equal aspect have taken their share of the
        # figure): lay the figure out once and keep that layout.
        fig.draw_without_rendering()
        fig.set_layout_engine("none")
        ax.apply_aspect()
        if equal_aspect:
            _centre_on_page(fig, ax)
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


def profile(x, y, xlabel, *, smoothed=None, log=True, preset="publication",
            columns=1, fmt="png", ylabel=None, xlim=None, marks=None):
    """Mean counts per pixel against one coordinate; NaN draws as a gap.

    smoothed : the smoothed profile; the measured one is then drawn faded
               underneath it.  None draws the measured profile alone.
    xlim     : [lo, hi] x-axis limits (either end None for the data's own).
               Only the points inside are drawn, so the y-axis fits them.
    marks    : labelled vertical lines, [{"x", "label", "style"}] (label from
               mark_label(), style an index into MARK_STYLES); those outside
               the x-axis limits are left out.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = axis_limits(xlim, np.nanmin(x), np.nanmax(x), "x")
    view = (x >= lo) & (x <= hi)
    if view.sum() < 2:
        raise FigureError("fewer than 2 points lie inside the x-axis limits")
    with mpl.rc_context():
        apply_style(preset)
        fig, ax = plt.subplots(figsize=figure_size(preset, columns), layout="constrained")
        width = mpl.rcParams["lines.linewidth"]
        if smoothed is None:
            ax.plot(x[view], y[view], "-", color=PROFILE_SMOOTHED["color"],
                    linewidth=width)
        else:
            smoothed = np.asarray(smoothed, dtype=float)
            ax.plot(x[view], y[view], "-", color=PROFILE_MEASURED["color"],
                    linewidth=PROFILE_MEASURED["width"] * width, zorder=1)
            ax.plot(x[view], smoothed[view], "-", color=PROFILE_SMOOTHED["color"],
                    linewidth=PROFILE_SMOOTHED["width"] * width, zorder=2)
        if log:
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        if marks:
            _draw_marks(ax, marks, lo, hi, log, width)
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
    profile = {"measured": to_hex(PROFILE_MEASURED["color"]),
               "smoothed": to_hex(PROFILE_SMOOTHED["color"])}
    return {"scales": scales, "under": under, "bad": bad, "profile": profile,
            "marks": [dict(m) for m in MARK_STYLES], "mark_headroom": MARK_HEADROOM}
