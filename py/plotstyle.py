"""Publication-quality matplotlib defaults, in Times for both text and maths.

Call it once, near the top of a notebook, and every figure made afterwards picks
the settings up:

    import plotstyle
    plotstyle.use_publication_style()

Then size each figure to the column it will be printed in:

    fig, ax = plt.subplots(figsize=plotstyle.figure_size(columns=1))

and save it with fonts embedded and no wasted white space:

    plotstyle.save_figure(fig, "fig_map_000")     # writes .pdf and .png

MATHS IN TIMES.  Text in Times and maths in the matplotlib default would not
match, so the maths font is set too.  There are two ways, and this module offers
both:

  * engine="stix"  (the default).  Maths is drawn by matplotlib itself using
    STIXGeneral, which was designed to sit beside Times: the shapes and the
    stroke weights match closely.  Needs nothing installed, and is fast.

  * engine="latex".  Maths, and all the text, is typeset by a real LaTeX with
    the newtx (or mathptmx) Times fonts.  This is genuine Times maths rather
    than a match for it, and it is what a journal's own typesetting will look
    like.  It needs a working LaTeX, and every figure takes noticeably longer
    to draw.  This machine has TeX Live 2025, so it works here.

Start with "stix"; switch to "latex" for the final version if the maths in a
figure is heavy enough that the difference shows.

ONE GOTCHA.  Inside $...$ matplotlib italicises digits as well as letters, so
"$\\alpha_i = 0.300^\\circ$" prints the number in italics, which is wrong: the
convention is italic variables, upright numbers.  Keep the numbers outside the
maths and you get it right in both engines:

    ax.set_title(r"$\\alpha_i$ = 0.300$^\\circ$")     # good
    ax.set_title(r"$\\alpha_i = 0.300^\\circ$")       # italic 0.300

Real LaTeX (engine="latex") sets digits upright by itself, so this only bites in
"stix" mode, which is exactly where you will not be looking for it.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Widths of one and two journal columns, in inches.  3.375 in = 86 mm is the
# usual single column (APS, AIP, Elsevier); 6.9 in = 175 mm spans two.
SINGLE_COLUMN = 3.375
DOUBLE_COLUMN = 6.9

# The LaTeX packages used when engine="latex".  newtx is the modern Times
# package; if it is missing, mathptmx is the older fallback that ships with
# nearly every LaTeX installation.
LATEX_PREAMBLE = r"""
\usepackage{newtxtext}
\usepackage{newtxmath}
\usepackage{siunitx}
"""

LATEX_PREAMBLE_FALLBACK = r"""
\usepackage{mathptmx}
"""


def use_publication_style(font_size=9, engine="stix", font="Times New Roman",
                          dpi=600, linewidth=0.8, minor_ticks=True):
    """Set matplotlib's defaults to something a journal will accept.

    font_size   : size of the tick labels, in points.  Axis labels come out one
                  point larger and the title two, so a figure printed at its
                  true size stays readable.  8-10 pt suits most journals.
    engine      : "stix" for matplotlib's own Times-matched maths, or "latex"
                  to typeset everything with a real LaTeX.  See the note above.
    font        : the text font.  Times New Roman by default; "STIXGeneral" is
                  the metric-compatible substitute if Times is missing.
    dpi         : resolution for saved raster figures.  600 is the usual
                  requirement for line art; 300 is the floor.
    linewidth   : width of the axes frame and ticks, in points.
    minor_ticks : draw unlabelled minor ticks between the labelled ones.

    Returns the rcParams dict it set, so you can see what changed.
    """
    if engine not in ("stix", "latex"):
        raise ValueError('engine must be "stix" or "latex"')

    settings = {
        # --- fonts --------------------------------------------------------
        # A list of fallbacks: matplotlib walks it until it finds one that is
        # installed, so this still works on a machine without Times.
        "font.family": "serif",
        "font.serif": [font, "STIXGeneral", "DejaVu Serif"],
        "font.size": font_size,

        # --- sizes, relative to font.size ---------------------------------
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size + 2,

        # --- axes ---------------------------------------------------------
        "axes.linewidth": linewidth,
        "axes.labelpad": 3.0,
        # Powers of ten on an axis are drawn as 10^4, not as 1e4.
        "axes.formatter.use_mathtext": True,
        "axes.grid": False,

        # --- ticks --------------------------------------------------------
        # Ticks pointing inwards, on all four sides, is the convention in most
        # physics journals.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": linewidth,
        "ytick.major.width": linewidth,
        "xtick.minor.width": linewidth * 0.75,
        "ytick.minor.width": linewidth * 0.75,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": minor_ticks,
        "ytick.minor.visible": minor_ticks,

        # --- lines and legend ---------------------------------------------
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,

        # --- images (intensity maps) --------------------------------------
        # "nearest" shows the pixels as measured.  Any other interpolation
        # invents values between them, which is fine for a photograph and not
        # fine for data.
        "image.interpolation": "nearest",
        "image.origin": "lower",

        # --- figure and saving --------------------------------------------
        "figure.dpi": 110,          # on screen
        "savefig.dpi": dpi,         # in the file
        "figure.figsize": (SINGLE_COLUMN, SINGLE_COLUMN * 0.75),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Type 42 means the fonts are embedded as TrueType and the text stays
        # selectable and searchable.  Matplotlib's default, Type 3, is rejected
        # outright by several publishers.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    if engine == "stix":
        settings["text.usetex"] = False
        # STIXGeneral is a Times-metric maths font, so $\alpha_f$ sits beside
        # Times text without looking pasted in.
        settings["mathtext.fontset"] = "stix"
        # Italic variables, upright everything else: the ISO convention.
        settings["mathtext.default"] = "it"
    else:
        settings["text.usetex"] = True
        settings["text.latex.preamble"] = LATEX_PREAMBLE

    mpl.rcParams.update(settings)
    return settings


def figure_size(columns=1, ratio=0.75, width=None):
    """A figure size in inches, sized to the column it will be printed in.

    columns : 1 for a single-column figure (3.375 in), 2 for a full-width one
              (6.9 in).  Sizing the figure correctly from the start is what
              makes the fonts come out the right size on paper: if you draw it
              too large and let the journal shrink it, 9 pt type becomes 5 pt.
    ratio   : height as a fraction of width.  0.75 is a comfortable default;
              use 1.0 for a square intensity map plus its colour bar.
    width   : an explicit width in inches, overriding `columns`.

    Returns (width, height), ready to pass to plt.subplots(figsize=...).
    """
    if width is None:
        width = SINGLE_COLUMN if columns == 1 else DOUBLE_COLUMN
    return (width, width * ratio)


def intensity_cmap(name="turbo", zero_colour="black", bad_colour="0.35"):
    """A colour map set up for a counting detector on a logarithmic scale.

    A log scale has no place for zero, so two colours are reserved:

        under : pixels below the bottom of the scale, i.e. the empty ones
        bad   : nan, meaning no measurement -- a dead pixel or a masked one

    Matplotlib treats 0 as invalid on a log scale and would colour empty pixels
    with `bad`, the same as the dead ones.  Nudge them just under the scale
    first and the two stay distinguishable:

        I = np.where(I == 0, 0.5, I)            # 0.5 is positive, but < vmin
        ax.imshow(I, cmap=intensity_cmap(),
                  norm=LogNorm(vmin=1, vmax=np.nanmax(I)))

    "turbo" reads well on a dark background; for a printed page on white,
    "gist_heat_r" with zero_colour="white" is the usual choice.
    """
    cmap = plt.get_cmap(name).copy()
    cmap.set_under(zero_colour)
    cmap.set_bad(bad_colour)
    return cmap


def save_figure(fig, stem, formats=("pdf", "png"), **kwargs):
    """Save one figure in several formats at once.

    stem    : the file name without an extension, e.g. "fig_map_000".
    formats : "pdf" is the vector version to submit; "png" is the one to drop
              into a slide or an email.

    Returns the list of files written.
    """
    written = []
    for extension in formats:
        path = f"{stem}.{extension}"
        fig.savefig(path, **kwargs)
        written.append(path)
    return written


def latex_available():
    """True if engine="latex" will actually work on this machine.

    Draws a tiny throwaway figure containing maths and sees whether it renders.
    """
    saved = dict(mpl.rcParams)
    try:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
        figure = plt.figure(figsize=(0.5, 0.5))
        figure.text(0.5, 0.5, r"$\alpha_f$")
        figure.canvas.draw()
        plt.close(figure)
        return True
    except Exception:
        return False
    finally:
        mpl.rcParams.update(saved)
