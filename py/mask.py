"""Mask the specular rod in an angular GISAXS frame.

The specular rod is the mirror reflection of the beam off the film surface.  It
carries the film's vertical structure, not its lateral structure, and it is
overwhelmingly brighter than the diffuse scattering beside it, so any cut or fit
that includes it is really just measuring the rod.  This module finds it and
removes it, in five steps:

    1. Sum the intensity down every column -> a 1-D profile against phi_f.
    2. The rod is the tallest part of that profile.  Its position is measured,
       never assumed to be phi_f = 0.
    3. Measure the rod's full width at half maximum (FWHM) from the same
       profile, and turn it into a Gaussian sigma.
    4. Blank out every pixel within n_sigma of the rod centre, by setting its
       intensity to nan.  The rows stay in place, so the frame is still a
       rectangular image that can be reshaped and plotted.
    5. Re-measure the profile to confirm the rod is gone.

Nothing about a particular detector, file or beamline is written into this
module: the pixel size, the background level and the search windows are all
measured from whatever file is passed in.  Any file of comma-separated columns
phi_f, alpha_f, intensity[, variance] with '#' comment lines will work.

From Python:

    import mask
    masked = mask.mask_file("a1.dat", "a1-masked.dat")

From a terminal:

    python mask.py a1.dat                  # writes a1-masked.dat
    python mask.py a3.dat out.dat --n-sigma 5

Step 5 is what sets the default n_sigma = 4.  Measured on a1.dat of this data
set, the tallest column left standing after masking is:

    3.0 sigma -> 64,865 cts      3.5 sigma -> 2,345 cts
    4.0 sigma ->    676 cts      5.0 sigma ->   447 cts

against a diffuse shoulder of roughly 270 cts just outside the mask.  Three
sigma is enough for a true Gaussian, but this rod has heavier tails than one, so
it is not enough here; by four sigma the leftover is the same order as the
diffuse scattering around it, and going wider only eats good data.
"""

import os
import re

import numpy as np
import pandas as pd

# For a Gaussian, FWHM = 2*sqrt(2*ln 2) * sigma = 2.354820 * sigma.
# This is a property of the Gaussian shape, not a fitted number.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# How wide the mask is by default, in sigma.  See the note above.
DEFAULT_N_SIGMA = 4.0

# The background is taken from the half of the columns furthest from the rod.
# Expressed as a fraction rather than a fixed angle, so it adapts to whatever
# range of phi_f the file happens to cover.
BACKGROUND_FRACTION = 0.5

# When measuring the rod's width, look only this fraction of the total phi_f
# span either side of it, so a bright spot elsewhere cannot widen the answer.
SEARCH_FRACTION = 0.1

# Column names by width, for files that do not name their own columns.
COLUMN_NAMES = {
    3: ["phi_f", "alpha_f", "intensity"],
    4: ["phi_f", "alpha_f", "intensity", "variance"],
}


def _header_ints(path):
    """The 'Rows:' and 'Columns:' numbers from the file's comment header.

    The exact angle formulas make phi_f depend on y as well as x, and alpha_f on
    x as well as y, so two pixels in the same detector column no longer share a
    phi_f to the last digit.  Grouping on the float values would therefore make
    one group per pixel.  The detector grid is the thing that is actually
    regular, so it is read from the header and used instead.
    """
    found = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            digits = re.findall(r"\d+", value.replace(",", ""))
            if digits:
                found[key.strip().lower()] = int(digits[0])
    return found


def read_frame(path, columns=None):
    """Read one angular .dat file into a DataFrame.

    comment="#" skips the header lines and header=None says the file has no
    column names of its own.  The names are then chosen from how many columns
    the file turned out to have, or taken from `columns` if given.

    The detector grid, if the header states it, is kept in df.attrs so the rest
    of the module can reshape the frame back into an image.
    """
    df = pd.read_csv(path, comment="#", header=None)

    if columns is None:
        if df.shape[1] not in COLUMN_NAMES:
            raise ValueError(
                f"{path}: expected 3 or 4 columns, found {df.shape[1]}. "
                f"Pass columns=[...] to name them yourself.")
        columns = COLUMN_NAMES[df.shape[1]]

    df.columns = columns

    header = _header_ints(path)
    nx = header.get("columns")
    if nx is None:
        # No header to go on: fall back to counting distinct phi_f, which is
        # right for a file written with the flat-detector formula.
        nx = df["phi_f"].nunique()
    df.attrs["nx"] = int(nx)
    df.attrs["ny"] = len(df) // int(nx)
    return df


def grid(df):
    """(ny, nx) of the detector frame this DataFrame holds."""
    nx = df.attrs.get("nx") or df["phi_f"].nunique()
    if nx <= 0 or len(df) % nx:
        raise ValueError(
            f"cannot work out the detector grid: {len(df):,} pixels is not a "
            f"whole number of rows of {nx:,}. Give the file a '# Columns: n' "
            f"header line, or set df.attrs['nx'] yourself.")
    return len(df) // nx, nx


def as_image(df, column="intensity"):
    """One column of the frame, reshaped back into the (ny, nx) detector image."""
    ny, nx = grid(df)
    return df[column].to_numpy().reshape(ny, nx)


def column_profile(df):
    """Step 1: add up the intensity of every pixel in each detector column.

    The frame is reshaped into the detector image and summed down the columns.
    Each column is then labelled with the mean phi_f of its pixels: phi_f drifts
    by about 0.14 of a pixel from the bottom of a column to the top, so the mean
    is a fair label for the whole column.  nan pixels are skipped.
    """
    intensity = as_image(df, "intensity")
    phi = as_image(df, "phi_f")
    return pd.Series(np.nansum(intensity, axis=0), index=phi.mean(axis=0))


def row_profile(df):
    """The mean counts per pixel in each detector row, against alpha_f.

    The companion of column_profile: the image is averaged along each row and
    the row is labelled with its mean alpha_f, which varies by about 0.27 of a
    pixel across a row.  nan pixels are left out of the mean.
    """
    intensity = as_image(df, "intensity")
    alpha = as_image(df, "alpha_f")
    with np.errstate(invalid="ignore"):
        means = np.nanmean(intensity, axis=1)
    return pd.Series(means, index=alpha.mean(axis=1))


def pixel_size(profile):
    """The angular width of one detector column, measured from the file.

    The phi_f values of an arctan axis are very slightly unevenly spaced, so the
    median gap is used rather than the first one.
    """
    phi_values = np.sort(profile.index.to_numpy())
    return float(np.median(np.diff(phi_values)))


def find_rod(df):
    """Steps 2 and 3: measure where the rod is and how wide it is.

    Returns a dict, so each number comes back with its name attached.
    """
    profile = column_profile(df)
    if len(profile) < 3:
        raise ValueError("need at least 3 distinct phi_f columns to find a rod")

    # The phi_f values themselves, as a Series, so they can be compared
    # elementwise.  profile.index is the list of phi_f values.
    phi = profile.index.to_series()
    pixel = pixel_size(profile)
    phi_span = float(phi.max() - phi.min())

    # --- Step 2: position, to the nearest pixel ---------------------------
    # idxmax() gives the INDEX of the largest value, which here is the phi_f of
    # the tallest column.
    tallest = profile.idxmax()

    # The flat background, taken from the half of the columns furthest from the
    # rod.  The median is used rather than the mean because it ignores the odd
    # bright outlier.
    distance = (phi - tallest).abs()
    far_away = distance > distance.quantile(BACKGROUND_FRACTION)
    background = float(profile[far_away].median())

    peak_height = float(profile.max() - background)
    if peak_height <= 0:
        raise ValueError("no peak stands above the background in this frame")

    # --- Step 3: width ----------------------------------------------------
    # Look only at the columns near the rod.
    search = SEARCH_FRACTION * phi_span
    near_rod = profile[(phi > tallest - search) & (phi < tallest + search)]

    # Every column that stands more than half the peak height above background.
    above_half = near_rod[near_rod - background > peak_height / 2.0]

    # The FWHM is the distance from the first such column to the last.  A rod
    # only one column wide has no width by that measure, so it is given one
    # pixel, otherwise sigma would come out zero and the mask would be empty.
    fwhm = float(above_half.index.max() - above_half.index.min())
    if fwhm <= 0:
        fwhm = pixel
    sigma = fwhm / FWHM_PER_SIGMA

    # --- Step 2, refined --------------------------------------------------
    # The tallest column is only accurate to half a pixel, and the two tallest
    # are often within a rounding error of each other, so idxmax() picks one of
    # them almost at random and the mask ends up half a pixel off centre.  The
    # centre of mass of the top half of the peak is steadier: it uses every
    # column, weighted by how far it stands above background.
    top_half = above_half - background
    centre = float((top_half.index.to_series() * top_half).sum() / top_half.sum())

    return {
        "centre": centre,            # phi_f of the rod, centre of mass [deg]
        "tallest": tallest,          # phi_f of the single tallest column [deg]
        "fwhm": fwhm,                # full width at half maximum [deg]
        "sigma": sigma,              # equivalent Gaussian width [deg]
        "background": background,    # summed counts per column, off the rod
        "peak_height": peak_height,  # summed counts per column, on the rod
        "pixel": pixel,              # width of one column [deg]
    }


def mask_rod(df, centre, half_width):
    """Step 4: blank out every pixel within half_width of the rod centre.

    The intensity (and the variance, if the file has one) of those pixels
    becomes nan; the rows themselves stay, so the frame keeps its rectangular
    shape.  Returns the new DataFrame and the number of pixels masked.
    """
    # .copy() makes a new DataFrame, so the one passed in is left untouched.
    masked = df.copy()

    # A True/False value for every row: is this pixel inside the rod band?
    inside_rod = (masked["phi_f"] - centre).abs() <= half_width

    # Blank the data columns this file actually has.
    blank = [c for c in ("intensity", "variance") if c in masked.columns]

    # .loc[rows, columns] = value assigns to just those rows and columns.
    masked.loc[inside_rod, blank] = np.nan

    return masked, int(inside_rod.sum())


def write_frame(df, path, header_lines):
    """Write a DataFrame back out in the same format as the input file."""
    with open(path, "w", newline="") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f, index=False, header=False, float_format="%.6g",
                  na_rep="nan", lineterminator="\n")


def default_output_path(in_path):
    """"a1.dat" -> "a1-masked.dat", whatever the name and extension are."""
    stem, extension = os.path.splitext(in_path)
    return f"{stem}-masked{extension}"


def mask_file(in_path, out_path=None, n_sigma=DEFAULT_N_SIGMA, half_width=None,
              columns=None, verbose=True):
    """Find and mask the specular rod in one angular .dat file.

    in_path    : the angular file to read, e.g. "a1.dat"
    out_path   : where to write the masked frame; None writes nothing
    n_sigma    : how many sigma either side of the rod centre to blank out.
                 See the note at the top of this file for why 4 and not 3.
    half_width : give this to override n_sigma and mask a fixed band [deg].
                 Use it when several frames must share one mask width.
    columns    : column names, if the file is not 3 or 4 of the usual ones.

    Returns the masked DataFrame.
    """
    df = read_frame(in_path, columns=columns)
    rod = find_rod(df)

    if half_width is None:
        half_width = n_sigma * rod["sigma"]

    masked, n_masked = mask_rod(df, rod["centre"], half_width)

    # --- Step 5: check the rod is actually gone ---------------------------
    # The tallest column left standing, and for comparison the typical diffuse
    # level just outside the mask.  If the rod is properly gone these two are
    # the same order of magnitude; if the first is far larger, the mask is too
    # narrow and some of the rod survived at its edge.
    after = column_profile(masked)
    after_phi = after.index.to_series()
    left_over = float(after.max() - rod["background"])
    just_outside = after[(after_phi - rod["centre"]).abs().between(
        half_width, 3.0 * half_width)]
    diffuse = float(just_outside.median() - rod["background"])

    if verbose:
        pixel = rod["pixel"]
        print(f"{in_path}: {len(df):,} pixels,"
              f" {len(after):,} columns of {pixel * 1000:.3f} mdeg")
        print(f"  rod centre     : {rod['centre']:+.4f} deg"
              f"   ({rod['centre'] / pixel:+.1f} pixels from phi_f = 0)")
        print(f"  rod FWHM       : {rod['fwhm']:.4f} deg"
              f"   ({rod['fwhm'] / pixel:.1f} pixels)")
        print(f"  sigma          : {rod['sigma']:.4f} deg")
        print(f"  masked band    : {rod['centre'] - half_width:+.4f}"
              f" .. {rod['centre'] + half_width:+.4f} deg"
              f"   (+/- {half_width:.4f} deg)")
        print(f"  pixels masked  : {n_masked:,}"
              f"   ({n_masked / len(df) * 100:.1f} % of the frame)")
        print(f"  column peak    : {rod['peak_height']:,.0f} cts before"
              f"  ->  {left_over:,.0f} cts after")
        print(f"  diffuse beside : {diffuse:,.0f} cts per column"
              f"   (the leftover should be near this)")

    if out_path is not None:
        header_lines = [
            f"# {os.path.basename(in_path)} with the specular rod masked",
            "# " + ", ".join(df.columns),
            "# Units: phi_f [deg], alpha_f [deg], intensity [cts]",
            f"# Columns: {df.attrs['nx']}",
            f"# Rows: {len(df) // df.attrs['nx']}",
            f"# Rod centre measured at phi_f = {rod['centre']:.6f} deg,"
            f" FWHM {rod['fwhm']:.6f} deg",
            f"# Masked band: |phi_f - {rod['centre']:.6f}| <= {half_width:.6f} deg",
            f"# Pixels masked: {n_masked:,} of {len(df):,}",
            "# nan intensity = masked rod, dead pixel, or both",
            "# 0 = a measured zero, not missing data",
        ]
        write_frame(masked, out_path, header_lines)
        if verbose:
            print(f"  wrote          : {out_path}")

    return masked


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mask the specular rod in an angular GISAXS frame.")
    parser.add_argument("input", help="angular .dat file to mask, e.g. a1.dat")
    parser.add_argument("output", nargs="?", default=None,
                        help="where to write it; default <input>-masked.dat")
    parser.add_argument("--n-sigma", type=float, default=DEFAULT_N_SIGMA,
                        help=f"mask half-width in sigma (default {DEFAULT_N_SIGMA})")
    parser.add_argument("--half-width", type=float, default=None,
                        help="mask half-width in deg, overriding --n-sigma")
    args = parser.parse_args()

    mask_file(args.input,
              args.output or default_output_path(args.input),
              n_sigma=args.n_sigma,
              half_width=args.half_width)
