"""Mask the specular spot: an ellipse sized from the data.

The specular spot is the mirror reflection of the beam, at phi_f = 0 and
alpha_f = alpha_i.  It is found and measured like this:

    1. Take a window of +/- W pixel widths around the predicted position (or,
       with follow_peak, around the local maximum the window climbs to from
       there, for a spot that is not where alpha_i predicts it).
    2. Average the window over its rows -> P(phi_f), and over its columns
       -> Q(alpha_f).
    3. For each profile: background B = median of the outer 20 % of the
       window, height H = max - B, FWHM from the two crossings of
       (P - B) = H/2 found by linear interpolation, sigma = FWHM / 2.3548,
       centre = centroid of (P - B) over the points above H/2.
    4. Mask every pixel inside the ellipse
           ((phi_f - phi0) / (n sigma_phi))^2 + ((alpha_f - alpha0) / (n sigma_alpha))^2 <= 1
       by setting intensity (and variance) to NaN, as mask.mask_rod does.

A FWHM narrower than one pixel is replaced by one pixel.  Nothing about the
detector is assumed: the pixel widths are measured from the frame.
"""

import warnings

import numpy as np

from mask import DEFAULT_N_SIGMA, FWHM_PER_SIGMA, as_image

# Half-width of the search window, in pixel angular widths.
DEFAULT_WINDOW = 15

# The background is the median of this outer fraction of the window
# (half of it at each end).
BACKGROUND_FRACTION = 0.2

# follow_peak gives up after this many re-centring steps.
MAX_FOLLOW_STEPS = 50


class SpotError(ValueError):
    """The spot could not be measured; the message says why and what to try."""


def _pixel(values):
    """Median spacing of a sorted 1D coordinate: one pixel width [deg]."""
    return float(np.median(np.abs(np.diff(values))))


def measure_peak(coord, profile, centre, half_width, pixel):
    """Background, height, FWHM, sigma and centroid of one 1D peak.

    coord      : window coordinates (ascending) [deg]
    profile    : mean counts at each coordinate (NaN allowed)
    centre     : the window centre [deg]
    half_width : window half-width [deg]
    pixel      : one pixel width [deg], the smallest FWHM allowed
    """
    coord = np.asarray(coord, dtype=float)
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(profile)
    if finite.sum() < 5:
        raise SpotError("fewer than 5 unmasked points in the window")

    outer = np.abs(coord - centre) >= (1.0 - BACKGROUND_FRACTION) * half_width
    outer &= finite
    if not outer.any():
        raise SpotError("no unmasked points in the outer 20 % of the window "
                        "to measure the background from")
    background = float(np.median(profile[outer]))
    peak = int(np.nanargmax(profile))
    height = float(profile[peak] - background)
    if not height > 0:
        raise SpotError("no peak stands above the background in the window")

    signal = profile - background
    half = height / 2.0

    def crossing(step):
        """Walk from the peak until the signal drops below half maximum and
        interpolate linearly between the last point above and the first
        point below.  NaN points are stepped over."""
        last = peak
        i = peak + step
        while 0 <= i < len(signal):
            if np.isfinite(signal[i]):
                if signal[i] < half:
                    s0, s1 = signal[last], signal[i]
                    t = (s0 - half) / (s0 - s1)
                    return coord[last] + t * (coord[i] - coord[last]), last
                last = i
            i += step
        return None, last

    left, first_above = crossing(-1)
    right, last_above = crossing(+1)
    if left is None or right is None:
        side = "below" if left is None else "above"
        raise SpotError(
            f"the profile does not fall to half maximum {side} its peak inside "
            f"the window (the peak is at {coord[peak]:+.4f} deg, window "
            f"{centre - half_width:+.4f} .. {centre + half_width:+.4f} deg)")

    fwhm = float(right - left)
    narrow = fwhm < pixel
    if narrow:
        fwhm = pixel
    sigma = fwhm / FWHM_PER_SIGMA

    top = slice(first_above, last_above + 1)
    weights = np.where(np.isfinite(signal[top]), signal[top], 0.0)
    weights = np.clip(weights, 0.0, None)
    centroid = float(np.sum(coord[top] * weights) / np.sum(weights))

    return {
        "centre": centroid, "fwhm": fwhm, "sigma": sigma,
        "background": background, "height": height,
        "left": float(left), "right": float(right),
        "peak": float(coord[peak]), "fwhm_was_below_pixel": bool(narrow),
    }


def find_spot(df, alpha_i, window=DEFAULT_WINDOW, follow_peak=False):
    """Locate and measure the specular spot in an angular frame.

    df          : frame with phi_f, alpha_f, intensity and df.attrs["nx"]
    alpha_i     : incidence angle [deg]; the spot is predicted at
                  (phi_f, alpha_f) = (0, alpha_i)
    window      : half-width W of the window, in pixel angular widths
    follow_peak : re-centre the window on its brightest pixel, and repeat
                  until the window is centred on the brightest pixel in it.
                  Off by default: the window stays on the prediction.

    Returns a dict with phi0, alpha0, sigma_phi, sigma_alpha, the two
    profile measurements and the window that was used.
    """
    phi_col = as_image(df, "phi_f").mean(axis=0)
    alpha_row = as_image(df, "alpha_f").mean(axis=1)
    image = as_image(df, "intensity")
    pix_phi = _pixel(phi_col)
    pix_alpha = _pixel(alpha_row)
    half_phi = window * pix_phi
    half_alpha = window * pix_alpha

    predicted = (0.0, float(alpha_i))
    phi_c, alpha_c = predicted

    def select(phi_c, alpha_c):
        cols = np.flatnonzero(np.abs(phi_col - phi_c) <= half_phi)
        rows = np.flatnonzero(np.abs(alpha_row - alpha_c) <= half_alpha)
        if len(cols) < 5 or len(rows) < 5:
            raise SpotError(
                f"the window around phi_f = {phi_c:+.4f} deg, alpha_f = "
                f"{alpha_c:.4f} deg lies (mostly) outside the frame")
        return rows, cols

    rows, cols = select(phi_c, alpha_c)
    steps = 0
    if follow_peak:
        # Hill-climb: move the window onto its brightest pixel until the
        # brightest pixel is the one the window is already centred on.
        centre_pixel = None
        seen = set()
        while steps < MAX_FOLLOW_STEPS:
            sub = image[np.ix_(rows, cols)]
            if not np.isfinite(sub).any():
                break
            r, c = np.unravel_index(np.nanargmax(sub), sub.shape)
            brightest = (int(rows[r]), int(cols[c]))
            if brightest == centre_pixel or brightest in seen:
                break
            seen.add(brightest)
            centre_pixel = brightest
            phi_c, alpha_c = float(phi_col[cols[c]]), float(alpha_row[rows[r]])
            rows, cols = select(phi_c, alpha_c)
            steps += 1

    sub = image[np.ix_(rows, cols)]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        P = np.nanmean(sub, axis=0)      # mean over rows    -> P(phi_f)
        Q = np.nanmean(sub, axis=1)      # mean over columns -> Q(alpha_f)

    try:
        horiz = measure_peak(phi_col[cols], P, phi_c, half_phi, pix_phi)
        vert = measure_peak(alpha_row[rows], Q, alpha_c, half_alpha, pix_alpha)
    except SpotError as err:
        hint = _hint(image, phi_col, alpha_row, predicted, 4 * half_phi,
                     4 * half_alpha, pix_phi, pix_alpha, follow_peak)
        raise SpotError(f"{err}. {hint}") from None

    return {
        "phi0": horiz["centre"], "alpha0": vert["centre"],
        "sigma_phi": horiz["sigma"], "sigma_alpha": vert["sigma"],
        "fwhm_phi": horiz["fwhm"], "fwhm_alpha": vert["fwhm"],
        "phi_profile": horiz, "alpha_profile": vert,
        "predicted": predicted,
        "window_centre": (float(phi_c), float(alpha_c)),
        "window_pixels": window,
        "half_width_phi": half_phi, "half_width_alpha": half_alpha,
        "pixel_phi": pix_phi, "pixel_alpha": pix_alpha,
        "follow_steps": steps,
    }


def _hint(image, phi_col, alpha_row, predicted, reach_phi, reach_alpha,
          pix_phi, pix_alpha, follow_peak):
    """Where the brightest pixel near the prediction is, to suggest a fix."""
    cols = np.flatnonzero(np.abs(phi_col - predicted[0]) <= reach_phi)
    rows = np.flatnonzero(np.abs(alpha_row - predicted[1]) <= reach_alpha)
    if len(cols) == 0 or len(rows) == 0:
        return "Check alpha_i."
    sub = image[np.ix_(rows, cols)]
    if not np.isfinite(sub).any():
        return "The region around the predicted position is fully masked."
    r, c = np.unravel_index(np.nanargmax(sub), sub.shape)
    phi, alpha = phi_col[cols[c]], alpha_row[rows[r]]
    away = max(abs(phi - predicted[0]) / pix_phi,
               abs(alpha - predicted[1]) / pix_alpha)
    text = (f"The brightest pixel near the prediction is at phi_f = {phi:+.4f} "
            f"deg, alpha_f = {alpha:.4f} deg ({sub[r, c]:,.0f} cts, "
            f"{away:.0f} pixels from phi_f = 0, alpha_f = alpha_i).")
    if follow_peak:
        return text + " Try a larger window W."
    return text + (" If that is the spot, tick 'follow the peak' or make the "
                   f"window W larger than {np.ceil(away * 1.3):.0f} pixels.")


def mask_spot(df, phi0, alpha0, half_phi, half_alpha):
    """Set intensity and variance to NaN inside the ellipse; return (df, n)."""
    masked = df.copy()
    inside = (((masked["phi_f"] - phi0) / half_phi) ** 2
              + ((masked["alpha_f"] - alpha0) / half_alpha) ** 2) <= 1.0
    blank = [c for c in ("intensity", "variance") if c in masked.columns]
    masked.loc[inside, blank] = np.nan
    return masked, int(inside.sum())


def find_and_mask(df, alpha_i, n_sigma=DEFAULT_N_SIGMA, window=DEFAULT_WINDOW,
                  follow_peak=False):
    """find_spot() then mask_spot() with half-axes n_sigma * sigma."""
    spot = find_spot(df, alpha_i, window=window, follow_peak=follow_peak)
    half_phi = n_sigma * spot["sigma_phi"]
    half_alpha = n_sigma * spot["sigma_alpha"]
    masked, n = mask_spot(df, spot["phi0"], spot["alpha0"], half_phi, half_alpha)
    spot.update(n_sigma=n_sigma, half_axis_phi=half_phi,
                half_axis_alpha=half_alpha, n_masked=n)
    return masked, spot
