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

The same measurement also runs on the raw detector frame, in x and y (mm),
before any angle is computed: find_offset() uses it on the specular spot and
on the direct beam to find where the direct beam sits on the detector,
(x0, y0), when it is not at the origin.
"""

import warnings

import numpy as np

from gisaxs import direct_beam_offset, exit_angles, specular_y
from mask import DEFAULT_N_SIGMA, FWHM_PER_SIGMA, as_image

# Half-width of the search window, in pixel angular widths.
DEFAULT_WINDOW = 15

# The background is the median of this outer fraction of the window
# (half of it at each end).
BACKGROUND_FRACTION = 0.2

# follow_peak gives up after this many re-centring steps.
MAX_FOLLOW_STEPS = 50

# find_offset() accepts the direct beam as the origin if, measured from it,
# the specular spot sits at alpha_f = alpha_i within this many pixels.
OFFSET_TOLERANCE = 1.0


class SpotError(ValueError):
    """The spot could not be measured; the message says why and what to try."""


def _pixel(values):
    """Median spacing of a sorted 1D coordinate: one pixel width [deg]."""
    return float(np.median(np.abs(np.diff(values))))


def measure_peak(coord, profile, centre, half_width, pixel, unit="deg"):
    """Background, height, FWHM, sigma and centroid of one 1D peak.

    coord      : window coordinates (ascending) [deg, or mm on the detector]
    profile    : mean counts at each coordinate (NaN allowed)
    centre     : the window centre
    half_width : window half-width
    pixel      : one pixel width, the smallest FWHM allowed
    unit       : unit of the coordinates, for messages
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
            f"the window (the peak is at {coord[peak]:+.4f} {unit}, window "
            f"{centre - half_width:+.4f} .. {centre + half_width:+.4f} {unit})")

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


def _locate(df, columns, predicted, window, follow_peak, unit, names):
    """Find and measure a peak in the frame, in the coordinates `columns`.

    columns   : (horizontal, vertical) column names, e.g. ("phi_f", "alpha_f")
                or ("x", "y")
    predicted : where the peak should be, in those coordinates
    names     : what to call the two coordinates in messages
    Returns centres, widths and the window, keyed h* (horizontal) and v*
    (vertical).
    """
    h_col = as_image(df, columns[0]).mean(axis=0)
    v_row = as_image(df, columns[1]).mean(axis=1)
    image = as_image(df, "intensity")
    pix_h = _pixel(h_col)
    pix_v = _pixel(v_row)
    half_h = window * pix_h
    half_v = window * pix_v
    predicted = (float(predicted[0]), float(predicted[1]))
    h_c, v_c = predicted

    def select(h_c, v_c):
        cols = np.flatnonzero(np.abs(h_col - h_c) <= half_h)
        rows = np.flatnonzero(np.abs(v_row - v_c) <= half_v)
        if len(cols) < 5 or len(rows) < 5:
            raise SpotError(
                f"the window around {names[0]} = {h_c:+.4f} {unit}, {names[1]} = "
                f"{v_c:.4f} {unit} lies (mostly) outside the frame")
        return rows, cols

    rows, cols = select(h_c, v_c)
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
            h_c, v_c = float(h_col[cols[c]]), float(v_row[rows[r]])
            rows, cols = select(h_c, v_c)
            steps += 1

    sub = image[np.ix_(rows, cols)]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        P = np.nanmean(sub, axis=0)      # mean over rows    -> P(horizontal)
        Q = np.nanmean(sub, axis=1)      # mean over columns -> Q(vertical)

    try:
        horiz = measure_peak(h_col[cols], P, h_c, half_h, pix_h, unit)
        vert = measure_peak(v_row[rows], Q, v_c, half_v, pix_v, unit)
    except SpotError as err:
        hint = _hint(image, h_col, v_row, predicted, 4 * half_h, 4 * half_v,
                     pix_h, pix_v, follow_peak, unit, names)
        raise SpotError(f"{err}. {hint}") from None

    return {
        "h0": horiz["centre"], "v0": vert["centre"],
        "h_profile": horiz, "v_profile": vert,
        "predicted": predicted,
        "window_centre": (float(h_c), float(v_c)),
        "window_pixels": window,
        "half_width_h": half_h, "half_width_v": half_v,
        "pixel_h": pix_h, "pixel_v": pix_v,
        "follow_steps": steps,
    }


def _hint(image, h_col, v_row, predicted, reach_h, reach_v, pix_h, pix_v,
          follow_peak, unit, names):
    """Where the brightest pixel near the prediction is, to suggest a fix."""
    cols = np.flatnonzero(np.abs(h_col - predicted[0]) <= reach_h)
    rows = np.flatnonzero(np.abs(v_row - predicted[1]) <= reach_v)
    if len(cols) == 0 or len(rows) == 0:
        return "Check alpha_i."
    sub = image[np.ix_(rows, cols)]
    if not np.isfinite(sub).any():
        return "The region around the predicted position is fully masked."
    r, c = np.unravel_index(np.nanargmax(sub), sub.shape)
    h, v = h_col[cols[c]], v_row[rows[r]]
    away = max(abs(h - predicted[0]) / pix_h, abs(v - predicted[1]) / pix_v)
    text = (f"The brightest pixel near the prediction is at {names[0]} = {h:+.4f} "
            f"{unit}, {names[1]} = {v:.4f} {unit} ({sub[r, c]:,.0f} cts, "
            f"{away:.0f} pixels from the predicted position).")
    if follow_peak:
        return text + " Try a larger window W."
    return text + (" If that is the spot, tick 'follow the peak' or make the "
                   f"window W larger than {np.ceil(away * 1.3):.0f} pixels.")


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
    got = _locate(df, ("phi_f", "alpha_f"), (0.0, alpha_i), window, follow_peak,
                  "deg", ("phi_f", "alpha_f"))
    horiz, vert = got["h_profile"], got["v_profile"]
    return {
        "phi0": got["h0"], "alpha0": got["v0"],
        "sigma_phi": horiz["sigma"], "sigma_alpha": vert["sigma"],
        "fwhm_phi": horiz["fwhm"], "fwhm_alpha": vert["fwhm"],
        "phi_profile": horiz, "alpha_profile": vert,
        "predicted": got["predicted"],
        "window_centre": got["window_centre"],
        "window_pixels": window,
        "half_width_phi": got["half_width_h"],
        "half_width_alpha": got["half_width_v"],
        "pixel_phi": got["pixel_h"], "pixel_alpha": got["pixel_v"],
        "follow_steps": got["follow_steps"],
    }


def find_on_detector(df, L, alpha_i, window=DEFAULT_WINDOW, follow_peak=True):
    """Locate the specular spot on the raw detector, in x, y [mm].

    df          : the detector frame (x, y, intensity), before any conversion
    L           : sample-detector distance [mm]
    alpha_i     : the nominal incidence angle [deg]; the search starts where
                  it puts the spot, x = 0, y = L tan(2 alpha_i)
    window      : half-width W of the window, in pixels
    follow_peak : on by default here: the point is to find a spot that is
                  not where the nominal alpha_i puts it

    Returns x0, y0 (the centroids above half maximum), the FWHMs and the
    window that was used.
    """
    predicted = (0.0, L * np.tan(np.radians(2.0 * alpha_i)))
    got = _locate(df, ("x", "y"), predicted, window, follow_peak, "mm",
                  ("x", "y"))
    horiz, vert = got["h_profile"], got["v_profile"]
    return {
        "x0": got["h0"], "y0": got["v0"],
        "fwhm_x": horiz["fwhm"], "fwhm_y": vert["fwhm"],
        "x_profile": horiz, "y_profile": vert,
        "predicted": got["predicted"],
        "window_centre": got["window_centre"],
        "window_pixels": window,
        "pixel_x": got["pixel_h"], "pixel_y": got["pixel_v"],
        "follow_steps": got["follow_steps"],
    }


def find_direct_beam(df, y_guess, window=DEFAULT_WINDOW, follow_peak=True,
                     x_guess=0.0):
    """Locate the direct beam on the raw detector, if it shows at all.

    It is usually behind a beamstop, and an edge of it may be cut off, which
    pulls the centroid away from the true centre: use it as a cross-check.
    The search starts at x = x_guess, y = y_guess [mm].
    """
    got = _locate(df, ("x", "y"), (x_guess, y_guess), window, follow_peak, "mm",
                  ("x", "y"))
    return {
        "x0": got["h0"], "y0": got["v0"],
        "fwhm_x": got["h_profile"]["fwhm"], "fwhm_y": got["v_profile"]["fwhm"],
        "height": got["v_profile"]["height"],
        "follow_steps": got["follow_steps"],
    }


def find_offset(df, geom, window=DEFAULT_WINDOW, tolerance=OFFSET_TOLERANCE):
    """Where the direct beam sits on the detector, (x0, y0), found from data.

    The formulas assume the direct beam at the origin; then the specular spot
    sits at phi_f = 0, alpha_f = alpha_i (x = 0, y = L tan 2 alpha_i).  When
    the direct beam is really at (x0, y0), every pixel has to move by
    (-x0, -y0).  Two estimates are made on the raw frame df and checked, for
    each axis on its own:

      1. the direct beam itself, if it shows: its centroid.  Used for an axis
         if, measured from it, the specular spot sits at its target on that
         axis (phi_f = 0 for x, alpha_f = alpha_i for y) within `tolerance`
         pixels.
      2. the specular spot at (x_s, y_s): x0 = x_s, y0 = y_s - L tan(2 alpha_i),
         which puts the spot at phi_f = 0, alpha_f = alpha_i by construction.

    The direct beam is preferred, being the measured origin; where it fails
    the check (not found, or partly behind the beamstop so that its centroid
    is off) the spot is used.  Raises SpotError if the spot is not found.

    Returns a dict: the spot, both estimates ("direct" is None if the beam does
    not show), the method used for each axis ("direct" or "specular"), x0 and
    y0 [mm], and where the spot then sits (phi_f_spot, alpha_f_spot [deg]).
    """
    found = find_on_detector(df, geom.L, geom.alpha_i, window=window)
    x_s, y_s = found["x0"], found["y0"]
    pixel_phi = float(np.degrees(found["pixel_x"] / geom.L))    # one pixel [deg]
    pixel_alpha = float(np.degrees(found["pixel_y"] / geom.L))

    def spot_after(x0, y0):
        """phi_f, alpha_f of the spot once every pixel has moved by (-x0, -y0)."""
        phi, alpha = exit_angles(x_s - x0, y_s - y0, geom)
        return float(phi), float(alpha)

    predicted_y = float(specular_y(geom))                     # L tan 2 alpha_i
    phi_raw, alpha_raw = spot_after(0.0, 0.0)
    spot = {
        "x": x_s, "y": y_s, "fwhm_x": found["fwhm_x"], "fwhm_y": found["fwhm_y"],
        "phi_f": phi_raw, "alpha_f": alpha_raw,
        "predicted_y": predicted_y,
        "pixels_off_x": float(x_s / found["pixel_x"]),
        "pixels_off_y": float((y_s - predicted_y) / found["pixel_y"]),
    }

    x0_spec, y0_spec = direct_beam_offset(x_s, y_s, geom)
    phi, alpha = spot_after(x0_spec, y0_spec)
    specular = {"x0": x0_spec, "y0": y0_spec, "phi_f_spot": phi, "alpha_f_spot": alpha}

    direct = None
    try:
        # Look where the spot says the direct beam should be.  Each axis is
        # checked with the other one taken from the spot.
        db = find_direct_beam(df, y0_spec, window=window, x_guess=x0_spec)
        phi = spot_after(db["x0"], y0_spec)[0]
        alpha = spot_after(x0_spec, db["y0"])[1]
        error_x = phi / pixel_phi
        error_y = (alpha - geom.alpha_i) / pixel_alpha
        direct = dict(db, phi_f_spot=phi, alpha_f_spot=alpha,
                      error_x=error_x, error_y=error_y,
                      ok_x=bool(abs(error_x) <= tolerance),
                      ok_y=bool(abs(error_y) <= tolerance))
    except SpotError:
        pass

    use_x = direct is not None and direct["ok_x"]
    use_y = direct is not None and direct["ok_y"]
    x0 = direct["x0"] if use_x else x0_spec
    y0 = direct["y0"] if use_y else y0_spec
    phi_f_spot, alpha_f_spot = spot_after(x0, y0)
    return {
        "alpha_i": float(geom.alpha_i),
        "pixel_phi_deg": pixel_phi, "pixel_alpha_deg": pixel_alpha,
        "tolerance_pixels": float(tolerance),
        "spot": spot, "direct": direct, "specular": specular,
        "method_x": "direct" if use_x else "specular",
        "method_y": "direct" if use_y else "specular",
        "x0": float(x0), "y0": float(y0),
        "phi_f_spot": phi_f_spot, "alpha_f_spot": alpha_f_spot,
    }


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
