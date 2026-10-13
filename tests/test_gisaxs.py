"""Tests for the GISAXS conversion, masks, profiles and export (plain CPython).

    pip install numpy pandas matplotlib pytest
    pytest tests/
"""

import gzip
import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

import gisaxs  # noqa: E402
import mask  # noqa: E402
import read_pro  # noqa: E402
import spot  # noqa: E402
from gisaxs import Geometry, InputError  # noqa: E402

TEST_DATA = ROOT / "test_data"
REAL_PRO = TEST_DATA / "various-angles_000.pro"
REAL_DAT = TEST_DATA / "frame-000.dat"

TOY = Geometry(L=86.0, alpha_i=1.0, wavelength=0.15406)
TOL = 1e-5

# plotstyle.py (provided, unchanged) uses Colormap.set_under/set_bad, which
# matplotlib >= 3.11 flags as pending deprecation.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The set_(under|bad) function:PendingDeprecationWarning")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_frame(nx, ny, pitch, x0, y0, counts=None, variance=True):
    """A row-major detector frame (bottom row first, x ascending in a row)."""
    xs = x0 + pitch * np.arange(nx)
    ys = y0 + pitch * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)
    df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel()})
    df["intensity"] = (np.zeros(nx * ny) if counts is None
                       else np.asarray(counts, dtype=float).ravel())
    if variance:
        df["variance"] = df["intensity"]
    df.attrs["nx"], df.attrs["ny"] = nx, ny
    return df


def write_dat(path, df, rows=None, columns=None, extra_lines=0):
    ny, nx = df.attrs["ny"], df.attrs["nx"]
    with open(path, "w", newline="\n") as f:
        f.write("# 2D Dataset\n")
        f.write(f"# Rows: {rows if rows is not None else ny}\n")
        f.write(f"# Columns: {columns if columns is not None else nx}\n")
        f.write("# X, Y, Z [, ZVariance]\n")
        f.write("# Units: X [mm], Y [mm], Z [cts]\n")
        df.to_csv(f, header=False, index=False, float_format="%.17g")
        for _ in range(extra_lines):
            f.write("0,0,0,0\n")


# ---------------------------------------------------------------------------
# conversion: the table of section 9
# ---------------------------------------------------------------------------

EXPECTED = [
    # x, y, phi_f, alpha_f, kfx, kfy, kfz, qx, qy, qz
    (0, 0, 0, -1.000000, 40.777803, 0, -0.711779, 0, 0, 0),
    (0, 3, 0, 0.997880, 40.777829, 0, 0.710270, 0.000026, 0, 1.422049),
    (2, 2, 1.331882, 0.332130, 40.772311, 0.947953, 0.236414,
     -0.005492, 0.947953, 0.948193),
    (3, 6, 1.995755, 2.989102, 40.703821, 1.418390, 2.126724,
     -0.073982, 1.418390, 2.838503),
]
NAMES = ["phi_f", "alpha_f", "kfx", "kfy", "kfz", "qx", "qy", "qz"]


@pytest.mark.parametrize("row", EXPECTED, ids=[f"({r[0]},{r[1]})" for r in EXPECTED])
def test_toy_geometry_table(row):
    x, y, *expected = row
    got = gisaxs.pixel(x, y, TOY)
    for name, value in zip(NAMES, expected):
        assert got[name] == pytest.approx(value, abs=TOL), name


def test_vectorised_matches_scalar():
    x = np.array([r[0] for r in EXPECTED], dtype=float)
    y = np.array([r[1] for r in EXPECTED], dtype=float)
    got = gisaxs.pixel(x, y, TOY)
    for i, row in enumerate(EXPECTED):
        for name, value in zip(NAMES, row[2:]):
            assert got[name][i] == pytest.approx(value, abs=TOL)


def test_direct_beam_is_minus_alpha_i_and_q_zero():
    got = gisaxs.pixel(0.0, 0.0, TOY)
    assert got["alpha_f"] == pytest.approx(-TOY.alpha_i, abs=1e-12)
    assert got["phi_f"] == 0.0
    for c in ("qx", "qy", "qz"):
        assert got[c] == pytest.approx(0.0, abs=1e-12)


def test_specular_position():
    y = TOY.L * np.tan(np.radians(2 * TOY.alpha_i))
    got = gisaxs.pixel(0.0, y, TOY)
    assert got["alpha_f"] == pytest.approx(TOY.alpha_i, abs=1e-12)
    assert got["phi_f"] == 0.0
    assert got["qx"] == pytest.approx(0.0, abs=1e-12)
    assert got["qy"] == pytest.approx(0.0, abs=1e-12)
    assert got["qz"] == pytest.approx(2 * TOY.k * np.sin(np.radians(TOY.alpha_i)),
                                      abs=1e-12)


def test_horizon_position():
    got = gisaxs.pixel(0.0, gisaxs.horizon_y(TOY), TOY)
    assert got["alpha_f"] == pytest.approx(0.0, abs=1e-12)


def test_kf_has_length_k_everywhere():
    rng = np.random.default_rng(1)
    x = rng.uniform(-40, 40, 10_000)
    y = rng.uniform(-5, 80, 10_000)
    kfx, kfy, kfz = gisaxs.final_wavevector(x, y, TOY)
    assert np.allclose(np.sqrt(kfx ** 2 + kfy ** 2 + kfz ** 2), TOY.k,
                       rtol=1e-13, atol=0)


def test_phi_f_increases_with_x():
    phi, _ = gisaxs.exit_angles(np.array([-2.0, 0.0, 2.0]), np.zeros(3), TOY)
    assert phi[0] < 0 == phi[1] < phi[2]


def test_angle_form_of_q_matches_a_over_r_form():
    x, y = np.array([2.0, -3.0]), np.array([2.0, 6.0])
    got = gisaxs.pixel(x, y, TOY)
    af, pf = np.radians(got["alpha_f"]), np.radians(got["phi_f"])
    ai, k = np.radians(TOY.alpha_i), TOY.k
    assert np.allclose(got["qx"], k * (np.cos(af) * np.cos(pf) - np.cos(ai)))
    assert np.allclose(got["qy"], k * np.cos(af) * np.sin(pf))
    assert np.allclose(got["qz"], k * (np.sin(af) + np.sin(ai)))


def test_dataframe_steps_add_columns():
    df = make_frame(4, 3, 1.0, -2.0, 0.0)
    gisaxs.add_exit_angles(df, TOY)
    assert list(df.columns[-2:]) == ["phi_f", "alpha_f"]
    gisaxs.add_final_wavevector(df, TOY)
    assert list(df.columns[-3:]) == ["kfx", "kfy", "kfz"]
    gisaxs.add_scattering_vector(df, TOY)
    assert list(df.columns[-3:]) == ["qx", "qy", "qz"]


# ---------------------------------------------------------------------------
# horizon cut
# ---------------------------------------------------------------------------

def test_horizon_cut_keeps_whole_rows():
    nx, ny = 7, 40
    df = gisaxs.convert(make_frame(nx, ny, 0.1, -0.3, -0.5), TOY)
    cut = gisaxs.horizon_cut(df, TOY)
    rows = cut.attrs["ny"]
    assert cut.attrs["nx"] == nx
    assert rows * nx == len(cut)
    assert cut.attrs["rows_dropped"] == ny - rows
    # every kept row is at or above the horizon, every dropped one below
    y_row = mask.as_image(df, "y").mean(axis=1)
    horizon = gisaxs.horizon_y(TOY)
    assert rows == np.sum(y_row >= horizon - 1e-12)
    assert mask.as_image(cut, "y").min() >= horizon - 1e-12
    # the image still reshapes and the row alpha_f is never negative
    assert (gisaxs.row_alpha_f(cut, TOY) >= 0).all()


def test_horizon_cut_everything_below_raises():
    df = gisaxs.convert(make_frame(3, 3, 0.1, 0.0, -1.0), TOY)
    with pytest.raises(InputError):
        gisaxs.horizon_cut(df, TOY)


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

def test_profiles_are_means_and_masked_columns_are_gaps():
    nx, ny = 5, 4
    counts = np.arange(nx * ny, dtype=float).reshape(ny, nx)
    counts[:, 2] = np.nan                       # a fully masked column
    counts[1, 0] = np.nan                       # one masked pixel
    df = gisaxs.convert(make_frame(nx, ny, 0.5, -1.0, 3.0, counts), TOY)

    col = gisaxs.column_mean_profile(df, "phi_f")
    assert len(col) == nx
    assert np.isnan(col.iloc[2])
    assert col.iloc[0] == pytest.approx(np.nanmean(counts[:, 0]))
    assert col.iloc[4] == pytest.approx(counts[:, 4].mean())
    assert np.all(np.diff(col.index) > 0)

    row = gisaxs.row_profile(df, "qz")
    assert len(row) == ny
    assert row.iloc[3] == pytest.approx(np.nanmean(counts[3]))
    assert np.all(np.diff(row.index) > 0)

    # mask.py's sum-based column_profile is untouched
    assert mask.column_profile(df).iloc[1] == pytest.approx(counts[:, 1].sum())


@pytest.mark.parametrize("window", [3, 5, 7, 9, 15])
def test_smoothing_is_the_centred_rolling_mean(window):
    rng = np.random.default_rng(3)
    profile = pd.Series(rng.poisson(20, 60).astype(float),
                        index=np.linspace(-1.0, 1.0, 60))
    got = gisaxs.smooth_profile(profile, window)
    expected = profile.rolling(window, center=True, min_periods=1).mean()
    pd.testing.assert_series_equal(got, expected)
    # min_periods=1: the first point averages the half-window that exists
    assert got.iloc[0] == pytest.approx(profile.iloc[:window // 2 + 1].mean())
    assert got.iloc[30] == pytest.approx(profile.iloc[30 - window // 2:31 + window // 2].mean())


def test_smoothing_keeps_masked_gaps():
    profile = pd.Series(np.arange(20, dtype=float), index=np.arange(20) * 0.1)
    profile.iloc[8:12] = np.nan                  # a fully masked rod: 4 columns
    got = gisaxs.smooth_profile(profile, 7)
    assert got.isna().tolist() == profile.isna().tolist()
    # next to the gap, the mean is over the unmasked points of the window
    assert got.iloc[7] == pytest.approx(np.mean([4, 5, 6, 7]))
    assert got.iloc[12] == pytest.approx(np.mean([12, 13, 14, 15]))


def test_smoothing_window_one_is_identity_and_bad_windows_are_refused():
    profile = pd.Series([1.0, np.nan, 3.0, 4.0])
    pd.testing.assert_series_equal(gisaxs.smooth_profile(profile, 1), profile)
    for bad in (6, 0, -3, 2.5):
        with pytest.raises(ValueError, match="odd whole number"):
            gisaxs.smooth_profile(profile, bad)
    with pytest.raises(ValueError, match="longer than the profile"):
        gisaxs.smooth_profile(profile, 5)
    assert gisaxs.smooth_profile(profile, 3.0).notna().sum() == 3     # 3.0 is fine


# ---------------------------------------------------------------------------
# reading .dat files
# ---------------------------------------------------------------------------

def test_read_dat_dead_pixels_and_attrs(tmp_path):
    counts = np.array([[0, 1, -1], [2, 3, 4]], dtype=float)
    df = make_frame(3, 2, 0.075, -0.075, 0.0, counts)
    write_dat(tmp_path / "a.dat", df)
    got = gisaxs.read_dat(tmp_path / "a.dat", nx=3, ny=2)
    assert list(got.columns) == ["x", "y", "intensity", "variance"]
    assert got.attrs["nx"] == 3 and got.attrs["ny"] == 2
    assert np.isnan(got["intensity"][2]) and np.isnan(got["variance"][2])
    assert got.attrs["dead_minus_one"] == 1
    assert got["intensity"][1] == 1


def test_read_dat_without_variance(tmp_path):
    df = make_frame(3, 2, 0.1, 0.0, 0.0, np.ones(6), variance=False)
    write_dat(tmp_path / "a.dat", df)
    got = gisaxs.read_dat(tmp_path / "a.dat")
    assert list(got.columns) == ["x", "y", "intensity"]


def test_read_dat_reorders_column_major(tmp_path):
    counts = np.arange(12, dtype=float).reshape(3, 4)
    df = make_frame(4, 3, 0.1, 0.0, 0.0, counts)
    shuffled = df.iloc[np.argsort(df["x"].to_numpy(), kind="stable")]
    shuffled.attrs = dict(df.attrs)
    write_dat(tmp_path / "a.dat", shuffled)
    got = gisaxs.read_dat(tmp_path / "a.dat")
    assert "re-sorted" in got.attrs["order"]
    assert np.array_equal(mask.as_image(got, "intensity"), counts)


def test_read_dat_header_line_count_mismatch(tmp_path):
    df = make_frame(3, 2, 0.1, 0.0, 0.0)
    write_dat(tmp_path / "a.dat", df, extra_lines=1)
    with pytest.raises(InputError, match="data lines"):
        gisaxs.read_dat(tmp_path / "a.dat")


def test_read_dat_grid_differs_from_pro(tmp_path):
    df = make_frame(3, 2, 0.1, 0.0, 0.0)
    write_dat(tmp_path / "a.dat", df)
    with pytest.raises(InputError, match=r"\.pro detector"):
        gisaxs.read_dat(tmp_path / "a.dat", nx=4, ny=2)


def test_read_dat_without_grid_header(tmp_path):
    (tmp_path / "a.dat").write_text("1,2,3\n4,5,6\n")
    with pytest.raises(InputError, match="Rows"):
        gisaxs.read_dat(tmp_path / "a.dat")


def test_read_dat_not_numbers(tmp_path):
    (tmp_path / "a.dat").write_text("# Rows: 1\n# Columns: 2\nabc,def,ghi\n1,2,3\n")
    with pytest.raises(InputError):
        gisaxs.read_dat(tmp_path / "a.dat")


# ---------------------------------------------------------------------------
# specular-spot mask
# ---------------------------------------------------------------------------

def gaussian_spot_frame(phi0=0.004, alpha0=None, s_phi=0.006, s_alpha=0.009,
                        background=3.0):
    """A converted frame with a Gaussian spot near (0, alpha_i) on a flat
    background, in the toy geometry."""
    geom = Geometry(L=1600.0, alpha_i=0.3, wavelength=0.15418)
    alpha0 = geom.alpha_i if alpha0 is None else alpha0
    df = gisaxs.convert(make_frame(121, 241, 0.075, -4.5, 7.0), geom)
    df = gisaxs.horizon_cut(df, geom)
    phi, alpha = df["phi_f"].to_numpy(), df["alpha_f"].to_numpy()
    df["intensity"] = background + 1000.0 * np.exp(
        -0.5 * ((phi - phi0) / s_phi) ** 2 - 0.5 * ((alpha - alpha0) / s_alpha) ** 2)
    df["variance"] = df["intensity"]
    return df, geom


def test_spot_found_and_measured():
    df, geom = gaussian_spot_frame()
    found = spot.find_spot(df, geom.alpha_i)
    # The centre is the centroid of the points above half maximum, which is
    # good to about half a pixel; the window profiles are means over the
    # window, so the widths are those of the (separable) Gaussian itself.
    assert found["phi0"] == pytest.approx(0.004, abs=0.5 * found["pixel_phi"])
    assert found["alpha0"] == pytest.approx(0.3, abs=0.5 * found["pixel_alpha"])
    assert found["sigma_phi"] == pytest.approx(0.006, rel=0.05)
    assert found["sigma_alpha"] == pytest.approx(0.009, rel=0.05)


def test_spot_mask_is_the_ellipse():
    df, geom = gaussian_spot_frame()
    masked, info = spot.find_and_mask(df, geom.alpha_i, n_sigma=4)
    inside = (((df["phi_f"] - info["phi0"]) / (4 * info["sigma_phi"])) ** 2
              + ((df["alpha_f"] - info["alpha0"]) / (4 * info["sigma_alpha"])) ** 2) <= 1
    assert info["n_masked"] == inside.sum() > 0
    assert masked.loc[inside, "intensity"].isna().all()
    assert masked.loc[inside, "variance"].isna().all()
    assert masked.loc[~inside, "intensity"].notna().all()
    assert df["intensity"].notna().all()             # input untouched


def test_spot_narrower_than_a_pixel_gets_one_pixel():
    # A one-point peak whose neighbours dip below the background: the
    # interpolated FWHM comes out under one pixel and is replaced by one.
    coord = np.arange(-15, 16, dtype=float)
    profile = np.full(coord.size, 10.0)
    profile[15] = 110.0
    profile[[14, 16]] = 0.0
    got = spot.measure_peak(coord, profile, 0.0, 15.0, pixel=1.0)
    assert got["fwhm_was_below_pixel"]
    assert got["fwhm"] == 1.0
    assert got["sigma"] == pytest.approx(1.0 / mask.FWHM_PER_SIGMA)
    assert got["centre"] == 0.0


def test_spot_outside_window_raises_unless_followed():
    # 40 pixels above the prediction: outside the default +/- 15 px window
    df, geom = gaussian_spot_frame(alpha0=0.3 + 40 * 0.075 / 1600 * 180 / np.pi)
    with pytest.raises(spot.SpotError, match="follow the peak"):
        spot.find_spot(df, geom.alpha_i)
    found = spot.find_spot(df, geom.alpha_i, follow_peak=True)
    assert found["alpha0"] == pytest.approx(0.3 + 40 * 0.075 / 1600 * 180 / np.pi,
                                            abs=0.5 * found["pixel_alpha"])


def test_spot_search_starts_at_the_brightest_row_along_phi_zero():
    # The spot sits 60 pixels above the prediction, with a weak bump right at
    # alpha_i: climbing from the prediction stops on the bump; scanning the
    # column at phi_f ~ 0 finds the spot.
    pix = 0.075 / 1600 * 180 / np.pi
    up = 0.3 + 60 * pix
    df, geom = gaussian_spot_frame(alpha0=up)
    phi, alpha = df["phi_f"].to_numpy(), df["alpha_f"].to_numpy()
    df["intensity"] += 30.0 * np.exp(-0.5 * (phi / 0.006) ** 2 - 0.5 * ((alpha - 0.3) / 0.006) ** 2)
    climbed = spot.find_spot(df, geom.alpha_i, follow_peak=True)
    assert climbed["alpha0"] == pytest.approx(0.3, abs=2 * pix)          # the bump
    found = spot.find_spot(df, geom.alpha_i, follow_peak=True, scan_column=True)
    assert found["scan_column"]
    assert found["alpha0"] == pytest.approx(up, abs=0.5 * found["pixel_alpha"])
    assert found["phi0"] == pytest.approx(0.004, abs=0.5 * found["pixel_phi"])
    # a single hot pixel on the column does not win over the spot
    on_column = np.flatnonzero(np.abs(phi) < 0.5 * pix)
    df.loc[on_column[np.argmin(np.abs(alpha[on_column] - 0.36))], "intensity"] = 5000.0
    found = spot.find_spot(df, geom.alpha_i, follow_peak=True, scan_column=True)
    assert found["alpha0"] == pytest.approx(up, abs=0.5 * found["pixel_alpha"])


CSV_PRO = TEST_DATA / "various-csv-v2_0001.pro"
CSV_DAT = TEST_DATA / "various-csv-v2_0001_2D.dat"


@pytest.mark.skipif(not (CSV_PRO.exists() and CSV_DAT.exists()), reason="no csv-series frame")
def test_spot_mask_on_a_frame_whose_spot_is_well_above_alpha_i():
    # In this series the spot sits ~0.14 deg (about 50 pixels) above
    # alpha_i = 0.3 deg, with a weak bump near alpha_i that the window used
    # to land on.
    pro = read_pro.read_pro(CSV_PRO)
    geom = Geometry.from_pro(pro)
    raw = gisaxs.read_dat(CSV_DAT, nx=pro["nx"], ny=pro["ny"])
    cut = gisaxs.horizon_cut(gisaxs.convert(raw, geom), geom)
    masked, info = spot.find_and_mask(cut, geom.alpha_i, follow_peak=True, scan_column=True)
    assert info["alpha0"] == pytest.approx(0.442, abs=0.003)
    image = mask.as_image(cut, "intensity")
    brightest = np.nanmax(image[:, np.abs(mask.as_image(cut, "phi_f").mean(axis=0)) < 0.02])
    assert np.isnan(masked["intensity"].to_numpy()[cut["intensity"].to_numpy() == brightest]).all()


def test_measure_peak_interpolates_half_maximum():
    coord = np.arange(-10, 11, dtype=float)
    profile = np.where(np.abs(coord) <= 2, 10.0, 0.0) + 1.0
    profile[np.abs(coord) == 3] = 6.0            # exactly half: 5 above B
    got = spot.measure_peak(coord, profile, 0.0, 10.0, 1.0)
    assert got["background"] == 1.0
    assert got["height"] == 10.0
    # (P - B) = 10 at |x| = 2 and 5 at |x| = 3 -> half max reached at 3,
    # below it at 4 (0): crossing at exactly 3
    assert got["left"] == pytest.approx(-3.0)
    assert got["right"] == pytest.approx(3.0)
    assert got["fwhm"] == pytest.approx(6.0)
    assert got["centre"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the direct-beam offset, found from the data
# ---------------------------------------------------------------------------

def test_shift_detector_puts_the_direct_beam_at_the_origin():
    # A detector whose direct beam sits at (x0, y0) instead of the origin: the
    # specular spot then shows at x0, y0 + L tan(2 alpha_i).
    x0, y0 = 0.4, 1.5
    df = make_frame(5, 60, 0.1, -0.2, -1.0)
    shifted = gisaxs.shift_detector(df, x0, y0)
    np.testing.assert_allclose(shifted["x"], df["x"] - x0)
    np.testing.assert_allclose(shifted["y"], df["y"] - y0)
    assert shifted.attrs == df.attrs
    assert np.allclose(df["y"].iloc[:3], -1.0)                  # input untouched
    np.testing.assert_allclose(gisaxs.shift_detector(df, y0=y0)["x"], df["x"])
    # the offset follows from the spot, and removing it puts the spot at
    # phi_f = 0, alpha_f = alpha_i
    x_spot, y_spot = x0, gisaxs.specular_y(TOY) + y0
    assert gisaxs.direct_beam_offset(x_spot, y_spot, TOY) == pytest.approx((x0, y0))
    before = gisaxs.pixel(x_spot, y_spot, TOY)
    assert before["alpha_f"] > TOY.alpha_i and before["phi_f"] > 0
    after = gisaxs.pixel(x_spot - x0, y_spot - y0, TOY)
    assert after["alpha_f"] == pytest.approx(TOY.alpha_i, abs=1e-12)
    assert after["phi_f"] == pytest.approx(0.0, abs=1e-12)
    # the whole conversion is the usual one on the shifted frame
    got = gisaxs.convert(shifted, TOY)
    want = gisaxs.pixel(df["x"].to_numpy() - x0, df["y"].to_numpy() - y0, TOY)
    for name in ("phi_f", "alpha_f", "qy", "qz", "kfx"):
        np.testing.assert_allclose(got[name], want[name])


def offset_frame(y_db=1.2, x_db=0.0, beam=True, clip=False, spot_dx=0.0,
                 L=1600.0, alpha_i=0.3, pitch=0.075):
    """A raw frame whose direct beam sits at (x_db, y_db) instead of the
    origin, with the specular spot at (x_db + spot_dx, y_db + L tan 2 alpha_i).
    clip cuts the lower half of the direct beam off, as a beamstop edge does;
    spot_dx moves the spot sideways from the beam, as a sample tilted about
    the beam would."""
    geom = Geometry(L=L, alpha_i=alpha_i, wavelength=0.15418)
    x_spot, y_spot = x_db + spot_dx, y_db + gisaxs.specular_y(geom)
    df = make_frame(81, 360, pitch, -3.0, -2.0)
    X, Y = df["x"].to_numpy(), df["y"].to_numpy()
    counts = 2.0 + 5000.0 * np.exp(-0.5 * ((X - x_spot) / 0.2) ** 2
                                   - 0.5 * ((Y - y_spot) / 0.25) ** 2)
    if beam:
        direct = 3000.0 * np.exp(-0.5 * ((X - x_db) / 0.2) ** 2
                                 - 0.5 * ((Y - y_db) / 0.3) ** 2)
        if clip:
            direct[Y < y_db + 0.05] = 0.0
            counts[Y < y_db + 0.05] = 0.0          # the beamstop's shadow
        counts = counts + direct
    df["intensity"] = counts
    return df, geom


def test_offset_from_a_clean_direct_beam():
    df, geom = offset_frame(y_db=1.2, x_db=0.3)
    got = spot.find_offset(df, geom)
    d = got["direct"]
    assert got["method_x"] == got["method_y"] == "direct" and d["ok_x"] and d["ok_y"]
    assert got["x0"] == pytest.approx(0.3, abs=0.5 * 0.075)
    assert got["y0"] == pytest.approx(1.2, abs=0.5 * 0.075)
    assert abs(d["error_x"]) <= 1 and abs(d["error_y"]) <= 1
    assert got["phi_f_spot"] == pytest.approx(0.0, abs=got["pixel_phi_deg"])
    assert got["alpha_f_spot"] == pytest.approx(geom.alpha_i, abs=got["pixel_alpha_deg"])
    assert got["spot"]["pixels_off_x"] == pytest.approx(0.3 / 0.075, abs=0.5)
    assert got["spot"]["pixels_off_y"] == pytest.approx(1.2 / 0.075, abs=0.5)


def test_offset_falls_back_to_the_spot_when_the_direct_beam_is_clipped():
    df, geom = offset_frame(y_db=1.2, x_db=-0.2, clip=True)
    got = spot.find_offset(df, geom)
    d = got["direct"]
    # the clipped beam's centroid sits too high: the spot then misses alpha_i;
    # sideways the beam is whole, and agrees with the spot
    assert d is not None and d["ok_x"] and not d["ok_y"]
    assert d["y0"] > 1.2 + 0.075
    assert got["method_x"] == "direct" and got["method_y"] == "specular"
    assert got["x0"] == pytest.approx(d["x0"])
    assert got["x0"] == pytest.approx(-0.2, abs=0.5 * 0.075)
    assert got["y0"] == pytest.approx(got["specular"]["y0"])
    assert got["y0"] == pytest.approx(1.2, abs=0.5 * 0.075)
    assert got["alpha_f_spot"] == pytest.approx(geom.alpha_i, abs=1e-4)


def test_offset_puts_a_sideways_spot_at_phi_f_zero():
    # The spot sits 0.3 mm (4 pixels) to the side of the direct beam: measured
    # from the beam it misses phi_f = 0, so x0 comes from the spot itself.
    df, geom = offset_frame(y_db=1.2, x_db=0.1, spot_dx=0.3)
    got = spot.find_offset(df, geom)
    d = got["direct"]
    assert not d["ok_x"] and d["ok_y"] and abs(d["error_x"]) > 1
    assert got["method_x"] == "specular" and got["method_y"] == "direct"
    assert got["x0"] == pytest.approx(got["spot"]["x"])
    assert got["x0"] == pytest.approx(0.4, abs=0.5 * 0.075)
    assert got["phi_f_spot"] == pytest.approx(0.0, abs=1e-9)
    assert got["specular"]["phi_f_spot"] == pytest.approx(0.0, abs=1e-9)
    assert got["specular"]["alpha_f_spot"] == pytest.approx(geom.alpha_i, abs=1e-9)


def test_offset_from_the_spot_when_no_direct_beam_shows():
    df, geom = offset_frame(y_db=1.2, x_db=0.3, beam=False)
    got = spot.find_offset(df, geom)
    assert got["direct"] is None
    assert got["method_x"] == got["method_y"] == "specular"
    assert got["x0"] == pytest.approx(0.3, abs=0.5 * 0.075)
    assert got["y0"] == pytest.approx(1.2, abs=0.5 * 0.075)


def reference_frame(x_d=0.3, y_d=1.2, half_cut=False, sigma=0.25, n=61, pitch=0.075):
    """A small reference frame (alpha_i = 0, beamstop out): one Gaussian beam
    at (x_d, y_d), or its upper half only when the sample half-cuts it."""
    df = make_frame(n, n, pitch, x_d - (n // 2) * pitch + 0.013, y_d - (n // 2) * pitch + 0.021)
    X, Y = df["x"].to_numpy(), df["y"].to_numpy()
    counts = 1.0 + 20000.0 * np.exp(-0.5 * (((X - x_d) / sigma) ** 2 + ((Y - y_d) / sigma) ** 2))
    if half_cut:
        counts[Y < y_d] = 0.0
    df["intensity"] = counts
    return df


def test_reference_beam_centre_or_lower_edge():
    full = spot.find_reference_beam(reference_frame(0.3, 1.2))
    assert not full["half_cut"]
    assert full["x0"] == pytest.approx(0.3, abs=0.5 * 0.075)
    assert full["y0"] == pytest.approx(1.2, abs=0.5 * 0.075)
    # half-cut by the sample: only the upper half shows; its centroid sits
    # ~0.8 sigma too high, its lower edge marks the surface
    cut = spot.find_reference_beam(reference_frame(0.3, 1.2, half_cut=True))
    assert cut["half_cut"]
    assert cut["y_centroid"] > 1.2 + 0.1
    assert cut["y0"] == pytest.approx(1.2, abs=0.5 * 0.075)
    assert cut["x0"] == pytest.approx(0.3, abs=0.5 * 0.075)
    # a hot pixel elsewhere does not win over the beam
    df = reference_frame(0.3, 1.2)
    df.loc[0, "intensity"] = 1e5           # brighter than the beam's peak pixel
    assert spot.find_reference_beam(df)["y0"] == pytest.approx(1.2, abs=0.5 * 0.075)


def test_alpha_i_from_specular_inverts_the_spot_height():
    geom = Geometry(L=1600.0, alpha_i=0.4123, wavelength=0.15418)
    y = gisaxs.specular_y(geom)
    assert gisaxs.alpha_i_from_specular(0.0, y, geom.L) == pytest.approx(0.4123, abs=1e-12)
    # sideways the spot is still 2 alpha_i from the beam
    assert gisaxs.alpha_i_from_specular(y, 0.0, geom.L) == pytest.approx(0.4123, abs=1e-12)


def test_offset_needs_the_specular_spot():
    df, geom = offset_frame()
    df["intensity"] = 2.0                         # nothing to find
    with pytest.raises(spot.SpotError):
        spot.find_offset(df, geom)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

EXPORT_META = dict(dat_name="a.dat", pro_name="a.pro", L=86.0, alpha_i=1.0,
                   wavelength=0.15406, k=TOY.k, pitch_x=0.5, pitch_y=0.5,
                   nx=5, ny=4, rows_dropped=2)


@pytest.mark.parametrize("q_unit, suffix, factor", [("nm-1", "nm-1", 1.0),
                                                     ("A-1", "A-1", 0.1)])
@pytest.mark.parametrize("gz", [False, True])
def test_csv_round_trip(tmp_path, q_unit, suffix, factor, gz):
    counts = np.arange(30, dtype=float).reshape(6, 5)
    counts[0, 1] = np.nan
    df = gisaxs.convert(make_frame(5, 6, 0.5, -1.0, 0.0, counts), TOY)
    path = tmp_path / ("t.csv.gz" if gz else "t.csv")
    header = gisaxs.export_header(EXPORT_META, q_unit)
    gisaxs.write_csv(df, path, header, q_unit, gzip_level=6 if gz else None)

    if gz:
        with gzip.open(path, "rt") as f:
            assert f.readline().startswith("# ")
    back = pd.read_csv(path, comment="#")
    assert list(back.columns) == [
        "x_mm", "y_mm", "counts", "variance", "phi_f_deg", "alpha_f_deg",
        f"kfx_{suffix}", f"kfy_{suffix}", f"kfz_{suffix}",
        f"qx_{suffix}", f"qy_{suffix}", f"qz_{suffix}"]
    assert len(back) == len(df)
    assert np.isnan(back["counts"][1])
    np.testing.assert_allclose(back["qz_" + suffix], df["qz"] * factor, rtol=1e-7)
    np.testing.assert_allclose(back["alpha_f_deg"], df["alpha_f"], rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(back["counts"], df["intensity"], equal_nan=True)


def test_csv_header_records_the_setup():
    lines = gisaxs.export_header({**EXPORT_META, "mask": "specular rod"}, "nm-1")
    text = "\n".join(lines)
    for needle in ("a.dat", "a.pro", "L = |N|", "alpha_i", "lambda", "k = 2 pi",
                   "Pixel pitch", "Grid: 4 rows x 5 columns", "horizon dropped: 2",
                   "unit: nm^-1", "Written:", "Mask: specular rod"):
        assert needle in text, needle
    assert all(line.startswith("#") for line in lines)


# ---------------------------------------------------------------------------
# figures (smoke tests: the files are produced and are what they claim)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["publication", "presentation"])
def test_figures_png_and_pdf(preset):
    figures = pytest.importorskip("figures")
    counts = np.arange(30, dtype=float).reshape(6, 5)
    counts[0, 0] = np.nan
    counts[1, :] = 0
    df = gisaxs.convert(make_frame(5, 6, 0.5, -1.0, 3.0, counts), TOY)
    png = figures.intensity_map(df["qy"], df["qz"], df["intensity"],
                                figures.axis_label("qy"), figures.axis_label("qz"),
                                preset=preset, fmt="png", pixel=(0.01, 0.01))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pdf = figures.intensity_map(df["phi_f"], df["alpha_f"], df["intensity"],
                                figures.axis_label("phi_f"),
                                figures.axis_label("alpha_f"),
                                log=False, cmap="cividis", zero_colour="white",
                                preset=preset, columns=2, fmt="pdf")
    assert pdf[:5] == b"%PDF-"
    prof = gisaxs.row_profile(df, "qz")
    out = figures.profile(prof.index, prof.to_numpy(),
                          figures.axis_label("qz", "A-1"), preset=preset, fmt="pdf")
    assert out[:5] == b"%PDF-"


def _page_ratio(data):
    """width / height of a PNG (pixels) or PDF (MediaBox)."""
    if data[:4] == b"\x89PNG":
        w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    else:
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", data)
        w, h = float(box.group(1)), float(box.group(2))
    return w / h


@pytest.mark.parametrize("preset,columns", [("publication", 1), ("publication", 2), ("presentation", 1)])
def test_figures_have_a4_proportions(preset, columns):
    figures = pytest.importorskip("figures")
    df = gisaxs.convert(make_frame(5, 6, 0.5, -1.0, 3.0, np.arange(30.0).reshape(6, 5)), TOY)
    a4 = 297.0 / 210.0
    for fmt in ("png", "pdf"):
        maps = [figures.intensity_map(df[xc], df[yc], df["intensity"], "x", "y", preset=preset,
                                      columns=columns, fmt=fmt, pixel=(0.3, 0.3))
                for xc, yc in (("phi_f", "alpha_f"), ("x", "y"))]
        prof = gisaxs.row_profile(df, "alpha_f")
        line = figures.profile(prof.index, prof.to_numpy(), "a", preset=preset, columns=columns, fmt=fmt)
        for data in maps + [line]:
            assert _page_ratio(data) == pytest.approx(a4, rel=0.01)
    width, height = figures.figure_size(preset, columns)
    assert width / height == pytest.approx(a4)


def test_figure_axis_limits_and_faded_profile():
    figures = pytest.importorskip("figures")
    df = gisaxs.convert(make_frame(5, 6, 0.5, -1.0, 3.0, np.arange(30.0).reshape(6, 5)), TOY)
    args = (df["phi_f"], df["alpha_f"], df["intensity"], "x", "y")
    png = figures.intensity_map(*args, pixel=(0.3, 0.3), xlim=[None, 0.5], ylim=[1.0, None])
    assert png[:4] == b"\x89PNG"
    with pytest.raises(figures.FigureError, match="smaller"):
        figures.intensity_map(*args, xlim=[1.0, 0.5])
    with pytest.raises(figures.FigureError, match="no pixels"):
        figures.intensity_map(*args, xlim=[50, 60])

    prof = gisaxs.row_profile(df, "alpha_f")
    smooth = gisaxs.smooth_profile(prof, 5).to_numpy()
    pdf = figures.profile(prof.index, prof.to_numpy(), "a", smoothed=smooth,
                          xlim=[None, prof.index[3]], fmt="pdf")
    assert pdf[:5] == b"%PDF-"
    with pytest.raises(figures.FigureError, match="fewer than 2"):
        figures.profile(prof.index, prof.to_numpy(), "a", xlim=[100, 200])


def test_profile_labels(monkeypatch):
    figures = pytest.importorskip("figures")
    assert figures.mark_label("alpha_f", "0.280") == r"$\alpha_f$ = 0.280°"
    assert figures.mark_label("qz", "-0.5", "A-1") == "$q_z$ = −0.5 Å$^{-\\mathrm{1}}$"
    # each text goes towards the wider gap: as in the user's figure, the
    # label at 0.280 sits left of its line and the one at 0.341 right of it
    assert figures.mark_sides([0.341, 0.280], 0.0, 0.6) == ["right", "left"]
    assert figures.mark_sides([0.1], 0.0, 0.6) == ["right"]
    assert figures.mark_sides([0.5], 0.0, 0.6) == ["left"]

    captured = {}

    def keep(fig, fmt):
        captured["ax"] = fig.axes[0]
        return b"figure"

    monkeypatch.setattr(figures, "_to_bytes", keep)
    x = np.linspace(0.1, 0.6, 51)
    y = 10.0 + 1000.0 * np.exp(-((x - 0.3) / 0.05) ** 2)
    marks = [{"x": 0.341, "label": figures.mark_label("alpha_f", "0.341"), "style": 1},
             {"x": 0.280, "label": figures.mark_label("alpha_f", "0.280"), "style": 0},
             {"x": 0.9, "label": "outside", "style": 2}]
    figures.profile(x, y, "a", marks=marks)
    ax = captured["ax"]
    try:
        texts = {t.get_text(): t for t in ax.texts}
        assert set(texts) == {r"$\alpha_f$ = 0.341°", r"$\alpha_f$ = 0.280°"}   # 0.9 left out
        left, right = texts[r"$\alpha_f$ = 0.280°"], texts[r"$\alpha_f$ = 0.341°"]
        assert left.get_ha() == "right" and right.get_ha() == "left"
        assert left.get_color() == figures.MARK_STYLES[0]["color"]
        assert right.get_color() == figures.MARK_STYLES[1]["color"]
        lines = sorted(float(l.get_xdata()[0]) for l in ax.lines if len(set(l.get_xdata())) == 1)
        assert lines == pytest.approx([0.280, 0.341])
        # the data keep to the bottom of the log axis, below the labels
        lo, hi = np.log10(ax.get_ylim())
        assert (np.log10(y.max()) - lo) / (hi - lo) <= 1.0 - figures.MARK_HEADROOM + 1e-9
    finally:
        import matplotlib.pyplot as plt
        plt.close("all")


def test_screen_colours_match_plotstyle():
    figures = pytest.importorskip("figures")
    colours = figures.screen_colours()
    assert set(colours["scales"]) == set(figures.COLORMAPS)
    assert colours["under"]["black"] == "#000000"
    assert colours["under"]["white"] == "#ffffff"
    assert next(iter(figures.ZERO_COLOURS)) == "white"   # maps default to white
    assert colours["bad"] == "#595959"          # plotstyle's bad colour "0.35"


# ---------------------------------------------------------------------------
# the real test files
# ---------------------------------------------------------------------------

needs_pro = pytest.mark.skipif(not REAL_PRO.exists(), reason="no real .pro file")
needs_both = pytest.mark.skipif(not (REAL_PRO.exists() and REAL_DAT.exists()),
                                reason="no real .dat/.pro files")


@needs_pro
def test_real_pro_geometry():
    pro = read_pro.read_pro(REAL_PRO)
    geom = Geometry.from_pro(pro)
    assert geom.L == pytest.approx(1600.0, abs=0.01)
    assert geom.alpha_i == pytest.approx(0.299991444658313, abs=1e-15)
    assert geom.wavelength == pytest.approx(0.15418)
    summary = gisaxs.instrument_summary(pro)
    assert summary["nx"] == 1028 and summary["ny"] == 1062
    assert summary["pitch_x"] == pytest.approx(0.075)


@needs_pro
def test_real_pro_pixel():
    geom = Geometry.from_pro(read_pro.read_pro(REAL_PRO))
    got = gisaxs.pixel(0.0, 20.3, geom)
    assert got["alpha_f"] == pytest.approx(0.426911, abs=TOL)
    assert got["qz"] == pytest.approx(0.517014, abs=TOL)


@pytest.fixture(scope="module")
def real_frames():
    pro = read_pro.read_pro(REAL_PRO)
    geom = Geometry.from_pro(pro)
    raw = gisaxs.read_dat(REAL_DAT, nx=pro["nx"], ny=pro["ny"])
    cut = gisaxs.horizon_cut(gisaxs.convert(raw, geom), geom)
    return pro, geom, raw, cut


@needs_both
def test_real_pipeline_keeps_925_rows(real_frames):
    pro, geom, raw, cut = real_frames
    assert len(raw) == 1062 * 1028
    assert raw.attrs["order"].startswith("row-major")
    assert cut.attrs["ny"] == 925 and cut.attrs["nx"] == 1028
    assert len(cut) == 925 * 1028 == 950_900
    assert np.allclose(np.sqrt(cut["kfx"] ** 2 + cut["kfy"] ** 2 + cut["kfz"] ** 2),
                       geom.k)


@needs_both
def test_real_rod_mask(real_frames):
    _, _, _, cut = real_frames
    rod = mask.find_rod(cut)
    masked, n = mask.mask_rod(cut, rod["centre"], mask.DEFAULT_N_SIGMA * rod["sigma"])
    assert n > 0 and n % cut.attrs["ny"] == 0          # whole columns
    assert abs(rod["centre"]) < 0.05
    profile = gisaxs.column_mean_profile(masked, "phi_f")
    assert profile.isna().sum() == n // cut.attrs["ny"]  # gaps in the profile


@needs_both
def test_bridge_end_to_end(tmp_path):
    """The command layer the browser uses, run in CPython on copies of the
    real files."""
    bridge = pytest.importorskip("bridge")
    shutil.copy(REAL_PRO, tmp_path / "p.pro")
    shutil.copy(REAL_DAT, tmp_path / "d.dat")

    def call(cmd, **kw):
        out = bridge.dispatch(cmd, json.dumps(kw))
        meta = json.loads(out["meta"])
        assert "error" not in meta, meta
        return meta, out

    call("init")
    meta, _ = call("load_pro", path=str(tmp_path / "p.pro"), name="p.pro")
    assert meta["summary"]["L"] == pytest.approx(1600.0, abs=0.01)
    meta, _ = call("load_dat", path=str(tmp_path / "d.dat"), name="d.dat")
    assert meta["pixels"] == 1_091_736 and meta["dead"] == 13
    meta, _ = call("convert", correct_offset=False)
    assert meta["steps"]["8"]["pixels"] == 950_900
    assert meta["steps"]["6"]["max_rel_dev"] < 1e-12

    meta, out = call("map_data", kind="qz_qy", q_unit="A-1")
    assert out["arrays"]["x"].dtype == np.float32 and len(out["arrays"]["c"]) == 950_900
    meta, out = call("profile_data", kind="qy")
    assert len(out["arrays"]["x"]) == 1028
    assert meta["smooth"] == gisaxs.DEFAULT_SMOOTHING == 7
    raw = gisaxs.column_mean_profile(bridge.S.data, "qy")
    np.testing.assert_allclose(out["arrays"]["y"],
                               raw.rolling(7, center=True, min_periods=1).mean())
    _, out = call("profile_data", kind="qy", smooth=1)
    np.testing.assert_allclose(out["arrays"]["y"], raw.to_numpy())
    _, out = call("profile_data", kind="qy", smooth=11)         # any odd number
    np.testing.assert_allclose(out["arrays"]["y"],
                               raw.rolling(11, center=True, min_periods=1).mean())
    bad = json.loads(bridge.dispatch("profile_data", json.dumps({"kind": "qy", "smooth": 6}))["meta"])
    assert bad.get("user_error") and "odd whole number" in bad["error"]

    meta, _ = call("apply_mask", kind="rod", n_sigma=4)
    assert meta["n_masked"] > 0
    meta, out = call("profile_data", kind="phi_f")
    unsmoothed, _ = call("profile_data", kind="phi_f", smooth=1)
    assert meta["n_gaps"] == unsmoothed["n_gaps"] > 0     # the rod stays a gap

    # without the offset correction the spot is ~47 pixels above where alpha_i
    # predicts it; the search starts at the brightest row along phi_f ~ 0, so
    # it is found with or without following the peak
    fixed, _ = call("apply_mask", kind="spot", n_sigma=4, window=15, follow_peak=False)
    meta, _ = call("apply_mask", kind="spot", n_sigma=4, window=15)       # follows
    assert meta["scan_column"] and meta["n_masked"] > 0
    for m in (fixed, meta):
        assert m["alpha0"] == pytest.approx(0.426, abs=0.002)

    meta, out = call("export_table", which="masked", fmt="csv.gz", q_unit="nm-1")
    blob = b"".join(bytes(piece) for piece in out["stream"])
    back = pd.read_csv(io.BytesIO(gzip.decompress(blob)), comment="#")
    assert len(back) == 950_900 and back.columns[0] == "x_mm"
    assert back["counts"].isna().sum() > 13          # dead pixels + the spot
    meta, out = call("export_figure", kind="qy", fmt="png", profile=True)
    assert bytes(out["arrays"]["file"][:4]) == b"\x89PNG"
    meta, out = call("export_figure", kind="alpha_f", fmt="pdf", profile=True, smooth=5)
    assert bytes(out["arrays"]["file"][:5]) == b"%PDF-"

    # profiles carry the measured curve too; figures take axis limits
    _, out = call("profile_data", kind="alpha_f")
    assert set(out["arrays"]) == {"x", "y", "raw"}
    _, out = call("export_figure", kind="alpha_f", fmt="png", profile=True, xlim=[0.0, 0.5])
    assert bytes(out["arrays"]["file"][:4]) == b"\x89PNG"
    _, out = call("export_figure", kind="alpha_f", fmt="pdf", profile=True,
                  marks=[{"x": 0.28, "text": "0.280", "style": 0},
                         {"x": 0.341, "text": "0.341", "style": 1}])
    assert bytes(out["arrays"]["file"][:5]) == b"%PDF-"
    labels = bridge._marks("qz", [{"x": 0.3, "text": "$\\oops"}, {"x": -1, "text": "-1.50"}], "nm-1")
    assert [m["label"] for m in labels] == [r"$q_z$ = 0.3 nm$^{-\mathrm{1}}$",
                                            "$q_z$ = −1.50 nm$^{-\\mathrm{1}}$"]
    bad = json.loads(bridge.dispatch("export_figure", json.dumps(
        {"kind": "alpha_f", "fmt": "png", "profile": True, "marks": [{"x": "abc"}]}))["meta"])
    assert bad.get("user_error") and "number" in bad["error"]
    _, out = call("export_figure", kind="af_pf", fmt="png", xlim=[-0.2, 0.2], ylim=[0.2, 0.6])
    assert bytes(out["arrays"]["file"][:4]) == b"\x89PNG"
    bad = json.loads(bridge.dispatch("export_figure", json.dumps(
        {"kind": "qz_qy", "fmt": "png", "xlim": [1.0, 0.0]}))["meta"])
    assert bad.get("user_error") and "smaller" in bad["error"]

    # The conversion finds the direct-beam offset by itself: the direct beam
    # shows at (-0.05, 3.76) mm but is clipped by the beamstop, and measured
    # from it the spot misses both alpha_i and phi_f = 0 by ~3 pixels; the spot
    # gives x0 and y0 instead
    pre, _ = call("convert")                                # corrects by default
    pre = pre["offset"]
    d = pre["direct"]
    assert d is not None and d["y0"] == pytest.approx(3.76, abs=0.05)
    assert d["x0"] == pytest.approx(-0.05, abs=0.05)
    assert not d["ok_y"] and abs(d["error_y"]) > 1
    assert not d["ok_x"] and abs(d["error_x"]) > 1
    assert pre["method_x"] == pre["method_y"] == "specular"
    assert pre["x0"] == pytest.approx(pre["spot"]["x"])
    assert pre["x0"] == pytest.approx(0.18, abs=0.01)
    assert pre["y0"] == pytest.approx(pre["spot"]["y"] - pre["spot"]["predicted_y"])
    assert pre["y0"] == pytest.approx(3.52, abs=0.01)
    assert pre["spot"]["alpha_f"] == pytest.approx(0.426, abs=1e-3)       # before
    assert pre["spot"]["phi_f"] == pytest.approx(0.0064, abs=1e-3)
    assert pre["alpha_f_spot"] == pytest.approx(0.299991444658313, abs=1e-9)  # after
    assert pre["phi_f_spot"] == pytest.approx(0.0, abs=1e-12)

    meta, _ = call("convert")                               # the same again
    assert meta["y0"] == pytest.approx(pre["y0"]) and meta["x0"] == pytest.approx(pre["x0"])
    assert meta["offset"]["method_y"] == "specular"
    assert meta["offset_error"] is None
    assert meta["alpha_i"] == pytest.approx(0.299991444658313)   # alpha_i untouched
    assert meta["steps"]["8"]["rows_kept"] == 878
    # the spot is now where alpha_i predicts it: the default window finds it
    meta, _ = call("apply_mask", kind="spot", n_sigma=4, window=15)
    assert meta["alpha0"] == pytest.approx(0.299991444658313, abs=meta["pixel_alpha"])
    assert meta["phi0"] == pytest.approx(0.0, abs=meta["pixel_phi"])
    header = gisaxs.export_header(bridge._export_meta(bridge.S.cut), "nm-1")
    assert any(line.startswith("# Pixels shifted: x_mm = x - 0.1799 mm and y_mm = y - 3.5188 mm")
               and "from the specular spot" in line for line in header)

    meta, _ = call("convert", correct_offset=False)
    assert meta["x0"] == meta["y0"] == 0 and meta["offset"] is None
    assert meta["steps"]["8"]["rows_kept"] == 925

    # no specular spot to be found: converted as it is, with the reason
    saved = bridge.S.raw
    bridge.S.raw = saved.assign(intensity=2.0)
    try:
        meta, _ = call("convert")
        assert meta["offset"] is None and meta["x0"] == meta["y0"] == 0
        assert "could not find the specular spot" in meta["offset_error"]
        assert meta["steps"]["8"]["rows_kept"] == 925
    finally:
        bridge.S.raw = saved


@needs_both
def test_bridge_with_a_reference_frame(tmp_path, monkeypatch):
    bridge = pytest.importorskip("bridge")

    def call(cmd, **kw):
        out = bridge.dispatch(cmd, json.dumps(kw))
        meta = json.loads(out["meta"])
        assert "error" not in meta, meta.get("error")
        return meta

    monkeypatch.setattr(bridge.S, "ref", None)       # restored afterwards
    call("load_pro", path=str(REAL_PRO), name=REAL_PRO.name)
    call("load_dat", path=str(REAL_DAT), name=REAL_DAT.name)
    spot_x, spot_y = 0.180, 20.27          # the specular spot in frame-000, raw

    # a reference whose beam is where the frame's leaked direct beam shows
    write_dat(tmp_path / "ref.dat", reference_frame(-0.049, 3.758))
    ref = call("load_ref", path=str(tmp_path / "ref.dat"), name="ref.dat")
    assert not ref["half_cut"] and ref["y0"] == pytest.approx(3.758, abs=0.04)
    meta = call("convert", correct_offset=False)
    cal = meta["reference"]
    assert meta["x0"] == pytest.approx(ref["x0"]) and meta["y0"] == pytest.approx(ref["y0"])
    assert cal["spot"]["x"] == pytest.approx(spot_x - ref["x0"], abs=0.04)
    assert cal["spot"]["y"] == pytest.approx(spot_y - ref["y0"], abs=0.04)
    want = gisaxs.alpha_i_from_specular(cal["spot"]["x"], cal["spot"]["y"], bridge.S.geom.L)
    assert meta["alpha_i"] == pytest.approx(want) and meta["alpha_i"] == pytest.approx(0.2957, abs=3e-4)
    assert cal["alpha_i_pro"] == pytest.approx(0.299991444658313)
    assert cal["delta"] == pytest.approx(meta["alpha_i"] - cal["alpha_i_pro"])
    assert cal["spot"]["alpha_f"] == pytest.approx(meta["alpha_i"], abs=1e-4)
    header = gisaxs.export_header(bridge._export_meta(bridge.S.cut), "nm-1")
    assert any(l.startswith("# Pixels shifted: x_mm = x - ") and "reference frame ref.dat" in l
               for l in header)
    assert any(l.startswith("# Incidence angle measured: alpha_i = 0.295") for l in header)
    assert any(l.startswith("# Incidence angle alpha_i = 0.295") for l in header)
    # the spot mask now looks where the measured alpha_i puts the spot
    m = call("apply_mask", kind="spot", n_sigma=4, window=15)
    assert m["alpha0"] == pytest.approx(meta["alpha_i"], abs=m["pixel_alpha"])

    # the same beam half-cut at the spot-only origin (3.519 mm): the lower
    # edge is taken, and alpha_i comes back to the .pro value
    write_dat(tmp_path / "ref.dat", reference_frame(0.18, 3.519, half_cut=True))
    ref = call("load_ref", path=str(tmp_path / "ref.dat"), name="half.dat")
    assert ref["half_cut"] and ref["y0"] == pytest.approx(3.519, abs=0.04)
    assert bridge.S.cut is None                      # the old conversion is gone
    meta = call("convert", correct_offset=False)
    assert meta["alpha_i"] == pytest.approx(0.3000, abs=5e-4)

    # the reference stays for the next frame of the series; clear_ref drops it
    call("load_pro", path=str(REAL_PRO), name=REAL_PRO.name)
    call("load_dat", path=str(REAL_DAT), name=REAL_DAT.name)
    assert call("convert", correct_offset=False)["reference"]["name"] == "half.dat"
    call("clear_ref")
    meta = call("convert", correct_offset=False)
    assert meta["reference"] is None and meta["alpha_i"] == pytest.approx(0.299991444658313)
    assert meta["x0"] == meta["y0"] == 0


@needs_both
def test_alpha_check_and_correction(monkeypatch):
    bridge = pytest.importorskip("bridge")

    def call(cmd, **kw):
        meta = json.loads(bridge.dispatch(cmd, json.dumps(kw))["meta"])
        assert "error" not in meta, meta.get("error")
        return meta

    monkeypatch.setattr(bridge.S, "ref", None)
    call("load_pro", path=str(REAL_PRO), name=REAL_PRO.name)
    call("load_dat", path=str(REAL_DAT), name=REAL_DAT.name)
    c = call("check_alpha")
    # frame-000: the spot is 47 pixels above alpha_i = 0.3 deg
    assert c["needs_correction"] and c["pixels_off"] > 40
    assert c["spot"]["alpha_f"] == pytest.approx(0.426, abs=1e-3)
    L = bridge.S.geom.L
    assert c["alpha_i_corrected"] == pytest.approx(
        gisaxs.alpha_i_from_specular(c["spot"]["x"], c["spot"]["y"], L))
    # the spot is 2 alpha_i above the direct beam: alpha_i + delta -> alpha_i + delta / 2
    assert c["alpha_i_corrected"] - c["alpha_i"] == pytest.approx(c["delta"] / 2, rel=0.02)
    assert c["alpha_i_corrected"] == pytest.approx(0.3630, abs=5e-4)

    m = call("convert", correct_offset=False, correct_alpha=True)       # "Yes"
    assert m["alpha_i"] == pytest.approx(c["alpha_i_corrected"])
    assert m["alpha_fix"]["alpha_i_pro"] == pytest.approx(0.299991444658313)
    assert m["x0"] == m["y0"] == 0                                      # pixels not moved
    s = call("apply_mask", kind="spot", n_sigma=4, window=15)
    assert s["alpha0"] == pytest.approx(m["alpha_i"], abs=s["pixel_alpha"])   # spot now at alpha_i
    header = gisaxs.export_header(bridge._export_meta(bridge.S.cut), "nm-1")
    assert any(l.startswith("# Incidence angle alpha_i = 0.363") for l in header)
    assert any(l.startswith("# Incidence angle measured: alpha_i = 0.363") and "0.299991 deg" in l
               for l in header)

    m = call("convert", correct_offset=False)                           # "No"
    assert m["alpha_i"] == pytest.approx(0.299991444658313) and m["alpha_fix"] is None
    header = gisaxs.export_header(bridge._export_meta(bridge.S.cut), "nm-1")
    assert not any(l.startswith("# Incidence angle measured") for l in header)

    # a spot where alpha_i puts it: no correction offered
    geom = bridge.S.geom
    raw = make_frame(81, 400, 0.075, -3.0, -2.0)
    X, Y = raw["x"].to_numpy(), raw["y"].to_numpy()
    raw["intensity"] = 2.0 + 5000.0 * np.exp(-0.5 * (X / 0.2) ** 2
                                             - 0.5 * ((Y - gisaxs.specular_y(geom)) / 0.25) ** 2)
    monkeypatch.setattr(bridge.S, "raw", raw)
    c = call("check_alpha")
    assert not c["needs_correction"] and abs(c["pixels_off"]) <= 1
