"""Tests for the GISAXS conversion, masks, profiles and export (plain CPython).

    pip install numpy pandas matplotlib pytest
    pytest tests/
"""

import gzip
import io
import json
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


@pytest.mark.parametrize("window", [5, 7])
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


def test_smoothing_window_one_is_identity_and_even_is_refused():
    profile = pd.Series([1.0, np.nan, 3.0, 4.0])
    pd.testing.assert_series_equal(gisaxs.smooth_profile(profile, 1), profile)
    with pytest.raises(ValueError, match="odd"):
        gisaxs.smooth_profile(profile, 6)


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
    meta, _ = call("convert")
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

    meta, _ = call("apply_mask", kind="rod", n_sigma=4)
    assert meta["n_masked"] > 0
    meta, out = call("profile_data", kind="phi_f")
    unsmoothed, _ = call("profile_data", kind="phi_f", smooth=1)
    assert meta["n_gaps"] == unsmoothed["n_gaps"] > 0     # the rod stays a gap

    # the spot is not where alpha_i predicts in this file (see README)
    err = json.loads(bridge.dispatch("apply_mask", json.dumps(
        {"kind": "spot", "n_sigma": 4, "window": 15}))["meta"])
    assert "error" in err and "follow the peak" in err["error"]
    meta, _ = call("apply_mask", kind="spot", n_sigma=4, window=15, follow_peak=True)
    assert meta["n_masked"] > 0

    meta, out = call("export_table", which="masked", fmt="csv.gz", q_unit="nm-1")
    blob = b"".join(bytes(piece) for piece in out["stream"])
    back = pd.read_csv(io.BytesIO(gzip.decompress(blob)), comment="#")
    assert len(back) == 950_900 and back.columns[0] == "x_mm"
    assert back["counts"].isna().sum() > 13          # dead pixels + the spot
    meta, out = call("export_figure", kind="qy", fmt="png", profile=True)
    assert bytes(out["arrays"]["file"][:4]) == b"\x89PNG"
    meta, out = call("export_figure", kind="alpha_f", fmt="pdf", profile=True, smooth=5)
    assert bytes(out["arrays"]["file"][:5]) == b"%PDF-"
