# Build a browser-based GISAXS analysis app (Pyodide + Plotly, hosted on GitHub Pages)

## 0. Your task

Build a **static web application** in this folder that lets a user upload a GISAXS
measurement (`.dat`) and its instrument file (`.pro`), converts every detector pixel
to exit angles, final wavevector and scattering vector, masks the specular rod or
spot, and plots 2D maps and 1D profiles that can be downloaded as PNG/PDF.

Hard constraints:

- **No server.** The app is hosted on **GitHub Pages**: static files only, no build
  step required, all paths **relative** (the site lives under `/<repo-name>/`).
- **All analysis runs in the browser in Python via Pyodide** (numpy, pandas,
  matplotlib). Plotly.js is used for the interactive on-screen plots.
- **Three Python files are provided by me and must be used unchanged**:
  `read_pro.py`, `mask.py`, `plotstyle.py`. Put them in `py/`. Do not edit them;
  write any extra logic in new modules.
- Add an empty `.nojekyll` file at the repo root so GitHub Pages serves every file
  as-is.
- **A geometry schematic is provided by me** as `assets/geometry.svg` (plus
  `geometry.png`, `geometry.pdf` and the TikZ source `docs/geometry.tex`). Use it
  unchanged; see section 3.0.

Work step by step, run the tests in section 9 locally with plain CPython before
wiring up the UI, and tell me when something in this spec is ambiguous instead of
guessing.

---

## 1. Suggested file layout

```
index.html            # single page, loads Pyodide + Plotly from CDN
css/style.css
js/app.js             # UI, file upload, Plotly rendering, downloads
js/worker.js          # (preferred) Pyodide running in a Web Worker
py/read_pro.py        # PROVIDED - unchanged
py/mask.py            # PROVIDED - unchanged
py/plotstyle.py       # PROVIDED - unchanged
py/gisaxs.py          # NEW: parsing, conversion, horizon cut, profiles, export
py/spot.py            # NEW: specular-spot mask
py/figures.py         # NEW: matplotlib PNG/PDF export using plotstyle
assets/geometry.svg   # PROVIDED - schematic of the setup (section 3.0)
assets/geometry.png   # PROVIDED - raster fallback
docs/geometry.pdf     # PROVIDED
docs/geometry.tex     # PROVIDED - TikZ source of the schematic
tests/test_gisaxs.py  # NEW: pytest, runs with normal CPython
README.md             # how to use, how to deploy, the equations
.nojekyll
```

- Load Pyodide and Plotly from a CDN (`cdn.jsdelivr.net`), **pinned to exact
  versions** (use the latest stable releases; check that numpy, pandas and
  matplotlib are available as Pyodide packages).
- Fetch the `py/*.py` files with relative URLs, write them into the Pyodide virtual
  file system and `import` them.
- Prefer running Pyodide in a **Web Worker** so the page stays responsive; show
  progress messages ("Loading Python…", "Parsing 1,091,736 pixels…", …).

---

## 2. Input files

### 2.1 `.pro` (BornAgain 1.7 XML project file)

Read it with `read_pro.read_pro(path)`, which returns a dict:

| key | meaning | unit |
|---|---|---|
| `nx`, `ny` | detector bins along x and y | – |
| `width`, `height` | `(Nbins − 1) × pitch` | mm |
| `N` | detector normal vector (sample → detector) | mm |
| `u0`, `v0` | foot of the detector normal on the detector | mm |
| `wavelength` | λ | **nm** |
| `alpha_i` | incidence angle αᵢ | **degrees** |

Derived quantities the app must compute and display (step 2 of the workflow):

- sample–detector distance **L = |N|** (only the length of N is used; the detector
  is treated as perpendicular to the beam, see section 3)
- pixel pitch = `width / (nx − 1)` (0.075 mm for the test file)
- k = 2π/λ
- angular size of one pixel = pitch / L (in mdeg)

Example values from my test file: `nx = 1028`, `ny = 1062`, `L = 1600.0 mm`,
`λ = 0.15418 nm`, `αᵢ = 0.29999°`, pitch 0.075 mm.

### 2.2 `.dat` (raw detector data)

Text file, comment lines start with `#`, then comma-separated rows:

```
# 2D Dataset
# Rows: 1062
# Columns: 1028
# X, Y, Z [, ZVariance]
# Units: X [mm], Y [mm], Z [cts]
-38.230423160199486,-1.8889617810215893,0,0
-38.15542316019949,-1.8889617810215893,0,0
...
```

- One line per pixel: `x [mm], y [mm], counts [, variance]`.
- **x and y are measured from the direct beam** on the detector.
- Rows/Columns come from the header (`# Rows:`, `# Columns:`). Read them; do not
  infer them from the data.
- Row-major order: y ascending (bottom detector row first), x ascending within a row.
  **Verify this from the data** and reshape to `(ny, nx)` accordingly.
- The variance column may be absent or all zeros. Keep it if present.
- Counts of **−1** mark dead pixels → convert to `NaN` on reading.
- Internally name the columns `x`, `y`, `intensity`, `variance` (these names are
  what `mask.py` expects).

---

## 3. Geometry and equations

### 3.0 Schematic and frames

Show the provided schematic `assets/geometry.svg` in the app in a collapsible
panel **"Geometry & equations"**, visible from step 2 onwards (open by default the
first time), and embed it at the top of the README. It contains:

- (a) side view in the **lab frame** with the **sample frame** axes,
- (b) the detector face (Y–Z plane, seen from behind the detector) with the
  **direct beam**, **sample horizon**, **specular spot**, specular rod and a pixel (x, y),
- (c) top view with φ_f,
- (d) the vectors: sample axes, **r** in both frames, **k**ᵢ, **k**_f and **q**.

Three coordinate systems are used. Keep these names consistently in code, UI and
docs:

| system | axes | meaning |
|---|---|---|
| **lab frame** | X, Y, Z | X along the incident beam, Z vertical up, Y = Z × X (horizontal) |
| **sample frame** | û, v̂, n̂ | û in the surface along the beam, v̂ = Ŷ in the surface, n̂ = surface normal |
| **detector** | x, y | position on the detector in mm, measured from the direct beam; the detector plane is the Y–Z plane: x along Y, y along Z (up) |

The sample is tilted by αᵢ about Y, so in lab components

$$
\hat{\mathbf u}=(\cos\alpha_i,\ 0,\ \sin\alpha_i),\qquad
\hat{\mathbf v}=(0,\ 1,\ 0),\qquad
\hat{\mathbf n}=(-\sin\alpha_i,\ 0,\ \cos\alpha_i)
$$

**All results (φ_f, α_f, k_f, q) are expressed in the sample frame**: the
components written x, y, z below (k_fx, q_x, …) are the components along û, v̂, n̂.

Conventions:

- **Sample frame**: the sample surface is the û–v̂ plane, n̂ is the surface normal;
  the incident beam travels along the û direction and comes **down** onto the
  surface at αᵢ.
- The **detector is perpendicular to the incident beam** at distance L.
- `x`, `y` from the `.dat` file are measured from the direct-beam spot.
- Angles in the UI and exported files are in **degrees**; trigonometry is done in
  radians.
- q and k are in **nm⁻¹** by default (λ is in nm); a UI toggle switches to Å⁻¹
  (multiply by 0.1).

### 3.1 Pixel position in the sample frame

The pixel vector **r** points from the sample to the pixel. In the lab frame

$$
\mathbf r = L\,\hat{\mathbf X} + x\,\hat{\mathbf Y} + y\,\hat{\mathbf Z}
$$

Projecting onto the sample axes (a = **r**·û, b = **r**·v̂, c = **r**·n̂) gives the same
vector in the sample frame,

$$
\mathbf r = a\,\hat{\mathbf u} + b\,\hat{\mathbf v} + c\,\hat{\mathbf n}
$$

with

$$
a = L\cos\alpha_i + y\sin\alpha_i,\qquad
b = x,\qquad
c = y\cos\alpha_i - L\sin\alpha_i
$$

$$
R = \sqrt{L^2 + x^2 + y^2} = \sqrt{a^2 + b^2 + c^2}
$$

### 3.2 Exit angles

$$
\varphi_f = \operatorname{atan2}(b,\ a),\qquad
\alpha_f = \operatorname{atan2}\!\left(c,\ \sqrt{a^2+b^2}\right)
$$

(φ_f increases with x. BornAgain uses the opposite sign for φ_f and q_y; this app
does not.)

### 3.3 Final wavevector

$$
k = \frac{2\pi}{\lambda},\qquad
\mathbf k_f = \frac{k}{R}\,(a,\ b,\ c)
$$

$$
k_{fx} = k\frac{a}{R} = k\cos\alpha_f\cos\varphi_f,\quad
k_{fy} = k\frac{b}{R} = k\cos\alpha_f\sin\varphi_f,\quad
k_{fz} = k\frac{c}{R} = k\sin\alpha_f
$$

### 3.4 Incident wavevector

$$
\mathbf k_i = k\,(\cos\alpha_i,\ 0,\ -\sin\alpha_i)
$$

### 3.5 Scattering vector  q = k_f − k_i

In the sample frame, **q** = q_x û + q_y v̂ + q_z n̂ with

$$
q_x = k\left(\frac{a}{R} - \cos\alpha_i\right),\qquad
q_y = k\,\frac{b}{R},\qquad
q_z = k\left(\frac{c}{R} + \sin\alpha_i\right)
$$

Equivalently:
$q_x = k(\cos\alpha_f\cos\varphi_f - \cos\alpha_i)$,
$q_y = k\cos\alpha_f\sin\varphi_f$,
$q_z = k(\sin\alpha_f + \sin\alpha_i)$.

Use the a/R, b/R, c/R form in code (vectorised numpy, no per-pixel trig).

### 3.6 Special positions (for checks and tests)

| feature | condition | detector position |
|---|---|---|
| direct beam | α_f = −αᵢ, φ_f = 0 | x = 0, y = 0 |
| sample horizon | α_f = 0 | y = L tan αᵢ |
| specular spot | α_f = αᵢ, φ_f = 0 | x = 0, y = L tan 2αᵢ |

---

## 4. Workflow (the UI follows these steps in order)

1. **Upload** `.dat` and `.pro` (two file inputs or one drop zone). Files never leave
   the browser.
2. **Display the `.pro` information** from section 2.1, including the derived
   quantities, as a small table.
3. **Display the first rows** of the raw data, like `df.head()` (render as an HTML
   table), plus the total number of pixels and the grid size.
4. **Optional raw intensity map (y vs x)**: checkbox "Show raw intensity map",
   with a **log-scale checkbox** and a **colormap selector** (section 6).
5. **Convert (x, y) → (φ_f, α_f)** with section 3.2. Adds 2 columns.
6. **Compute k_fx, k_fy, k_fz** with section 3.3. Adds 3 columns.
7. **Compute q_x, q_y, q_z** with section 3.5. Adds 3 columns.
8. **Drop everything below the sample horizon**, **per whole detector row**, so the
   data stays a rectangular image: compute α_f at x = 0 for each row
   (α_f,row = atan2(c, a) with b = 0) and drop every row with α_f,row < 0.
   Then set `df.attrs["nx"] = nx` and `df.attrs["ny"] = remaining rows`
   (`mask.py`'s `as_image()` needs `nx`). For my test file this keeps 925 × 1028 =
   950,900 pixels. The beamstop lies below the horizon and is removed with it.
9. **Display `df.head()`** of the new table and offer a **download**:
   - default: **`.csv.gz`** (gzip, level 6), numbers with **8 significant digits**
     (`float_format="%.8g"`); about 31 MB for the test file.
   - second button: plain `.csv` (about 108 MB).
   - Column names with units:
     `x_mm, y_mm, counts, variance, phi_f_deg, alpha_f_deg, kfx_nm-1, kfy_nm-1,
     kfz_nm-1, qx_nm-1, qy_nm-1, qz_nm-1` (use `_A-1` when Å⁻¹ is selected).
   - A `#`-comment header before the column names recording: source file names,
     L, αᵢ, λ, k, pitch, grid size, q unit, date, and that rows below the horizon
     were dropped. The file must load with `pandas.read_csv(path, comment="#")`.
10. **Ask: "Mask the specular rod?"**
    - **Yes** → use `mask.py`: `rod = mask.find_rod(df)`, then
      `df, n = mask.mask_rod(df, rod["centre"], n_sigma * rod["sigma"])` with an
      **n_sigma** input (default `mask.DEFAULT_N_SIGMA` = 4) and an optional fixed
      half-width in degrees. Show the report: rod centre, FWHM, σ, masked band,
      number of masked pixels. `mask_rod` sets `intensity` and `variance` to `NaN`
      (keep this behaviour). Skip step 11 (the spot lies on the rod).
    - **No** → go to step 11.
11. **Ask: "Mask the specular spot?"** If yes, use `spot.py` (section 5).
12. **2D intensity maps** (tickboxes, any combination):
    - q_z (y-axis) vs q_y (x-axis)
    - α_f vs φ_f
    - y vs x
    Each is a **scatter plot of every pixel, coloured by counts**; no averaging or
    regridding. Each has its own **log-scale checkbox** and **colormap selector**.
13. **1D profiles** (tickboxes): **mean counts per pixel** vs φ_f, q_y, α_f, q_z
    (section 7). Log/linear y-axis toggle. **No error bars.**
14. **Download** every plot as **PNG or PDF** (section 8).

Recompute only what a changed option affects (e.g. toggling log scale must not
re-run the conversion).

---

## 5. Specular-spot mask (`py/spot.py`)

Mask an ellipse around the specular spot, sized from the data:

1. Predicted position: φ_f = 0, α_f = αᵢ. Take a local window of ±W around it
   (default W = 15 pixel angular widths in both directions).
2. In the window build a horizontal profile P(φ_f) (mean over rows) and a vertical
   profile Q(α_f) (mean over columns).
3. For each profile: background B = median of the outer 20 % of the window;
   height H = max − B; **FWHM from the two half-maximum crossings of (P − B) = H/2,
   found by linear interpolation** between neighbouring points; σ = FWHM / (2√(2 ln 2));
   centre = centroid of (P − B) over the points above H/2.
4. Mask (set `intensity` and `variance` to `NaN`) every pixel with

$$
\left(\frac{\varphi_f-\varphi_0}{n\,\sigma_\varphi}\right)^2+
\left(\frac{\alpha_f-\alpha_0}{n\,\sigma_\alpha}\right)^2\le 1
$$

   with n from a UI input (default 4). If a FWHM comes out narrower than one pixel,
   use one pixel.
5. Report φ₀, α₀, σ_φ, σ_α, number of masked pixels.

---

## 6. Log scaling and colormaps

- **Never transform the data.** Profiles are computed on linear counts; log is
  applied only to the display.
- Zeros (≈96 % of pixels) are real measurements: on a log scale draw them in a
  dedicated "under" colour. `NaN` (dead or masked) is drawn in a different "bad"
  colour or omitted from the scatter. Follow `plotstyle.intensity_cmap()`
  (nudge 0 → 0.5 with vmin = 1).
- Default colour range: vmin = 1 count, vmax = max of the unmasked data; provide
  optional manual vmin/vmax inputs.
- **Plotly (screen):** plot colour = log₁₀(counts) when log is ticked; label the
  colourbar at 1, 10, 10², … with real counts; the hover label must show the real
  count, not its logarithm (pass linear counts via `customdata`).
- **matplotlib (export):** `LogNorm(vmin, vmax)` with `plotstyle.intensity_cmap(name)`.
- **Colormaps offered** (projector-friendly, colour-blind-safe first):
  viridis (default), cividis, inferno, magma, turbo. No red–green maps.

---

## 7. 1D profiles: mean counts per pixel

Use the detector grid instead of binning. `NaN` pixels are ignored; a fully masked
row/column becomes `NaN` and is drawn as a **gap**.

Vertical direction (vs α_f and vs q_z): this function is based on `mask.py`'s
`row_profile`, generalised to any label column:

```python
def row_profile(df, label="alpha_f"):
    intensity = as_image(df, "intensity")
    coord = as_image(df, label)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(intensity, axis=1)
    return pd.Series(means, index=coord.mean(axis=1))
```

Horizontal direction (vs φ_f and vs q_y): the matching **mean**-based function.
Name it `column_mean_profile` so it does **not** shadow `mask.py`'s sum-based
`column_profile`, which `find_rod()` relies on:

```python
def column_mean_profile(df, label="phi_f"):
    intensity = as_image(df, "intensity")
    coord = as_image(df, label)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(intensity, axis=0)
    return pd.Series(means, index=coord.mean(axis=0))
```

Put both in `gisaxs.py` (import `as_image` from `mask`).

**Origin-style hover** on the 1D plots: crosshair spike lines on both axes
(`showspikes`, `spikemode: "across"`), `hovermode: "x"` or `"closest"`, and a
hover label showing both coordinates with sensible precision, e.g.
`q_y = 0.1234 nm⁻¹, I = 15.3 cts`.

---

## 8. Plot styling and export

- **On screen:** Plotly. 2D maps use **`scattergl`** (WebGL) with ~1 million points;
  send data to JS as typed arrays (`Float32Array`), not as Python lists. Marker size
  about one pixel; square aspect where it makes sense.
- **Export (PNG/PDF):** matplotlib inside Pyodide, styled with
  `plotstyle.use_publication_style(engine="stix")`. LaTeX is not available in the
  browser; Times New Roman is not available, so `plotstyle` falls back to
  STIXGeneral (that is intended).
- Two style presets selectable in the UI:
  - **Publication**: `plotstyle` defaults, `figure_size(columns=1 or 2)`.
  - **Presentation / projector**: same function with `font_size=16`,
    `linewidth=1.5`, thicker lines (`lines.linewidth` 2), white background, larger
    figure (e.g. 8 × 6 in).
- In 2D exports the scatter layer is `rasterized=True` (axes, labels and colourbar
  stay vector). Otherwise the PDF would contain a million vector markers.
- Follow `plotstyle`'s gotcha: keep numbers outside `$...$` in labels
  (e.g. `r"$\alpha_i$ = 0.300$^\circ$"`).
- Axis labels with units, e.g. `$q_y$ (nm$^{-1}$)`, `$\alpha_f$ (deg)`,
  `Mean counts per pixel`.
- Save to `io.BytesIO` with `fig.savefig(buf, format="png"|"pdf")`, pass the bytes
  to JS, download via `Blob` + `URL.createObjectURL`. File names should include the
  input file stem and the plot type, e.g. `various-angles_000_qz-vs-qy.pdf`.

---

## 9. Tests (`tests/test_gisaxs.py`, pytest, plain CPython)

Test the conversion with a toy geometry: **L = 86 mm, αᵢ = 1°, λ = 0.15406 nm**
(q in nm⁻¹). Expected values (tolerance 1e-5):

| x (mm) | y (mm) | φ_f (°) | α_f (°) | k_fx | k_fy | k_fz | q_x | q_y | q_z |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | −1.000000 | 40.777803 | 0 | −0.711779 | 0 | 0 | 0 |
| 0 | 3 | 0 | 0.997880 | 40.777829 | 0 | 0.710270 | 0.000026 | 0 | 1.422049 |
| 2 | 2 | 1.331882 | 0.332130 | 40.772311 | 0.947953 | 0.236414 | −0.005492 | 0.947953 | 0.948193 |
| 3 | 6 | 1.995755 | 2.989102 | 40.703821 | 1.418390 | 2.126724 | −0.073982 | 1.418390 | 2.838503 |

Also test:

- direct beam (0, 0) gives α_f = −αᵢ exactly and q = 0;
- specular position (0, L tan 2αᵢ) gives α_f = αᵢ, q_x = q_y = 0, q_z = 2k sin αᵢ;
- horizon position (0, L tan αᵢ) gives α_f = 0;
- |k_f| = k for every pixel;
- the horizon cut keeps whole rows only (row count × nx = pixel count);
- with the real test file (`various-angles_000.pro`, L = 1600 mm, αᵢ = 0.299991444658313°,
  λ = 0.15418 nm): pixel (x, y) = (0, 20.3) gives α_f = 0.426911°, q_z = 0.517014 nm⁻¹.

If you have the real `.dat`/`.pro` files in the folder, add an integration test that
runs the whole pipeline and checks that 925 × 1028 pixels remain after the horizon cut.

---

## 10. UX details

- Clear step-by-step layout (numbered panels that unlock in order).
- Friendly error messages for: wrong file type, header Rows × Columns not matching
  the number of data lines, grid size different from the `.pro` `nx`/`ny`,
  missing `.pro` parameters.
- A "q unit" toggle (nm⁻¹ / Å⁻¹) that affects tables, plots and exports.
- Everything must work offline after the first load except the CDN downloads.
- Test in Chrome and Firefox with the 46 MB `.dat` file.

---

## 11. README.md

Include: the schematic `assets/geometry.svg` at the top, what the app does, how to
use it, the frames table and the equations of section 3, the
assumptions (detector perpendicular to beam, x/y from direct beam, φ_f sign
convention), and **deployment**:

1. Push the folder to a GitHub repository.
2. Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. Open `https://<user>.github.io/<repo>/`.

Also how to run the tests locally (`pip install numpy pandas matplotlib pytest`,
`pytest tests/`) and how to preview locally (`python -m http.server` in the repo
folder, then open `http://localhost:8000`; opening `index.html` directly from disk
will not work because the Python files are fetched).