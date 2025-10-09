# Flat-Field Correction for RTI (Sony ARW)

Polynomial flat-fielding for Reflectance Transformation Imaging (RTI) and related photometric workflows. The scripts estimate a smooth illumination model from white-board captures (one per LED position) and apply the matched correction to each object image.

- Works in **linear** space (γ = 1) to preserve multiplicative ratios  
- **Grayscale** or **per-channel** (RGB) fits  
- Memory-safe polynomial evaluation (no giant design matrices)  
- 8-bit **JPEG** for RTI Relight, **16-bit TIFF/PNG**, and flexible median exports (TIFF/PNG/JPG)

---

## Installation

```bash
pip install numpy opencv-python rawpy tifffile scikit-learn
```

Python ≥3.8 recommended.

---

## Repository layout (expected)

```
project_root/
├─ RTI_flats/                 # White-board captures (image_001.ARW, …)
├─ RTI_images/                # One subfolder per RTI set (RTI_01/, RTI_02/, …)
├─ flat_coeffs/               # Created on coefficient generation
├─ calculate_polynomial_coefficients.py
├─ flatfield_main.py
└─ README.md
```

**Naming:** Files matched by number—`FLAT_01.ARW` ↔ `image_01_coeffs.pkl` works. Prefixes can differ; digits must match.

---

## Quick start

### 1) Generate coefficients from flats
```bash
python calculate_polynomial_coefficients.py   --flat-dir RTI_flats   --output-dir flat_coeffs   --degree 5   --per-channel
```
- Use `--per-channel` if RGB channels show different vignetting/LED patterns.  
- Degree: 3 (very smooth) · **5 (default)** · 7/9 (sharper LED hotspots).

### 2) Apply to RTI sets
```bash
# 8-bit JPEGs for RTI Relight
python flatfield_main.py   --coeffs-path flat_coeffs   --parent-folder RTI_images   --per-channel --jpg

# High-bit outputs for analysis
python flatfield_main.py   --coeffs-path flat_coeffs   --parent-folder RTI_images   --per-channel --tiff --bit-depth 16
```

**Notes on outputs**
- `--all` generates TIFF + JPG + median (default if no flags specified).
- `--jpg` → 8-bit JPEG only (quality 100). For 16-bit output, use `--tiff` or `--median-format png`.
- `--tiff` supports 8-bit or 16-bit (controlled by `--bit-depth`).
- `--median` honours `--median-format {tiff,png,jpg}` and `--median-bit-depth` (defaults to `--bit-depth`). JPEG medians are always 8-bit with a warning if 16-bit is requested.
- `--num-cores N` sets parallel processing threads (default: 10).

Use the median flags to tailor outputs for downstream relighting, segmentation, or archival workflows.

### Optional: diffuse/albedo median
```bash
python flatfield_main.py   --coeffs-path flat_coeffs   --parent-folder RTI_images   --median \
    --median-format png --median-bit-depth 16 --median-range "01-08"
```
Pick 4–8 near-perpendicular lights to suppress highlights and shadows, adjusting format/bit depth to suit downstream tools.

---

## Colour-correction

Flat-fielding is a **multiplicative** correction and must be done on **linear** data.

- The scripts load RAW with `gamma=(1,1)`, `no_auto_bright=True`, apply camera WB (multiplicative), and then divide by the fitted flat.
- **Linear adjustments** (safe before *or* after flat-field): per-channel gains, exposure scalars, 3×3 colour matrices.
- **Non-linear adjustments** (do **after** flat-field): tone curves, gamma ≠ 1, LUTs, filmic/DRC, local contrast.

**Using RawTherapee before flat-field:** allowed **only** if you export **linear** 16-bit TIFF and apply *strictly linear* calibration (WB, exposure scalar, chart-derived matrix/gains), with identical settings for every frame. Otherwise, flat-field first, then colour-grade.

For RTI/photometric-stereo stacks, avoid any frame-wise auto adjustments; keep settings uniform across the stack.

---

## Tips & troubleshooting

- **Mode mismatch**: If coefficients were made with `--per-channel`, you must also run application with `--per-channel` (and vice versa).
- **No coefficients found**: check that flat and image numbering match (`image_001`, `image_002`, …).
- **Numbers drive matching**: The applicator extracts the numeric token (e.g. `01`) and pairs it with any file named like `*_01_coeffs*.pkl`. Prefixes may differ (`RTI_01`, `FLAT_01`, etc.), but the digits must match.
- **Shape warnings**: coefficients were derived at a different resolution/crop. Re-capture/recompute for best results.
- **Clipping warnings**: Corrected values exceed 16-bit range. Review flat captures if >1% of pixels clip.
- **Performance**: set `--num-cores` sensibly; RAW decode and TIFF write are I/O-heavy.

---

## Rationale

- Polynomial surfaces (degree 3–9) approximate combined lens vignetting and LED angular falloff without overfitting.
- Evaluation uses precomputed exponent maps to avoid building an `(H·W)×M` design matrix (large images remain memory-safe).
- Working in linear space ensures that division by the flat preserves true intensity ratios.

---

## Licence

**GNU General Public License v3.0 (GPL-3.0).**  
See `LICENSE` for full terms.

---

## Citation / acknowledgements

If you use this in publications, please cite the repository and acknowledge the use of polynomial flat-fielding for cultural-heritage RTI/photometric workflows.

---

### Common commands (copy–paste)

```bash
# Coefficients (default degree 5, per-channel)
python calculate_polynomial_coefficients.py --flat-dir RTI_flats --output-dir flat_coeffs --degree 5 --per-channel

# JPEGs for RTI Relight
python flatfield_main.py --coeffs-path flat_coeffs --parent-folder RTI_images --per-channel --jpg

# 16-bit TIFFs for analysis
python flatfield_main.py --coeffs-path flat_coeffs --parent-folder RTI_images --per-channel --tiff --bit-depth 16

# Diffuse/albedo median (images 1–8)
python flatfield_main.py --coeffs-path flat_coeffs --parent-folder RTI_images \
    --median --median-format png --median-bit-depth 16 --median-range "01-08"
```
