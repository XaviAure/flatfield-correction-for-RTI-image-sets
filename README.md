# Flat-Field Correction for RTI (Sony ARW)

Polynomial flat-fielding for Reflectance Transformation Imaging (RTI) and related photometric workflows. The scripts estimate a smooth illumination model from white‑board captures (one per LED position) and apply the matched correction to each object image.

- Works from **RAW (.ARW)** inputs (Sony); outputs TIFF/PNG/JPEG
- **Grayscale** or **per‑channel** (RGB) polynomial fits (degree 3–9)
- 8‑bit **JPEG** for RTI Relight; **TIFF/PNG** for analysis/archival
- Optional median generation for diffuse/albedo textures

---

## Installation

```bash
pip install numpy opencv-python rawpy tifffile scikit-learn
```

Python ≥ 3.8 recommended.

---

## Repository layout (expected)

```
project_root/
├─ RTI_flats/                 # White-board captures (image_001.ARW, …)
├─ RTI_images/                # One subfolder per RTI set (RTI_01/, RTI_02/, …)
├─ flat_coeffs/               # Created on coefficient generation
├─ calculate_polynomial_coefficients.py
├─ flatfield_main.py
└── README.md
```

**Naming:** `image_001.ARW` ↔ `image_001_coeffs.pkl` (grayscale) or `image_001_coeffs_rgb.pkl` (per‑channel).

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
python flatfield_main.py   --coeffs-path flat_coeffs   --parent-folder RTI_images   --per-channel --jpg --bit-depth 8

# High-bit outputs for analysis
python flatfield_main.py   --coeffs-path flat_coeffs   --parent-folder RTI_images   --per-channel --tiff --bit-depth 16
```

### Optional: diffuse/albedo median
```bash
python flatfield_main.py   --coeffs-path flat_coeffs   --parent-folder RTI_images   --median --median-range "01-08"
```
Pick 4–8 near‑perpendicular lights to suppress highlights and shadows.

---

## Output behaviour

- `--jpg --bit-depth 8` → **8‑bit JPEG** (quality 100).
- `--jpg --bit-depth 16` → **JPEG is 8‑bit only** (OpenCV will down‑cast). For true 16‑bit, use `--tiff --bit-depth 16` (or PNG via `--png` if available in your branch).
- `--tiff` supports 8‑bit or 16‑bit.
- `--median` saves a TIFF at the **selected bit‑depth** for the run.

---

## Colour‑correction policy (tool‑agnostic)

Flat‑fielding is a **multiplicative** operation and should be performed on data with a **linear** tone response.

- **RAW development in this tool.** The code develops RAW with camera white balance and **no auto‑brightening**. If you require strict linearity, set `gamma=(1,1)` in the RAW post‑processing call within the scripts.
- **Linear adjustments** (safe before *or* after flat‑field): per‑channel gains, a global exposure scalar, and a 3×3 colour matrix (these commute with division).
- **Non‑linear adjustments** (do **after** flat‑field): tone curves, gamma encoding (γ ≠ 1), LUTs, “filmic/DRC”, local contrast/sharpening, highlight reconstruction with tone mapping, and creative looks.
- **Uniform settings for stacks.** For RTI/photometric‑stereo stacks, avoid any frame‑wise auto adjustments; keep settings **identical across all images**.

**Important:** This project **only ingests RAW (.ARW)**. If you prefer external calibration in third‑party software, run **flat‑fielding here first on the RAWs**, then apply calibration/colour work to the **flat‑fielded outputs** in that editor.

---

## Tips & troubleshooting

- **Mode mismatch**: If coefficients were made with `--per-channel`, you must also run application with `--per-channel` (and vice versa).
- **No coefficients found**: check that flat and image numbering match (`image_001`, `image_002`, …).
- **Shape warnings**: coefficients were derived at a different resolution/crop. Re‑capture/recompute for best results.
- **Performance**: set `--num-cores` sensibly; RAW decode and TIFF writing are I/O‑heavy.

---

## Rationale (brief)

- Polynomial surfaces (degree 3–9) approximate combined lens vignetting and LED angular falloff without overfitting.
- Working (or developing) in a linear tone space ensures division by the flat preserves intensity ratios needed for RTI/PS analysis.

---

## Licence

**GNU General Public License v3.0 (GPL‑3.0).**  
See `LICENSE` for full terms.

---

## Citation / acknowledgements

If you use this in publications, please cite the repository and acknowledge the use of polynomial flat‑fielding for cultural‑heritage RTI/photometric workflows.

---

### Common commands (copy–paste)

```bash
# Coefficients (default degree 5, per-channel)
python calculate_polynomial_coefficients.py --flat-dir RTI_flats --output-dir flat_coeffs --degree 5 --per-channel

# JPEGs for RTI Relight
python flatfield_main.py --coeffs-path flat_coeffs --parent-folder RTI_images --per-channel --jpg --bit-depth 8

# 16-bit TIFFs for analysis
python flatfield_main.py --coeffs-path flat_coeffs --parent-folder RTI_images --per-channel --tiff --bit-depth 16

# Diffuse/albedo median (images 1–8)
python flatfield_main.py --coeffs-path flat_coeffs --parent-folder RTI_images --median --median-range "01-08"
```
