# Delhi NCR Urban Change Intelligence (2022–2026)

Satellite-based land-cover and urban change detection across the Delhi National Capital Region, built on Sentinel-2 imagery, a deep learning change detection model, and Monte Carlo Dropout uncertainty quantification.

**Status:** Phase 4 complete (baseline model trained and evaluated) — Phase 5 (Siamese U-Net) in progress. See [`PROJECT_STATE.md`](PROJECT_STATE.md) for full build log.

---

## Overview

This project detects and quantifies physical land-cover change across Delhi NCR by comparing Sentinel-2 L2A surface reflectance composites from two time periods, January–February 2022 and January–February 2026. It combines remote sensing, deep learning, and spatial analysis into a single reproducible pipeline, rather than a one-off notebook.

The system identifies built-up expansion, vegetation loss and gain, and other land-cover transitions at a 10m pixel resolution, and reports model uncertainty alongside every prediction, not just a point estimate.

This is a technical remote sensing and machine learning project. No political, administrative, or causal framing is applied to any observed change — the scope is strictly satellite-observed land-cover change over a fixed calendar window.

## Study Area

Delhi, Gurugram, Noida, Ghaziabad, Faridabad — bounding box `[76.75, 28.30, 77.60, 28.95]` (approximate, not an administrative boundary).

## Data

| | |
|---|---|
| Source | Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`), via Google Earth Engine |
| T1 | Jan 1 – Feb 15, 2022 (11 cloud-free scenes) |
| T2 | Jan 1 – Feb 15, 2026 (12 cloud-free scenes) |
| Bands | B2, B3, B4, B8, B11, B12 |
| Cloud masking | Scene-level filter (< 20% cloud cover) + SCL-based pixel-level masking |
| Resolution | 10m, CRS EPSG:4326, 9463 × 7244 px |
| Alignment | Pixel-for-pixel verified (identical CRS, transform, dimensions) |

## Labels

Labels are **spectral-index-derived and weakly-supervised**, not manually annotated ground truth. This is standard practice in remote sensing change detection when pixel-perfect ground truth isn't available, and is documented honestly rather than presented as verified ground truth across the full dataset.

- Change indices: ΔNDVI and ΔNDBI between T1 and T2.
- Threshold: adaptive, `mean ± k·std` of the delta distribution over valid (non-cloud-masked) pixels, recomputed and logged on every run.
- A pixel is labeled **change** if it crosses either threshold.
- Final parameters: `ndvi_std_multiplier = 2.2`, `ndbi_std_multiplier = 1.5` (raised from an initial 1.5/1.5 after manual chip review showed seasonal crop-cycle noise in agricultural parcels).
- **Manual verification:** a stratified sample of 45 patches (low / medium / high change fraction) was exported as before/after/overlay image chips and visually reviewed. Low and medium strata showed labels consistent with real visual change; a minority of high-stratum patches retain residual agricultural NDVI noise — a known, disclosed limitation.
- **Split:** geographically separated by column stripe with a buffer zone, not randomly per-patch, to prevent spatial leakage between neighboring patches. Train 700 / val 140 / test 56 patches.

## Model

**Phase 4 (complete): CNN baseline.** A shallow encoder-decoder network over the concatenated 12-band (T1 + T2) input, trained to establish an honest performance floor before adding architectural complexity.

| Split | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| Validation | 0.815 | 0.809 | 0.805 | 0.676 |
| Test (held out) | 0.819 | 0.852 | 0.833 | 0.714 |

**Phase 5 (in progress): Siamese U-Net.** Shared-weight encoder branches for T1 and T2, feature fusion, U-Net decoder with skip connections — evaluated against the baseline above on the identical split to confirm the added complexity is justified.


**Planned:** Monte Carlo Dropout at inference for per-pixel uncertainty maps, spatial hotspot and proximity analysis, and a Streamlit dashboard for interactive exploration.

## Methodology Pipeline

Sentinel-2 (Google Earth Engine)
-> Cloud masking + preprocessing + alignment
-> Patch generation (256x256)
-> Spectral-index change labels + manual verification sample
-> Geographic train/val/test split
-> Baseline CNN -> Siamese U-Net
-> Monte Carlo Dropout uncertainty
-> Spatial hotspot + proximity analysis
-> Streamlit dashboard

## Project Structure

delhi-ncr-satellite-change/
  src/
    data/        GEE acquisition, preprocessing, label generation, patch caching
    models/      Baseline CNN, Siamese U-Net
    training/    Losses, metrics, training loop
    inference/   Prediction, MC Dropout uncertainty
    spatial/     Change area, hotspots, proximity analysis
  notebooks/     Colab notebooks (GEE acquisition, preprocessing, labeling, training)
  app/           Streamlit dashboard
  reports/       Verification chips, final report
  PROJECT_STATE.md   Full build log and current status
  requirements.txt

  
## Reproduction

Compute is split across two environments: a local machine for repository and file management, and Google Colab (free-tier GPU) for Earth Engine queries and model training.

```bash
git clone https://github.com/karan02566-prog/delhi-ncr-satellite-change.git
cd delhi-ncr-satellite-change
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then open `notebooks/00_colab_setup.ipynb` in Google Colab to mount Drive and authenticate Earth Engine, and run the numbered notebooks in order.

## Stack

Python, PyTorch, Google Earth Engine, rasterio, geopandas, scikit-learn, Streamlit.

## Limitations

- Labels are spectral-index-derived and weakly-supervised, manually verified on a sample rather than the full dataset.
- A minority of high-change-fraction patches retain agricultural NDVI seasonal noise that a connected-component speckle filter was designed to address but has not yet been applied.
- The held-out test split (56 patches) is smaller than the intended 15% target due to non-uniform patch density near split boundaries; results are reported as-is rather than resampled to a round number.

## License

Not yet specified.
