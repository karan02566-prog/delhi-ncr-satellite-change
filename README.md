# Delhi NCR Urban Change Intelligence (2022–2026)

Satellite-based land-cover and urban change detection across the Delhi National Capital Region (NCR), combining multi-temporal Sentinel-2 surface reflectance composites, a deep learning change detection CNN, Monte Carlo (MC) Dropout uncertainty quantification, and spatial intelligence (physical area quantification, data-driven hotspot clustering, and urban edge proximity analysis).

> **Current Status: In Progress (Phases 0–9 Complete | Phases 10–12 In Progress)**  
> All model training, evaluation, uncertainty estimation, and full-scene spatial analytics are completed and empirically verified against real Sentinel-2 data. Current work is focused on Phase 10 (Interactive Streamlit Dashboard). See [`PROJECT_STATE.md`](PROJECT_STATE.md) for the detailed chronological build log and locked parameters.

---

## Overview

This project quantifies physical land-cover transitions across the Delhi NCR region by comparing Sentinel-2 L2A surface reflectance composites between two winter calendar windows: **January 1 – February 15, 2022 (T1)** and **January 1 – February 15, 2026 (T2)**. 

The pipeline:
1. Acquires and aligns cloud-masked 6-band Sentinel-2 composites at 10m spatial resolution via Google Earth Engine.
2. Generates statistically adaptive, weakly-supervised change labels verified via stratified visual review.
3. Trains and evaluates deep learning architectures on a geographically isolated column-stripe split with spatial buffer zones.
4. Performs MC Dropout inference ($N=20$ stochastic forward passes) to estimate per-pixel model predictive uncertainty.
5. Characterizes change into physical categories (built-up expansion, vegetation loss, vegetation gain, other/uncertain) and computes real-world physical area changes ($\text{km}^2$, $\text{ha}$).
6. Discovers spatial growth agglomerations via density-based clustering (DBSCAN) and measures Euclidean proximity to the baseline urban core.

*Scope Note:* This is a technical remote sensing and geospatial machine learning project. Analysis is strictly based on continuous geographic coordinates and physical raster geometry. No political, administrative, or causal framing is applied.

---

## Study Area & Temporal Design

- **Region of Interest (ROI):** Rectangular bounding box `[76.75, 28.30, 77.60, 28.95]` covering Delhi, Gurugram, Noida, Ghaziabad, Faridabad, and surrounding peri-urban/agricultural matrix (approximate bounding box, not an administrative boundary).
- **T1 Composite:** Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`), Jan 1 – Feb 15, 2022 (11 cloud-free scenes, cloud cover $< 20\%$ + SCL pixel masking).
- **T2 Composite:** Sentinel-2 L2A, Jan 1 – Feb 15, 2026 (12 cloud-free scenes, cloud cover $< 20\%$ + SCL pixel masking).
- **Raster Dimensions:** $9463 \times 7244$ pixels at 10m resolution (EPSG:4326), 6 bands (`B2`, `B3`, `B4`, `B8`, `B11`, `B12`). Pixel-for-pixel alignment confirmed (`aligned=True`).

---

## Weakly-Supervised Labels & Geographic Split

- **Labeling Method:** Adaptive spectral index thresholding ($\Delta\text{NDVI}$ and $\Delta\text{NDBI}$) computed over valid non-cloud pixels ($\mu \pm k \cdot \sigma$). Final parameters: $\text{NDVI multiplier} = 2.2$, $\text{NDBI multiplier} = 1.5$.
- **Manual Verification:** Stratified sample of 45 patches (low/medium/high change) reviewed. Red overlays aligned with visible construction and settlement edge expansion, with minor residual agricultural crop-cycle noise in high-change patches.
- **Geographic Train/Val/Test Split:** Partitioned along the raster width into non-overlapping column stripes with a 256-pixel buffer zone between splits to prevent spatial autocorrelation leakage:
  - **Train:** 700 patches (69.4%, mean change fraction = 0.1276)
  - **Validation:** 140 patches (13.9%, mean change fraction = 0.1450)
  - **Test (Held-Out):** 56 patches (5.6%, mean change fraction = 0.1422)
  - **Buffer Zone:** 112 patches (excluded from training/eval)

---

## Model Architecture & Evaluation

Two architectures were trained on the identical geographic split, patch cache, and loss function (Masked BCE + Dice):

1. **BaselineChangeCNN (Locked Model):** Shallow encoder-decoder CNN over concatenated 12-channel input (T1 + T2) with skip connections. Updated in Phase 7 to include `Dropout2d(p=0.3)` after each block to support MC Dropout inference.
2. **Siamese U-Net:** Dual-branch shared-weight encoder (6 bands per branch), absolute-difference feature fusion, and U-Net decoder.

### Performance Comparison & Architecture Decision

| Model Architecture | Val F1 | Val IoU | Test F1 | Test IoU | Test Precision | Test Recall | Status |
|---|---|---|---|---|---|---|---|
| **Baseline CNN (No Dropout, Phase 4/6)** | 0.8050 | 0.6761 | **0.8329** | **0.7144** | 0.8185 | 0.8519 | Preserved for reference |
| **Siamese U-Net (Phase 5)** | 0.7877 | 0.6534 | **0.8243** | **0.7020** | 0.8092 | 0.8476 | Dropped (underperformed) |
| **Baseline CNN (WITH Dropout, Phase 7)** | 0.7907 | 0.6572 | **0.7768** | **0.6359** | 0.7734 | 0.7849 | **LOCKED Active Model** |

*Decision Rationale:* Siamese U-Net underperformed the baseline across all validation and test metrics, confirmed at the pixel level by confusion matrix analysis (higher false positive rate and higher false negative rate). Adding `Dropout2d(p=0.3)` to the baseline cost ~5–6 F1 points on the small 56-patch test set (an expected regularization trade-off disclosed honestly), but enabled spatial uncertainty estimation. The dropout-enabled Baseline CNN was locked as the active model for all subsequent phases.

---

## Spatial Intelligence & Empirical Results (Phase 9)

Inference was scaled across all $n=1008$ usable scene patches ($6,585.87\text{ km}^2$ valid area) using the locked Baseline CNN with MC Dropout ($N=20$ stochastic forward passes, `model.train()`) and rule-based change characterization:

### 1. Full-Scene Land-Cover Change Breakdown
- **Total Valid Evaluated Area:** $6,585.87\text{ km}^2$ ($658,587\text{ ha}$)
- **Total Changed Area (Net):** $917.75\text{ km}^2$ (**$13.94\%$** of valid area)

| Land-Cover Change Category | Physical Area ($\text{km}^2$) | Area ($\text{ha}$) | $\%$ of Valid Scene | $\%$ of Total Change |
|---|---|---|---|---|
| **Built-Up Expansion** | **$477.22\text{ km}^2$** | $47,722\text{ ha}$ | **$7.25\%$** | **$52.00\%$** |
| **Vegetation Gain** *(Agricultural Phenology)* | **$300.66\text{ km}^2$** | $30,066\text{ ha}$ | **$4.57\%$** | **$32.76\%$** |
| **Other / Uncertain Change** | **$75.25\text{ km}^2$** | $7,525\text{ ha}$ | **$1.14\%$** | **$8.20\%$** |
| **Vegetation Loss** | **$64.62\text{ km}^2$** | $6,462\text{ ha}$ | **$0.98\%$** | **$7.04\%$** |

### 2. Predictive Uncertainty (MC Dropout)
- **Mean Uncertainty (Changed Pixels):** $0.0836$
- **Mean Uncertainty (No-Change Pixels):** $0.0238$
- *Qualitative Review:* High uncertainty concentrates sharply along structural boundaries and mixed-pixel transition edges (e.g. road margins, building fringes), while solid interiors exhibit low model uncertainty.

### 3. Spatial Growth Hotspots (DBSCAN)
- **Parameters:** $z_{\text{threshold}} = 1.645$ (upper 95% one-tailed significance on built-up expansion density), $\text{eps} = 0.0369^\circ$ (derived from $2.56\text{ km}$ patch stride to connect 8-neighbor tiles), $\text{min\_samples} = 2$.
- **Findings:** 58 candidate hotspot patches forming **9 multi-patch growth clusters** and 8 isolated point hotspots.

### 4. Proximity to Baseline Urban Core
- Evaluated on $3,884,061$ model-predicted built-up expansion pixels relative to adaptive T1 urban core ($\text{NDBI}_{\text{cutoff}} = 0.0561$):
  - **Mean Distance:** $86.1\text{ m}$ | **Median Distance:** $56.6\text{ m}$
  - **Within 500m (Direct Edge Expansion):** $99.3\%$
  - **Within 1000m (Near Urban Fringe):** $100.0\%$
  - **Beyond 2000m (Leapfrog Development):** $0.0\%$
  - *Morphological Context:* Reflects genuine Delhi NCR peri-urban morphology, where dense rural *abadi* and village settlement nodes are spaced ~1–2 km apart across agricultural land.

---

## Known Limitations

1. **Weakly-Supervised Labels:** Labels are derived from spectral indices ($\Delta\text{NDVI} / \Delta\text{NDBI}$) with sample verification, not pixel-perfect ground truth.
2. **Vegetation Gain Framing:** `vegetation_gain` ($300.66\text{ km}^2$, $32.76\%$ of change) represents seasonal agricultural crop-cycle phenology differences (fallow vs. planted fields between Jan/Feb 2022 and Jan/Feb 2026 windows across the NCR agricultural matrix), and is **not** an ecological afforestation trend.
3. **Small Held-Out Test Split:** The test split contains 56 patches ($5.6\%$ of the scene) due to fixed geographic column stripe constraints; test metrics should be interpreted with appropriate caution.
4. **Adaptive Urban Core Bleed:** The data-driven T1 NDBI cutoff ($0.0561$) exhibits minor bleeding into fallow/green agricultural parcels in 1 of 6 verification chips (chip 03, medium stratum).
5. **Absence of Vector Datasets:** Road proximity analysis was skipped due to lack of external road shapefiles/OSM vectors (preventing data fabrication). Administrative/sub-region breakdowns were skipped due to lack of verified boundary vectors and strict adherence to a non-political geospatial framing.
6. **MC Dropout Stochasticity:** Hotspot cluster counts and per-cluster areas carry minor run-to-run stochastic variance due to MC Dropout sampling.

---

## Tech Stack

- **Remote Sensing & Data:** Google Earth Engine Python API (`earthengine-api`), `geemap`, `rasterio`
- **Deep Learning:** PyTorch (`torch`, `torchvision`), MC Dropout inference
- **Geospatial & Spatial Analytics:** `geopandas`, `shapely`, `scipy` (`ndimage`, `spatial`, `stats`), `scikit-learn`
- **Data & Visualization:** `numpy`, `pandas`, `matplotlib`, `plotly`, `folium`, `streamlit`
- **Testing:** `pytest`

---

## Project Structure

```
delhi-ncr-satellite-change/
├── src/
│   ├── data/            # GEE download, preprocessing, label generation, patch cache, datasets
│   ├── models/          # BaselineChangeCNN (with dropout), SiameseUNet
│   ├── training/        # Losses, metrics, training loops, confusion matrix evaluation
│   ├── inference/       # MC Dropout uncertainty quantification
│   ├── spatial/         # Characterization, area conversions, adaptive cutoff, DBSCAN hotspots, proximity
│   └── visualization/   # Map plotting and dashboard components
├── notebooks/           # Numbered Google Colab execution notebooks (00 through 04)
├── tests/               # Unit test suite (test_spatial.py, etc.)
├── reports/             # Verification chips, error analysis, spatial summary artifacts
├── PROJECT_STATE.md     # Chronological build log, metrics, and locked architectural decisions
└── requirements.txt     # Python dependencies
```

---

## How to Reproduce

Compute is split between a local machine (file management, git, unit tests) and Google Colab (free GPU for Earth Engine queries and model training):

### Local Setup
```bash
git clone https://github.com/karan02566-prog/delhi-ncr-satellite-change.git
cd delhi-ncr-satellite-change
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
pytest tests/              # Run test suite
```

### Google Colab Execution
Run the numbered notebooks in `/notebooks` sequentially:
1. `notebooks/00_colab_setup.ipynb`: Mount Google Drive and authenticate Earth Engine.
2. `notebooks/01_data_acquisition.ipynb`: Query and export T1 (2022) and T2 (2026) cloud-masked composites.
3. `notebooks/02_preprocessing.ipynb`: Validate raster alignment and band normalization.
4. `notebooks/03_label_generation.ipynb`: Generate weakly-supervised change labels and geographic split manifest.
5. `notebooks/04_spatial_intelligence.ipynb`: Run full-scene MC Dropout inference, area aggregation, DBSCAN hotspot clustering, and urban edge proximity.

---

## Roadmap

- [x] **Phase 0–3:** Data acquisition, preprocessing, label generation & geographic split
- [x] **Phase 4–6:** Baseline CNN & Siamese U-Net training, evaluation & model locking
- [x] **Phase 7:** MC Dropout implementation & uncertainty validation
- [x] **Phase 8:** Rule-based change characterization
- [x] **Phase 9:** Full-scene spatial intelligence, hotspot clustering & urban proximity
- [ ] **Phase 10 (Current):** Interactive Streamlit dashboard with Folium overlays & Plotly analytics
- [ ] **Phase 11:** End-to-end pipeline validation & deployment packaging
- [ ] **Phase 12:** Comprehensive technical report & documentation
