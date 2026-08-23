# PROJECT STATE

## Current Phase
PHASE 10 — Interactive Dashboard (COMPLETE, user-confirmed working)

## Completed Phases
- Phase 0: Project Initialization
- Phase 1: Data Acquisition
- Phase 2: Preprocessing
- Phase 3: Ground Truth / Labels
- Phase 4: Baseline Model
- Phase 5: Siamese U-Net
- Phase 6: Model Evaluation
- Phase 7: MC Dropout Uncertainty
- Phase 8: Change Characterization
- Phase 9: Spatial Intelligence
- Phase 10: Interactive Dashboard

## Current Objective
Move into Phase 11 — Final Validation: recheck data quality, metrics,
leakage, alignment, uncertainty implementation, reproducibility,
documentation, and all claims made so far across the project.

## Dataset Information
ROI: rectangular bounding box [76.75, 28.30, 77.60, 28.95] (Delhi NCR,
approximate - not administrative boundary).
Temporal comparison: 2022 -> 2026 (see Phase 2 notes for why 2024 was
dropped).

T1: Sentinel-2 L2A, Jan 1 - Feb 15, 2022, cloud_threshold=20% +
SCL-based pixel masking. delhi_ncr_t1_2022_masked.tif
T2: Sentinel-2 L2A, Jan 1 - Feb 15, 2026, cloud_threshold=20% +
SCL-based pixel masking. delhi_ncr_t2_2026_masked.tif
Both: EPSG:4326, width 9463, height 7244, 6 bands (B2,B3,B4,B8,B11,B12).
Alignment confirmed (Phase 2): aligned=True.

## Labels (Phase 3)
Method: spectral-index-derived, weakly-supervised, manually verified on
a sample. NOT pixel-perfect ground truth - documented as such.

Change label = pixel flagged as CHANGE if delta-NDVI OR delta-NDBI
crosses an adaptive threshold: mean +/- k*std of the delta distribution,
computed over valid (non-cloud-masked) pixels only, recomputed and
logged every run.

FINAL thresholds used (after manual review):
  ndvi_std_multiplier = 2.2   (raised from initial 1.5)
  ndbi_std_multiplier = 1.5   (unchanged)
  ndvi_delta_mean = -0.0387, ndvi_delta_std = 0.1833, ndvi_abs_threshold = 0.4032
  ndbi_delta_mean =  0.0218, ndbi_delta_std = 0.1421, ndbi_abs_threshold = 0.2132
  valid_pixel_fraction = 0.9955
  change_pixel_fraction_of_valid = 0.1291 (12.9% of valid pixels flagged as changed)

Rationale for raising ndvi_std_multiplier from 1.5 to 2.2: manual
verification chip review (stratified low/medium/high change-fraction
sample, 45 chips) showed the initial threshold produced scattered
salt-and-pepper "change" speckle across agricultural field parcels in
several HIGH-stratum chips, consistent with seasonal crop-cycle NDVI
variation rather than real land-cover change. Raising the multiplier
reduced (did not eliminate) this. Residual speckle noise remains in a
minority of high-stratum chips - KNOWN LIMITATION, accepted for
portfolio scope, to be disclosed honestly in README/report (Phase 12).
A connected-component blob-size filter (remove_speckle_noise()) was
designed as a proper fix but NOT implemented - deferred.

Low and medium stratum chips reviewed manually looked good - red
change overlays matched real visible T1->T2 differences (new
construction, settlement edge expansion).

Verification artifacts saved to Drive:
  {DRIVE_DIR}/reports/label_verification/*.png (45 chips)
  {DRIVE_DIR}/reports/label_verification/verification_log.csv

Patch generation: reused preprocess.generate_patches() grid exactly
(same patch_size/stride) so image patches and label patches align
row/col-for-row/col. patch_size=256, min_valid_fraction=0.5.
Usable patches (>=50% valid pixels): 1008 total.

## Train/Val/Test Split (Phase 3)
Method: geographically separated by COLUMN STRIPE (not random
per-patch), with a buffer zone (=patch_size, 256px) between stripes to
prevent adjacent-patch leakage. Layout: [TRAIN | buffer | VAL | buffer
| TEST] along raster width.

ACTUAL split achieved (NOT exactly the requested 70/15/15):
  train: 700 patches (69.4%), mean_change_fraction = 0.1276
  val:   140 patches (13.9%), mean_change_fraction = 0.1450
  test:   56 patches ( 5.6%), mean_change_fraction = 0.1422
  dropped_buffer: 112 patches (excluded, sit in buffer zones)

Change-fraction distributions are consistent across all three splits
(0.11-0.15 range) - no concerning distribution shift between train/val/
test, which is the more important check than hitting exact percentages.

Split manifest saved to: {DRIVE_DIR}/data_labels_split_manifest.json

## Model Information
Two models trained/evaluated on the identical split/loss/patch cache:
BaselineChangeCNN (Phase 4/7) and SiameseUNet (Phase 5). BaselineChangeCNN
is the LOCKED model - carried forward into Phase 7 (MC Dropout, now
complete) and all subsequent phases. As of Phase 7, BaselineChangeCNN
includes Dropout2d(p=0.3) after each conv block (architecture change
from Phase 4/6 - required a retrain, see below). Compute: Google Colab
free GPU. Patch size locked at 256x256.

## Baseline Model Results — Phase 4/6 (NO dropout, superseded for
## inference but checkpoint preserved for reference)
Architecture: shallow encoder-decoder CNN, 12-channel input (T1+T2
concatenated), skip connections, single binary change-logit output.
NO dropout layers.
VAL (epoch 20, best): F1=0.8050, IoU=0.6761, precision=0.8152, recall=0.8093
TEST (n=56): F1=0.8329, IoU=0.7144, precision=0.8185, recall=0.8519
Checkpoints (renamed in Phase 7, preserved on Drive):
  {DRIVE_DIR}/checkpoints/baseline_nodropout_best.pt
  {DRIVE_DIR}/checkpoints/baseline_nodropout_latest.pt

## Baseline Model Results — Phase 7 (WITH dropout, CURRENT/ACTIVE model)
Architecture: identical to above but Dropout2d(p=0.3) added after each
conv block (src/models/baseline.py), required full retrain (dropout
changes the computation graph - old weights incompatible).
Loss: masked BCE + Dice (0.5/0.5), label 255 excluded via mask.
Trained: 20 epochs, batch_size=8, Adam lr=1e-3, Colab T4 GPU, resume=False.
Checkpoints: {DRIVE_DIR}/checkpoints/baseline_best.pt (best val F1),
baseline_latest.pt.

VAL (epoch 18, best): F1=0.7907, IoU=0.6572
TEST (n=56): F1=0.7768, IoU=0.6359, precision=0.7734, recall=0.7849

Dropout cost ~5-6 F1 points vs the non-dropout version (test F1 0.8329
-> 0.7768) - an expected, disclosed tradeoff (dropout regularization
vs raw accuracy), not a bug. Likely amplified by the small training
set (700 patches) relative to a fairly aggressive p=0.3 dropout rate.
This dropout-enabled checkpoint is the ACTIVE model for all Phase 8+
work (characterization, spatial analysis, dashboard).

## Siamese U-Net Results (Phase 5)
Architecture: SiameseUNet, shared-weight encoder branches for T1/T2
(6-band input each, NOT concatenated), absolute-difference feature
fusion at each encoder scale, U-Net decoder with skip connections.
Same loss/split/hyperparameters as baseline for direct comparison.
Checkpoints: {DRIVE_DIR}/checkpoints/siamese_best.pt, siamese_latest.pt

VAL (epoch 16, best): F1=0.7877, IoU=0.6534, precision=0.7887, recall=0.8074
TEST (n=56): F1=0.8243, IoU=0.7020, precision=0.8092, recall=0.8476

## Baseline vs Siamese U-Net Comparison (Phase 5/6, no-dropout baseline)
| Model          | Val F1 | Val IoU | Test F1 | Test IoU |
|----------------|--------|---------|---------|----------|
| Baseline CNN   | 0.8050 | 0.6761  | 0.8329  | 0.7144   |
| Siamese U-Net  | 0.7877 | 0.6534  | 0.8243  | 0.7020   |

## Phase 6 — Confusion Matrix (test set, pixel-level, no-dropout models)
Baseline CNN:   TP=398770 FP=89919  FN=66653  TN=3114674
  FP rate=0.0281, FN rate=0.1432
Siamese U-Net:  TP=396483 FP=97303  FN=68940  TN=3107290
  FP rate=0.0304, FN rate=0.1481

Siamese is worse on BOTH false positive rate AND false negative rate -
confirms Phase 5 aggregate metrics, not a precision/recall tradeoff in
a different direction. Consistent result across two independent
evaluation methods, on both val and test splits.

FP/FN error-analysis chips (T1 | ground truth | overlay with FP=blue,
FN=orange) saved to Drive for both models:
  {DRIVE_DIR}/reports/error_analysis_baseline/*.png (8 chips)
  {DRIVE_DIR}/reports/error_analysis_siamese/*.png (8 chips)

## DECISION (LOCKED)
BaselineChangeCNN is the model carried forward for Phase 7 onward.
Siamese U-Net is NOT pursued further. Rationale: consistent
underperformance across every metric on both val and test, confirmed
at pixel level by confusion matrix (worse FP rate AND worse FN rate,
not a tradeoff). Reported honestly rather than pursuing further
Siamese tuning without a compelling reason to do so.

## Phase 7 — MC Dropout Uncertainty
Added Dropout2d(p=0.3) after each conv block in BaselineChangeCNN,
requiring a retrain from scratch (dropout changes the architecture -
old checkpoints renamed to baseline_nodropout_best.pt /
baseline_nodropout_latest.pt and preserved on Drive, not deleted).
Results: see "Baseline Model Results — Phase 7" above.

MC Dropout implementation (src/inference/uncertainty.py):
mc_dropout_predict() runs N=20 stochastic forward passes with
model.train() active at inference (keeps Dropout2d sampling different
subnetworks each pass; BatchNorm also uses batch stats under
model.train() - accepted standard approximation for MC Dropout).
Computes mean probability map + std-deviation uncertainty map per
patch. Interpreted as MODEL uncertainty (disagreement across dropout
samples), NOT calibrated real-world/ground-truth uncertainty - stated
explicitly in code docstring and here.

Qualitative validation: exported 8 uncertainty chips (T1 | mean
probability | uncertainty heatmap) to
{DRIVE_DIR}/reports/uncertainty_maps/. Manually reviewed 3 - high
uncertainty consistently traces change-region BOUNDARIES and mixed-
pixel edges (e.g. road/canal margins), not scattered noise or solid
interior regions. Model is confidently certain in solid change and
solid no-change interiors, uncertain only at genuinely ambiguous
transition zones. This is the expected, correct MC Dropout behavior -
qualitatively validates the uncertainty estimates are meaningful.

## Phase 8 — Change Characterization
Rule-based categorization (src/spatial/characterization.py) applied
ONLY to pixels the LOCKED baseline model predicts as CHANGE (not
ground-truth labels). NOT a retrained multi-class model - deliberate
scope decision since Phase 3 labels are weakly-supervised and don't
support reliable multi-class ground truth.

Rule: NDBI increase beyond threshold (0.05) -> built-up expansion
(checked first, most specific category); else NDVI decrease beyond
threshold -> vegetation loss; else NDVI increase beyond threshold ->
vegetation gain; else -> other/uncertain.

## Phase 9 — Spatial Intelligence (COMPLETE)
Scaled inference across all usable patches (n=1008 across train, val,
test, and buffer stripes; 6,585.87 km² valid area) using the locked
BaselineChangeCNN (with MC Dropout, N=20 passes) and rule-based change
characterization. Executed via notebooks/04_spatial_intelligence.ipynb
in Colab with crash-resilient streaming checkpointing.

FINAL EMPIRICAL RESULTS (Full-Scene, n=1008 usable patches):
  Total Valid Area Evaluated: 6585.87 km²
  Total Changed Area (Net)  :  917.75 km² (13.94% of valid area)

Category Breakdown (Full Scene):
  - builtup_expansion : 477.22 km² (7.25% of valid | 52.00% of change)
  - vegetation_loss   :  64.62 km² (0.98% of valid |  7.04% of change)
  - vegetation_gain   : 300.66 km² (4.57% of valid | 32.76% of change)
  - other_uncertain   :  75.25 km² (1.14% of valid |  8.20% of change)

Predictive Uncertainty (MC Dropout, N=20):
  Mean Uncertainty (Changed Pixels)   : 0.0836
  Mean Uncertainty (No-Change Pixels) : 0.0238

Test-Split Sanity Check (n=56 patches):
  Test Set Change Rate   : 14.00% (consistent with locked Phase 3/6/7 rates ~12-15%)
  Test Set Built-up Rate :  7.44%
  Test Mean Uncertainty  : 0.0737

Adaptive Baseline Urban Core Cutoff (Data-Driven):
  Formula: urban_cutoff = mean(T1_NDBI) + 1.0 * std(T1_NDBI) = 0.0561
  Covers 16.6% of valid scene pixels.
  Visual verification: 6 stratified sample chips reviewed. Chips 00,
  01, 04, 05 showed clean alignment with dense built-up fabric. Chip
  03 (medium stratum, 12.8%) showed minor mask bleeding into fallow/
  green agricultural parcels in the top-right quadrant. Documented as
  an accepted weakly-supervised baseline limitation (analogous to the
  Phase 3 agricultural NDVI speckle note).

Proximity Analysis to Baseline Urban Core:
  Evaluated on 3,884,061 model-predicted built-up expansion pixels:
  Mean distance: 86.1 m | Median distance: 56.6 m
  Within 500m: 99.3% | Within 1000m: 100.0% | Beyond 2000m: 0.0%
  Bug fix verified: Original code used raw spectral delta-NDBI on patches;
  fixed to use exact model-predicted (category_map == BUILTUP_EXPANSION)
  mask directly from MC Dropout inference. Numbers shifted (85.4m ->
  86.1m mean, 50.0m -> 56.6m median, 99.1% -> 99.3% within 500m). The
  high percentage within 500m is not an artifact — it reflects genuine
  NCR peri-urban morphology, where rural abadi/village settlement clusters
  are spaced ~1-2km apart across the agricultural matrix.

Spatial Hotspot Detection (DBSCAN):
  Parameters derived from data/grid: z_threshold=1.645 (upper 95% one-tailed
  significance), eps_deg=0.0369 (derived from 2.56km patch stride to
  connect 8-neighbor tiles), min_samples=2 (minimum for multi-patch corridor).
  Results: 58 candidate hotspot patches, 9 multi-patch growth clusters,
  8 isolated point hotspots.
  NOTE: Cluster counts and total builtup km² per cluster carry inherent
  MC-Dropout-driven stochastic variance across runs (slight decimal shifts
  are expected due to stochastic sampling, not deterministic rigidity).

Framing of Vegetation Gain:
  Vegetation gain (300.66 km², 32.76% of total change) is framed
  explicitly as consistent with the Phase 3 agricultural NDVI variability
  limitation: crop-cycle phenology differences (fallow vs. planted fields
  between Jan/Feb 2022 and Jan/Feb 2026 windows across the surrounding
  agricultural matrix), NOT as an ecological or afforestation finding.

Explicitly Skipped Analyses (Documented Limitations):
  - Road proximity: Skipped due to lack of OSM/SHP/GeoJSON road vector
    data in the repository/data directories (documented as a data limitation,
    avoiding data fabrication).
  - Administrative / NCR Sub-Region breakdown: Skipped due to lack of
    administrative boundary vectors and strict adherence to the non-political,
    continuous physical/geospatial framing.

Resume Mechanism Verification:
  Tested on actual partial cache (FORCE_FRESH_RUN=True interrupted mid-run,
  then FORCE_FRESH_RUN=False successfully printed "Resuming: found 179
  already processed patches" and continued from patch 200 to 1008 without
  restarting or duplicating). (Note: an earlier test was invalid as it
  picked up a stale full cache; this is now verified on a real partial run).

Exported Artifacts on Drive:
  {DRIVE_DIR}/reports/spatial/patch_spatial_metrics.csv
  {DRIVE_DIR}/reports/spatial/full_scene_spatial_summary.json
  {DRIVE_DIR}/reports/spatial/test_split_sanity_check.json
  {DRIVE_DIR}/reports/spatial/urban_proximity_summary.json
  {DRIVE_DIR}/reports/spatial/spatial_intelligence_hotspot_maps.png
  {DRIVE_DIR}/reports/spatial/urban_core_verification/*.png (6 chips + log)
  {DRIVE_DIR}/reports/spatial/full_scene_masks.npz
  {DRIVE_DIR}/reports/spatial/patch_results_streaming.jsonl (raw per-patch
  streaming log, not required by dashboard but retained)

## Phase 10 — Interactive Dashboard (COMPLETE)
Built as a local Streamlit app (app/app.py), NOT run in Colab. Reads
the aggregated Phase 9 artifacts, which were transferred from Drive to
local disk since Google Drive for Desktop is not installed on the
user's machine.

Artifact transfer workflow (documented for reproducibility):
  [COLAB] shutil.make_archive() zips {DRIVE_DIR}/reports/spatial/ ->
    files.download() triggers browser download of spatial_artifacts.zip
  [LOCAL] Expand-Archive -Path <downloaded zip> -DestinationPath
    data\dashboard\spatial -Force

Local cache location: data/dashboard/spatial/ (new directory, not part
of the original Phase 0 tree - required because the dashboard runs on
the local machine, not Colab, and Phase 9 outputs live on Drive).

Confirmed real schema of all Phase 9 artifacts (verified via
scripts/inspect_dashboard_artifacts.py before writing dashboard code -
no guessed column names used):
  - patch_spatial_metrics.csv: 1008 rows x 18 cols (patch_index, row,
    col, split, valid_pixels, change_pixels, no_change_pixels,
    builtup_pixels, veg_loss_pixels, veg_gain_pixels, other_pixels,
    change_fraction, builtup_fraction, mean_uncertainty_all/change/
    nochange, centroid_x, centroid_y)
  - full_scene_spatial_summary.json, test_split_sanity_check.json,
    urban_proximity_summary.json: flat summary dicts
  - spatial_intelligence_hotspot_maps.png: 3398x1138 RGBA static figure
  - full_scene_masks.npz: ONLY key 'builtup_mask', shape (7244, 9463),
    dtype bool, file size 2.2MB. NO change mask or uncertainty raster
    was retained on disk from Phase 9 - only patch-aggregated stats
    exist for those. Dashboard discloses this explicitly rather than
    fabricating a substitute.

Dashboard architecture (app/app.py):
  6 tabs: Overview, Interactive Map, Uncertainty, Hotspots & Proximity,
  Built-up Mask, Methodology & Limitations.
  - Overview: KPI cards (valid area, net changed area, patch count,
    mean uncertainty by class) + category pie/bar charts + test-split
    sanity metrics.
  - Interactive Map: Folium map with real georeferenced patch
    footprints (rectangles, not point markers) sized from the actual
    centroid spacing in the data (not a hardcoded assumption). Basemap
    switcher: Satellite (Esri World Imagery tiles, free, no API key),
    Light (CartoDB positron), Dark (CartoDB dark_matter). Color-by
    selector: change_fraction / builtup_fraction / mean_uncertainty_all
    / dominant_category. Fullscreen control enabled.
  - Uncertainty: mean-by-class bar chart, per-patch uncertainty
    histogram, change-fraction vs uncertainty scatter colored by
    dominant category.
  - Hotspots & Proximity: embeds the Phase 9 static hotspot PNG,
    proximity percentile bar chart, disclosure note on the 500m/
    peri-urban-morphology interpretation.
  - Built-up Mask: downsampled (~15x) built-up mask image rendered
    from full_scene_masks.npz. Includes an explicit, NOT-YET-RESOLVED
    disclosure about a rectangular near-zero region visible in the
    mask (see Known Issues / Open Question below) - phrased as
    "under investigation," not asserted as either real signal or
    artifact.
  - Methodology & Limitations: full written summary of data/labels/
    split/model/uncertainty/characterization methodology plus every
    disclosed limitation carried over from Phases 3-9.
  Styling: custom CSS (Inter font, gradient header card with sensor/
  date/model badges, styled KPI cards, styled tab bar, blue-accented
  "disclosure box" callouts used for every honesty/limitation note).
  All Plotly charts use the `plotly_dark` template for visual
  consistency with the dark UI theme.

Dependencies added: streamlit, plotly, folium, streamlit-folium,
branca, pillow (all appended to requirements.txt).

USER CONFIRMATION: user ran `streamlit run app\app.py` locally,
reviewed all 6 tabs, confirmed the app renders correctly and is
"smooth enough" - Phase 10 accepted as complete.

## Important Decisions (LOCKED)
- Compute: Google Colab, free-tier GPU. Checkpoints saved to Drive.
- Data source: Google Earth Engine Python API.
- Labels: spectral-index-derived (delta-NDVI / delta-NDBI adaptive
  threshold), manually verified on a representative stratified sample.
  Documented as weakly-supervised, not pixel-perfect ground truth.
- No political framing anywhere in this project.
- Local machine (Windows/PowerShell) handles repo/git/file structure.
  Colab notebooks handle GEE queries and heavy compute. Streamlit
  dashboard also runs locally (Phase 10) - it is a THIRD workflow
  surface, reading artifacts synced down from Drive via manual zip/
  download rather than a live Drive mount.
- Patch size: 256x256, fixed from Phase 3 onward.
- Geographic (column-stripe) train/val/test split with buffer zones,
  not random patch split - prevents spatial leakage.
- DataLoader num_workers=0 for all training/eval (Drive FUSE mount is
  unstable under concurrent worker reads - see Known Issues).
- MODEL LOCKED: BaselineChangeCNN (WITH dropout, Phase 7 version)
  carried forward from Phase 7 onward. Siamese U-Net not pursued further.
- MC Dropout: N=20 passes, model.train() at inference, mean+std maps.
  Documented as model uncertainty, not real-world calibrated uncertainty.
- Change characterization: rule-based (NDBI/NDVI threshold=0.05),
  applied post-hoc to model change predictions, NOT a trained
  multi-class classifier.
- Spatial Intelligence: full-scene 1008-patch scope (6,585.87 km²),
  adaptive data-driven thresholds for urban core (NDBI cutoff 0.0561)
  and hotspots (z >= 1.645, DBSCAN eps=0.0369 deg, min_samples=2).
  Vegetation gain framed as agricultural phenology. Road/administrative
  breakdowns skipped due to vector data absence.
- Dashboard: patch-aggregated visualization only for change/
  uncertainty (no pixel-level rasters retained for those); only
  built-up mask exists as a raw pixel array and is shown at reduced
  resolution. This limitation is disclosed in-app, not hidden.
- Dashboard basemap: Esri World Imagery (free, no key) for satellite
  view; CartoDB positron/dark_matter as alternate light/dark options.

## Files Created
(Phase 0-9 files unchanged, plus:)
- app/app.py (Streamlit dashboard, v2 - custom CSS theming, real
  georeferenced patch-rectangle map with satellite basemap, 6 tabs)
- scripts/inspect_dashboard_artifacts.py (one-time diagnostic script,
  confirmed real schema of all Phase 9 exports before dashboard was
  written against them - not part of the production pipeline)
- data/dashboard/spatial/ (local cache dir; contains all Phase 9
  exports transferred from Drive via Colab zip + manual download)

## Known Issues
(Phase 0-9 issues unchanged, plus:)
- Colab free-tier RAM: full T1/T2 arrays as float64 (~3.3GB each) plus
  intermediate arrays caused OOM crash. FIX: cast to float32 on read.
- Residual agricultural NDVI speckle noise in a minority of high-change
  stratum patches - accepted limitation, not fixed.
- geographic_split() produced test=5.6% instead of requested 15%
  (patch availability non-uniform across raster width). Test set
  (n=56) is small - treat test metrics with appropriate caution.
- Colab runtime disconnects/resets on idle wipe ALL Python state even
  though old cell outputs remain visible. FIX: consolidated notebook
  into two robust cells with idempotent loading checks.
- Checkpoint key mismatch: torch.save in train.py used "model_state"
  rather than "model_state_dict". FIX: checkpoint.get("model_state", ...).
- Proximity analysis mask bug: initially used raw spectral delta-NDBI
  on patches; fixed to use model-predicted BUILTUP_EXPANSION mask directly.
- Chip 03 urban core mask bleeding: adaptive T1 NDBI cutoff (0.0561)
  exhibits minor bleeding into fallow/green agricultural plots in 1 of 6
  verification chips — accepted weakly-supervised limitation.
- Hotspot cluster metrics exhibit slight run-to-run variation due to
  inherent MC Dropout stochastic sampling (not a deterministic bug).
- Google Drive FUSE mount unstable under concurrent DataLoader worker
  reads. FIX: num_workers=0 everywhere.
- Google Drive for Desktop is NOT installed on the user's local
  machine - Phase 10 artifact sync uses a manual Colab-zip ->
  browser-download -> Expand-Archive workflow instead of a live
  Drive mount. Works fine, just an extra manual step vs. the ideal
  live-sync path; documented for reproducibility.
- full_scene_masks.npz retains ONLY the built-up mask, not change or
  uncertainty rasters - a scope gap from Phase 9's export step, not
  from Phase 10. Dashboard cannot show pixel-level change/uncertainty
  maps as a result; discloses this rather than fabricating a substitute.
- OPEN / UNRESOLVED: the downsampled built-up mask (Dashboard, Built-up
  Mask tab) shows a sharp-edged rectangular near-zero region in the
  top-left of the scene. NOT yet determined whether this reflects (a)
  a genuinely low-built-up area (e.g. agricultural/floodplain land), or
  (b) a nodata/coverage gap in the source Sentinel-2 tile that reads as
  "False" because no separate valid-pixel mask was exported alongside
  builtup_mask. A diagnostic check was proposed (cross-referencing
  valid_pixels from patch_spatial_metrics.csv for patches in that
  row/col region) but the result has not yet been reported back.
  Dashboard currently flags this region as "under investigation" rather
  than asserting either explanation - do not change this wording to a
  confident claim until the diagnostic is actually run and confirmed.

## Commands Already Run
(Phase 0-9 unchanged, plus:)
- Phase 10: created data/dashboard/spatial/ and scripts/ locally, wrote
  inspect_dashboard_artifacts.py, ran it against the transferred Phase 9
  artifacts to confirm real schema, wrote app/app.py (v1, then v2 full
  replacement with satellite basemap + custom theming), installed
  streamlit/plotly/folium/streamlit-folium/branca/pillow, ran
  `streamlit run app\app.py` locally and confirmed all 6 tabs render
  with real data.

## Last Successful Test/Build
Phase 10 dashboard (app/app.py) launched successfully via
`streamlit run app\app.py`; user reviewed all 6 tabs (Overview,
Interactive Map, Uncertainty, Hotspots & Proximity, Built-up Mask,
Methodology & Limitations) against real Phase 9 data and confirmed
the app works smoothly. Satellite basemap, patch-rectangle map
rendering, and styled KPI/chart components all confirmed functional.

## Next Exact Step
1. (Optional, recommended before Phase 11) Run the pending diagnostic
   on the built-up mask's rectangular near-zero region:
   `python -c "import pandas as pd; df = pd.read_csv('data/dashboard/spatial/patch_spatial_metrics.csv'); tl = df[(df['row'] < 2000) & (df['col'] < 3500)]; print(tl[['row','col','valid_pixels','builtup_fraction','change_fraction']].describe())"`
   and report the output back so the dashboard's mask-tab disclosure
   can be finalized with a confirmed explanation instead of "under
   investigation."
2. Start Phase 11 — Final Validation: recheck data quality, metrics,
   leakage, alignment, uncertainty implementation, reproducibility,
   documentation, and all claims made across Phases 0-10 before moving
   to Phase 12 (portfolio polish / README / final report).

## Anything to know before continuing
This project spans THREE environments now: local Windows machine
(repo/git/structure + Phase 10 Streamlit dashboard), Google Colab
(training/GEE queries/heavy compute), and Google Drive (storage layer
bridging the two, accessed locally via manual zip/download since Drive
for Desktop isn't installed). Commands are always labeled
[LOCAL - PowerShell] or [COLAB - Notebook cell].
GitHub repo: https://github.com/karan02566-prog/delhi-ncr-satellite-change
GEE project ID: stellar-stream-492412-p9
IMPORTANT: watch for Google account mismatches between Drive mount and
Earth Engine auth in future Colab sessions.
IMPORTANT: project is a 2022->2026 comparison (not 2024->2026).
IMPORTANT: labels are weakly-supervised/spectral-index-derived with
manual verification on a sample - must be described this way in any
README/report, never as manually verified ground truth across the full
dataset.
IMPORTANT: BaselineChangeCNN (WITH dropout, Phase 7 version) is LOCKED
as the model going forward. Siamese U-Net underperformance and the
dropout accuracy tradeoff are both real, disclosed findings - report
honestly, do not silently paper over either in later phases.
IMPORTANT: two baseline checkpoint sets exist on Drive -
baseline_best.pt/baseline_latest.pt (WITH dropout, ACTIVE/current) and
baseline_nodropout_best.pt/baseline_nodropout_latest.pt (reference
only, from Phase 4/6). Always load baseline_best.pt (not the
nodropout version) for any Phase 8+ work unless explicitly comparing.
IMPORTANT: vegetation_gain (300.66 km², 32.76% of total change) must
be reported as agricultural crop-cycle phenology variability, not as an
ecological finding.
IMPORTANT: the Phase 10 dashboard is functionally complete and
user-confirmed, but has one open, unresolved data question (the
rectangular near-zero built-up-mask region) - do not silently resolve
this in Phase 11/12 documentation without the diagnostic actually
having been run and its result reported.
IMPORTANT: Git commit for Phase 10 (`feat: add change intelligence
dashboard`) has NOT yet been confirmed as run - verify with user before
assuming it's committed, per the project's "never push automatically
unless asked" workflow rule.