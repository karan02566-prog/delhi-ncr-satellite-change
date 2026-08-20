# PROJECT STATE

## Current Phase
PHASE 3 — Ground Truth / Labels (COMPLETE)

## Completed Phases
- Phase 0: Project Initialization
- Phase 1: Data Acquisition
- Phase 2: Preprocessing
- Phase 3: Ground Truth / Labels

## Current Objective
Move into Phase 4 - Baseline model (simple CNN or image-difference
method, trained on Colab GPU, to establish a performance floor before
building the Siamese U-Net).

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
designed as a proper fix but NOT implemented - deferred, revisit only
if Phase 6 evaluation results are notably hurt by label noise.

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
Not yet built. Planned: Siamese U-Net, shared-weight encoders,
BCE + Dice loss, MC Dropout at inference. Baseline (simple CNN or
image-difference method) planned FIRST in Phase 4. Compute: Google
Colab free GPU. Patch size locked at 256x256.

## Important Decisions (LOCKED)
- Compute: Google Colab, free-tier GPU. Checkpoints saved to Drive.
- Data source: Google Earth Engine Python API.
- Labels: spectral-index-derived (delta-NDVI / delta-NDBI adaptive
  threshold), manually verified on a representative stratified sample.
  Documented as weakly-supervised, not pixel-perfect ground truth.
- No political framing anywhere in this project.
- Local machine (Windows/PowerShell) handles repo/git/file structure.
  Colab notebooks handle GEE queries and heavy compute.
- Patch size: 256x256, fixed from Phase 3 onward.
- Geographic (column-stripe) train/val/test split with buffer zones,
  not random patch split - prevents spatial leakage.

## Files Created
(Phase 0-2 files unchanged, plus:)
- src/data/label_generation.py (compute_ndvi, compute_ndbi,
  compute_valid_mask, compute_change_labels, label_patches,
  select_verification_sample, export_verification_chips,
  geographic_split, summarize_split, save_split_manifest)
- notebooks/03_label_generation.ipynb

## Known Issues
(Phase 0-2 issues unchanged, plus:)
- Colab free-tier RAM: loading full T1/T2 arrays as float64 (~3.3GB
  each) plus intermediate arrays in compute_change_labels caused an
  out-of-memory crash. FIX APPLIED: cast to float32 on read
  (astype(np.float32) immediately after src.read()). Apply this in any
  future notebook loading these full-scene GeoTIFFs.
- Residual agricultural NDVI speckle noise in a minority of high-change
  stratum patches - accepted limitation, not fixed. remove_speckle_noise()
  filter design exists but is NOT implemented in the codebase.
- geographic_split() produced test=5.6% instead of requested 15% because
  patch availability isn't perfectly uniform across raster width.
  Change-fraction distribution still consistent across splits, accepted
  as-is. Revisit by rebalancing split fractions if Phase 6 test metrics
  look unstable due to small n=56.
- Colab runtime disconnects/resets on idle wipe ALL Python state even
  though old cell outputs remain visible. If any NameError/
  ModuleNotFoundError appears for something that worked minutes earlier,
  assume session reset - rerun every cell from the top.

## Commands Already Run
(Phase 0-2 unchanged, plus:)
- Phase 3: computed change labels via compute_change_labels (initial
  ndvi_std_multiplier=1.5, revised to 2.2 after manual review),
  generated 1008 usable label patches, exported 45 stratified
  verification chips + CSV log to Drive, manually reviewed a subset of
  chips, generated geographic train/val/test split (700/140/56), saved
  split manifest to Drive.

## Last Successful Test/Build
Split manifest saved successfully. summarize_split() confirmed
consistent change-fraction distribution across train/val/test.
Verification chips manually reviewed - low/medium stratum labels look
accurate; high stratum has some accepted agricultural noise.

## Next Exact Step
Start Phase 4 - Baseline model. Implement a simple CNN or
image-difference baseline, build the training loop with checkpointing
to Drive, compute precision/recall/F1/IoU on the val split. Use
patch_size=256, the split from data_labels_split_manifest.json, and
treat label value 255 (invalid/cloud-masked) as an ignore-index in the
loss - do NOT treat it as no-change.

## Anything to know before continuing
This project spans two environments: local Windows machine (repo/git/
structure) and Google Colab (training/GEE queries). Commands are
always labeled [LOCAL - PowerShell] or [COLAB - Notebook cell].
GitHub repo: https://github.com/karan02566-prog/delhi-ncr-satellite-change
GEE project ID: stellar-stream-492412-p9
IMPORTANT: watch for Google account mismatches between Drive mount and
Earth Engine auth in future Colab sessions.
IMPORTANT: project is a 2022->2026 comparison (not 2024->2026).
IMPORTANT: labels are weakly-supervised/spectral-index-derived with
manual verification on a sample - must be described this way in any
README/report, never as manually verified ground truth across the full
dataset.