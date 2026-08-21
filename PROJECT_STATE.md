# PROJECT STATE

## Current Phase
PHASE 5 — Siamese U-Net (COMPLETE)

## Completed Phases
- Phase 0: Project Initialization
- Phase 1: Data Acquisition
- Phase 2: Preprocessing
- Phase 3: Ground Truth / Labels
- Phase 4: Baseline Model
- Phase 5: Siamese U-Net

## Current Objective
Move into Phase 6 - Model evaluation. Formal comparison (confusion
matrix, false positive/negative spatial analysis) between baseline CNN
and Siamese U-Net on the test set, and a decision on which model to
carry forward for Phases 7-10 (MC Dropout, characterization, spatial
analysis, dashboard).

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
Two models trained and evaluated on the identical split/loss/patch
cache for direct comparison: BaselineChangeCNN (Phase 4) and
SiameseUNet (Phase 5). See results below. Baseline currently the
stronger model. MC Dropout uncertainty (Phase 7) planned on whichever
model is selected in Phase 6. Compute: Google Colab free GPU. Patch
size locked at 256x256.

## Baseline Model Results (Phase 4)
Architecture: shallow encoder-decoder CNN (BaselineChangeCNN), 12-channel
input (T1 6-band + T2 6-band concatenated), skip connections, single
binary change-logit output.
Loss: masked BCE + Dice (0.5/0.5 weight), label 255 excluded via mask.
Trained: 20 epochs, batch_size=8, Adam lr=1e-3, Colab T4 GPU.
Checkpoints: {DRIVE_DIR}/checkpoints/baseline_best.pt (best val F1),
baseline_latest.pt (resume-safe, every epoch).

VAL (epoch 20, best): F1=0.8050, IoU=0.6761, precision=0.8152, recall=0.8093
TEST (n=56, held out, real numbers): F1=0.8329, IoU=0.7144,
  precision=0.8185, recall=0.8519

## Siamese U-Net Results (Phase 5)
Architecture: SiameseUNet, shared-weight encoder branches for T1/T2
(6-band input each, NOT concatenated), absolute-difference feature
fusion at each encoder scale (e1, e2, bottleneck), U-Net decoder with
skip connections. Same loss (masked BCE+Dice), same split, same
20 epochs/batch_size=8/Adam lr=1e-3 as baseline for direct comparison.
Checkpoints: {DRIVE_DIR}/checkpoints/siamese_best.pt, siamese_latest.pt

VAL (epoch 16, best): F1=0.7877, IoU=0.6534, precision=0.7887, recall=0.8074
TEST (n=56, held out, real numbers): F1=0.8243, IoU=0.7020,
  precision=0.8092, recall=0.8476

## Baseline vs Siamese U-Net Comparison
| Model          | Val F1 | Val IoU | Test F1 | Test IoU |
|----------------|--------|---------|---------|----------|
| Baseline CNN   | 0.8050 | 0.6761  | 0.8329  | 0.7144   |
| Siamese U-Net  | 0.7877 | 0.6534  | 0.8243  | 0.7020   |

FINDING: Siamese U-Net did NOT outperform the simpler baseline on
either split - marginally behind on every metric. Reported honestly
per project rule against fabricating or cherry-picking results.
Plausible causes: small dataset (700 train patches) may favor the
baseline's simpler joint-context architecture over the Siamese design's
assumption that shared low-level features help; Phase 3 label noise
(agricultural NDVI speckle) likely caps achievable performance
similarly for both architectures, reducing any advantage complexity
could offer. BASELINE CNN is the stronger model on this dataset and
is the leading candidate to carry forward for Phase 7 MC Dropout
uncertainty quantification, pending Phase 6's more detailed error
analysis - not yet attempted to tune Siamese further (lr schedule,
more epochs, augmentation).

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
- DataLoader num_workers=0 for all training (Drive FUSE mount is
  unstable under concurrent worker reads - see Known Issues).

## Files Created
(Phase 0-2 files unchanged, plus:)
- src/data/label_generation.py (compute_ndvi, compute_ndbi,
  compute_valid_mask, compute_change_labels, label_patches,
  select_verification_sample, export_verification_chips,
  geographic_split, summarize_split, save_split_manifest)
- notebooks/03_label_generation.ipynb
- src/data/patch_cache.py (build_patch_cache - caches T1/T2 patches
  as .npz to Drive per split manifest)
- src/data/patch_dataset.py (ChangeDetectionPatchDataset - concatenated
  12-channel input for baseline; SiamesePatchDataset - separate T1/T2
  6-channel tensors for Siamese U-Net. Both regenerate labels on-the-fly
  via compute_change_labels for consistency with Phase 3 logic.)
- src/models/baseline.py (BaselineChangeCNN)
- src/models/siamese_unet.py (SharedEncoder, SiameseUNet)
- src/training/losses.py (masked_bce_dice_loss - excludes label 255)
- src/training/metrics.py (compute_change_metrics - precision/recall/
  F1/IoU for the change class, masking invalid pixels)
- src/training/train.py (train_baseline - epoch loop, checkpointing,
  resume support)
- src/training/train_siamese.py (train_siamese - same structure,
  dual-input forward signature)

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
  as-is. Test set (n=56) is small - Phase 6 should treat test metrics
  with appropriate caution given limited sample size.
- Colab runtime disconnects/resets on idle wipe ALL Python state even
  though old cell outputs remain visible. If any NameError/
  ModuleNotFoundError appears for something that worked minutes earlier,
  assume session reset - rerun every cell from the top.
- Google Drive FUSE mount is unstable under concurrent DataLoader worker
  reads (OSError/ConnectionAbortedError with num_workers>0). FIX APPLIED:
  num_workers=0 in all DataLoaders. Slower but stable. Apply to any new
  DataLoader usage.
- Python module caching in Colab: after `git pull` updates a .py file
  that was already imported earlier in the session, plain re-import
  will NOT pick up changes. FIX: use importlib.reload() on the specific
  module, or Restart Session for a clean reload, before importing new
  names added to an already-imported module.
- Siamese U-Net underperformed the baseline CNN on both val and test
  (see comparison table above) - not a bug, a genuine architecture
  comparison result, reported as-is.

## Commands Already Run
(Phase 0-2 unchanged, plus:)
- Phase 3: computed change labels via compute_change_labels (initial
  ndvi_std_multiplier=1.5, revised to 2.2 after manual review),
  generated 1008 usable label patches, exported 45 stratified
  verification chips + CSV log to Drive, manually reviewed a subset of
  chips, generated geographic train/val/test split (700/140/56), saved
  split manifest to Drive.
- Phase 4: built patch cache (.npz per split) to Drive, trained
  BaselineChangeCNN for 20 epochs on Colab T4 GPU, evaluated on val
  (per-epoch) and test (final, once) splits.
- Phase 5: trained SiameseUNet for 20 epochs on the same patch cache/
  split/loss, evaluated on val (per-epoch) and test (final, once)
  splits, compared directly against baseline.

## Last Successful Test/Build
Both baseline and Siamese U-Net trained successfully for 20 epochs
each on Colab T4 GPU with no crashes (after fixes: float32 casting,
num_workers=0, importlib.reload for module cache issues). Test-set
metrics computed for both models (n=56 held-out patches). Comparison
table confirms baseline CNN is currently the stronger model.

## Next Exact Step
Start Phase 6 - Model evaluation. For BOTH models on the test set:
confusion matrix, false positive/negative spatial visualization
(overlay FP/FN on actual T1/T2 imagery for a handful of test patches
to understand WHERE and WHY each model fails - e.g. does it correlate
with the known agricultural noise patches from Phase 3?). Formally
decide baseline vs Siamese U-Net as the model to carry forward into
Phase 7 (MC Dropout uncertainty) - default to baseline given current
results, unless error analysis reveals a compelling reason otherwise.
Document the decision and rationale in PROJECT_STATE.md.

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
IMPORTANT: Siamese U-Net underperforming baseline is a real, disclosed
finding - do not let future phases silently favor Siamese U-Net without
re-justifying it against this result.