# PROJECT STATE

## Current Phase
PHASE 6 — Model Evaluation (COMPLETE)

## Completed Phases
- Phase 0: Project Initialization
- Phase 1: Data Acquisition
- Phase 2: Preprocessing
- Phase 3: Ground Truth / Labels
- Phase 4: Baseline Model
- Phase 5: Siamese U-Net
- Phase 6: Model Evaluation

## Current Objective
Move into Phase 7 - MC Dropout uncertainty quantification on the
LOCKED baseline model (see Phase 6 decision below).

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
Two models trained and evaluated on the identical split/loss/patch
cache: BaselineChangeCNN (Phase 4) and SiameseUNet (Phase 5).
BaselineChangeCNN is the LOCKED model as of Phase 6 (see decision
below) - carried forward into Phase 7 (MC Dropout) and all subsequent
phases. Compute: Google Colab free GPU. Patch size locked at 256x256.

## Baseline Model Results (Phase 4)
Architecture: shallow encoder-decoder CNN (BaselineChangeCNN), 12-channel
input (T1 6-band + T2 6-band concatenated), skip connections, single
binary change-logit output.
Loss: masked BCE + Dice (0.5/0.5 weight), label 255 excluded via mask.
Trained: 20 epochs, batch_size=8, Adam lr=1e-3, Colab T4 GPU.
Checkpoints: {DRIVE_DIR}/checkpoints/baseline_best.pt (best val F1),
baseline_latest.pt (resume-safe, every epoch).

VAL (epoch 20, best): F1=0.8050, IoU=0.6761, precision=0.8152, recall=0.8093
TEST (n=56, held out): F1=0.8329, IoU=0.7144, precision=0.8185, recall=0.8519

## Siamese U-Net Results (Phase 5)
Architecture: SiameseUNet, shared-weight encoder branches for T1/T2
(6-band input each, NOT concatenated), absolute-difference feature
fusion at each encoder scale, U-Net decoder with skip connections.
Same loss/split/hyperparameters as baseline for direct comparison.
Checkpoints: {DRIVE_DIR}/checkpoints/siamese_best.pt, siamese_latest.pt

VAL (epoch 16, best): F1=0.7877, IoU=0.6534, precision=0.7887, recall=0.8074
TEST (n=56, held out): F1=0.8243, IoU=0.7020, precision=0.8092, recall=0.8476

## Baseline vs Siamese U-Net Comparison (Phase 5/6)
| Model          | Val F1 | Val IoU | Test F1 | Test IoU |
|----------------|--------|---------|---------|----------|
| Baseline CNN   | 0.8050 | 0.6761  | 0.8329  | 0.7144   |
| Siamese U-Net  | 0.7877 | 0.6534  | 0.8243  | 0.7020   |

## Phase 6 — Confusion Matrix (test set, pixel-level)
Baseline CNN:   TP=398770 FP=89919  FN=66653  TN=3114674
  FP rate=0.0281, FN rate=0.1432
Siamese U-Net:  TP=396483 FP=97303  FN=68940  TN=3107290
  FP rate=0.0304, FN rate=0.1481

Siamese is worse on BOTH false positive rate AND false negative rate -
confirms Phase 5 aggregate metrics, not a precision/recall tradeoff in
a different direction. Consistent result across two independent
evaluation methods (aggregate F1/IoU, and pixel-level confusion
matrix), on both val and test splits.

FP/FN error-analysis chips (T1 | ground truth | overlay with FP=blue,
FN=orange) saved to Drive for both models:
  {DRIVE_DIR}/reports/error_analysis_baseline/*.png (8 chips)
  {DRIVE_DIR}/reports/error_analysis_siamese/*.png (8 chips)

## DECISION (LOCKED)
BaselineChangeCNN is the model carried forward for Phase 7 (MC Dropout)
and all subsequent phases (8-12). Siamese U-Net is NOT pursued further.
Rationale: consistent underperformance across every metric (F1, IoU,
precision, recall) on both val and test, confirmed at pixel level by
confusion matrix (worse FP rate AND worse FN rate, not a tradeoff).
Plausible causes: small dataset (700 train patches) may favor the
baseline's simpler joint-context architecture; Phase 3 label noise
(agricultural NDVI speckle) likely caps achievable performance
similarly for both, reducing any advantage complexity could offer.
Per project working rules, this is reported honestly rather than
pursuing further Siamese tuning without a compelling reason to do so.

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
- DataLoader num_workers=0 for all training/eval (Drive FUSE mount is
  unstable under concurrent worker reads - see Known Issues).
- MODEL LOCKED: BaselineChangeCNN carried forward from Phase 6 onward.
  Siamese U-Net not pursued further (see DECISION above).

## Files Created
(Phase 0-2 files unchanged, plus:)
- src/data/label_generation.py (compute_ndvi, compute_ndbi,
  compute_valid_mask, compute_change_labels, label_patches,
  select_verification_sample, export_verification_chips,
  geographic_split, summarize_split, save_split_manifest)
- notebooks/03_label_generation.ipynb
- src/data/patch_cache.py (build_patch_cache)
- src/data/patch_dataset.py (ChangeDetectionPatchDataset - concatenated
  12-channel input for baseline; SiamesePatchDataset - separate T1/T2
  6-channel tensors for Siamese U-Net. Both regenerate labels on-the-fly
  via compute_change_labels for consistency with Phase 3 logic.)
- src/models/baseline.py (BaselineChangeCNN)
- src/models/siamese_unet.py (SharedEncoder, SiameseUNet)
- src/training/losses.py (masked_bce_dice_loss)
- src/training/metrics.py (compute_change_metrics)
- src/training/train.py (train_baseline)
- src/training/train_siamese.py (train_siamese)
- src/training/evaluate.py (compute_confusion_matrix,
  print_confusion_matrix, export_fp_fn_chips)

## Known Issues
(Phase 0-2 issues unchanged, plus:)
- Colab free-tier RAM: full T1/T2 arrays as float64 (~3.3GB each) plus
  intermediate arrays caused OOM crash. FIX: cast to float32 on read.
- Residual agricultural NDVI speckle noise in a minority of high-change
  stratum patches - accepted limitation, not fixed.
- geographic_split() produced test=5.6% instead of requested 15%
  (patch availability non-uniform across raster width). Test set
  (n=56) is small - treat test metrics with appropriate caution.
- Colab runtime disconnects/resets on idle wipe ALL Python state even
  though old cell outputs remain visible. Assume session reset on any
  unexpected NameError/ModuleNotFoundError - rerun from the top.
- Google Drive FUSE mount unstable under concurrent DataLoader worker
  reads. FIX: num_workers=0 everywhere.
- Python module caching in Colab: git pull updating an already-imported
  .py file requires importlib.reload() or Restart Session to pick up
  changes (new names/classes won't appear via plain re-import).
- Common session bug encountered in Phase 6: reusing a bare `model`
  variable name across both baseline and Siamese training left it
  bound to whichever was trained/loaded last, causing a
  TypeError (SiameseUNet.forward() missing 't2') when accidentally
  passed to code expecting the baseline's single-input forward. FIX:
  always load evaluation models into explicitly distinct variable
  names (model_base, model_siam), never reuse a generic `model`.
- Siamese U-Net underperformed baseline CNN on both val/test aggregate
  metrics AND pixel-level confusion matrix (worse FP and FN rate) -
  not a bug, a genuine, consistently-confirmed comparison result.

## Commands Already Run
(Phase 0-2 unchanged, plus:)
- Phase 3: computed change labels, generated 1008 usable label patches,
  exported 45 verification chips, manually reviewed, generated
  geographic split (700/140/56), saved manifest to Drive.
- Phase 4: built patch cache, trained BaselineChangeCNN 20 epochs,
  evaluated val (per-epoch) and test (final) splits.
- Phase 5: trained SiameseUNet 20 epochs on same cache/split/loss,
  evaluated val/test, compared against baseline.
- Phase 6: computed pixel-level confusion matrix for both models on
  test set, exported 8 FP/FN error-analysis chips per model to Drive,
  formally locked in BaselineChangeCNN as the model going forward.

## Last Successful Test/Build
Confusion matrices computed successfully for both models on test set
(n=56 patches, 3,670,016 valid pixels). Baseline confirmed superior on
every axis (FP rate, FN rate) matching Phase 5's aggregate metrics.
Error-analysis chips exported to Drive for both models.

## Next Exact Step
Start Phase 7 - MC Dropout uncertainty. Add dropout layers to
BaselineChangeCNN (the LOCKED model). Enable dropout at inference
(call model.train() during inference to keep dropout active, NOT
model.eval()), run N stochastic forward passes per patch (start N=20),
compute mean probability map + standard deviation (uncertainty map).
Analyze whether high-uncertainty regions correlate with known trouble
spots (change-region boundaries, the accepted agricultural NDVI noise
from Phase 3 - check against the Phase 3 verification chip locations
if useful). Interpret uncertainty as MODEL uncertainty, not absolute
real-world uncertainty - state this explicitly in any documentation.

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
IMPORTANT: BaselineChangeCNN is LOCKED as the model going forward as of
Phase 6. Siamese U-Net underperformance is a real, disclosed finding -
do not silently favor Siamese U-Net in any future phase without
re-justifying it against this result.
IMPORTANT: current BaselineChangeCNN architecture (src/models/baseline.py)
has NO dropout layers yet - Phase 7 will need to add them before MC
Dropout inference is possible. This likely means retraining, or adding
dropout + fine-tuning from the existing checkpoint - decide in Phase 7.