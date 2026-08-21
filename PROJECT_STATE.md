# PROJECT STATE

## Current Phase
PHASE 7 — MC Dropout Uncertainty (COMPLETE)

## Completed Phases
- Phase 0: Project Initialization
- Phase 1: Data Acquisition
- Phase 2: Preprocessing
- Phase 3: Ground Truth / Labels
- Phase 4: Baseline Model
- Phase 5: Siamese U-Net
- Phase 6: Model Evaluation
- Phase 7: MC Dropout Uncertainty

## Current Objective
Move into Phase 8 - Change characterization (rule-based/spectral-index
categorization of detected change into built-up expansion, vegetation
loss/gain, other).

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
- MODEL LOCKED: BaselineChangeCNN (WITH dropout, Phase 7 version)
  carried forward from Phase 7 onward. Siamese U-Net not pursued further.
- MC Dropout: N=20 passes, model.train() at inference, mean+std maps.
  Documented as model uncertainty, not real-world calibrated uncertainty.

## Files Created
(Phase 0-2 files unchanged, plus:)
- src/data/label_generation.py (compute_ndvi, compute_ndbi,
  compute_valid_mask, compute_change_labels, label_patches,
  select_verification_sample, export_verification_chips,
  geographic_split, summarize_split, save_split_manifest)
- notebooks/03_label_generation.ipynb
- src/data/patch_cache.py (build_patch_cache)
- src/data/patch_dataset.py (ChangeDetectionPatchDataset,
  SiamesePatchDataset)
- src/models/baseline.py (BaselineChangeCNN - NOW includes
  Dropout2d(p=0.3) per block, updated in Phase 7)
- src/models/siamese_unet.py (SharedEncoder, SiameseUNet)
- src/training/losses.py (masked_bce_dice_loss)
- src/training/metrics.py (compute_change_metrics)
- src/training/train.py (train_baseline)
- src/training/train_siamese.py (train_siamese)
- src/training/evaluate.py (compute_confusion_matrix,
  print_confusion_matrix, export_fp_fn_chips)
- src/inference/uncertainty.py (mc_dropout_predict,
  export_uncertainty_chips)

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
- Reusing a bare `model` variable name across multiple models left it
  bound to whichever was trained/loaded last, causing a TypeError when
  passed to code expecting a different model's forward signature. FIX:
  always load models into explicitly distinct variable names
  (model_base, model_siam), never reuse a generic `model`.
- Siamese U-Net underperformed baseline CNN on both val/test aggregate
  metrics AND pixel-level confusion matrix - not a bug, a genuine,
  consistently-confirmed comparison result.
- Adding dropout to BaselineChangeCNN required a full retrain (weights
  incompatible with new architecture) and cost ~5-6 F1 points on test
  set vs the non-dropout version - expected regularization tradeoff,
  disclosed, not hidden. Non-dropout checkpoint preserved on Drive
  under baseline_nodropout_*.pt for reference/comparison only.

## Commands Already Run
(Phase 0-2 unchanged, plus:)
- Phase 3: computed change labels, generated 1008 usable label patches,
  exported 45 verification chips, manually reviewed, generated
  geographic split (700/140/56), saved manifest to Drive.
- Phase 4: built patch cache, trained BaselineChangeCNN (no dropout)
  20 epochs, evaluated val/test.
- Phase 5: trained SiameseUNet 20 epochs on same cache/split/loss,
  evaluated val/test, compared against baseline.
- Phase 6: computed pixel-level confusion matrix for both models on
  test set, exported FP/FN error-analysis chips, locked in
  BaselineChangeCNN as the model going forward.
- Phase 7: added Dropout2d(p=0.3) to BaselineChangeCNN, renamed old
  no-dropout checkpoints to preserve them, retrained 20 epochs from
  scratch (resume=False), evaluated val/test, ran MC Dropout (N=20
  passes) on test set, exported 8 uncertainty chips to Drive, manually
  reviewed 3 - confirmed uncertainty concentrates at change-region
  boundaries as expected.

## Last Successful Test/Build
MC Dropout uncertainty chips exported successfully to Drive. Manual
review of 3 chips confirmed uncertainty (std) heatmaps concentrate
along change-region boundaries and mixed-pixel edges, with low
uncertainty in solid interior regions - the expected, correct pattern,
qualitatively validating the uncertainty estimates.

## Next Exact Step
Start Phase 8 - Change characterization. Using the dropout-enabled
baseline's mean prediction map, apply rule-based/spectral-index
characterization (delta-NDVI sign/magnitude, delta-NDBI sign/magnitude)
to categorize detected change pixels as built-up expansion, vegetation
loss, vegetation gain, or other/uncertain. This is descriptive
characterization layered on top of the binary change detector, NOT a
retrained multi-class model (per project scope decision - binary core
ML + rule-based characterization layer since multi-class labels aren't
reliably achievable given Phase 3's weakly-supervised label approach).

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
IMPORTANT: BaselineChangeCNN (WITH dropout, Phase 7 version) is LOCKED
as the model going forward. Siamese U-Net underperformance and the
dropout accuracy tradeoff are both real, disclosed findings - report
honestly, do not silently paper over either in later phases.
IMPORTANT: two baseline checkpoint sets exist on Drive -
baseline_best.pt/baseline_latest.pt (WITH dropout, ACTIVE/current) and
baseline_nodropout_best.pt/baseline_nodropout_latest.pt (reference
only, from Phase 4/6). Always load baseline_best.pt (not the
nodropout version) for any Phase 8+ work unless explicitly comparing.