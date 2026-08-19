# PROJECT STATE

## Current Phase
PHASE 1 — Data Acquisition (COMPLETE)

## Completed Phases
- Phase 0: Project Initialization
- Phase 1: Data Acquisition
## Current Objective
Get local repo structure, venv, git, and GEE access set up so Phase 1
(data acquisition) can start.

## Files Created
- requirements.txt
- .gitignore
- README.md
- PROJECT_STATE.md
- Full folder structure (data/, src/, notebooks/, app/, outputs/, reports/, tests/)
- src/**/__init__.py files

## Files Modified
None yet.

## Current Architecture
Not yet implemented. Planned: Siamese U-Net, shared-weight encoders,
BCE + Dice loss, MC Dropout at inference.

## Dataset Information
ROI: rectangular bounding box [76.75, 28.30, 77.60, 28.95] (Delhi NCR,
approximate — not administrative boundary).
T1: Sentinel-2 L2A, Jan 1 - Feb 15, 2024, cloud_threshold=20%.
  image_count: 3
  cloud_percentages: [7.890186, 9.201495, 1.670303]
  acquisition_dates: ['2024-01-29', '2024-02-08', '2024-02-08']
  Exported file: delhi_ncr_t1_2024.tif (206 MB)

T2: Sentinel-2 L2A, Jan 1 - Feb 15, 2026, cloud_threshold=20%.
  image_count: 12
  cloud_percentages: [18.53, 7.39, 1.12, 13.18, 1.65, 17.45, 5.77,
    12.52, 17.52, 2.82, 13.30, 0.001]
  acquisition_dates: ['2026-01-05' (x2), '2026-01-18' (x3),
    '2026-02-07' (x7)]
  Exported file: delhi_ncr_t2_2026.tif (968 MB)

Bands: B2, B3, B4, B8, B11, B12.
Both files stored in Google Drive: delhi_ncr_change_detection/ folder.

## Model Information
Not yet built.

## Important Decisions (LOCKED)
- Compute: Google Colab, free-tier GPU. Checkpoints saved to Google Drive.
- Data source: Google Earth Engine Python API (not manual Copernicus downloads).
- Labels: spectral-index-derived (ΔNDVI / ΔNDBI threshold), manually verified
  on a representative sample — not full manual pixel labeling.
- No political framing anywhere in this project. Purely technical framing:
  "satellite-observed change, 2024–2026."
- Local machine (Windows/PowerShell) handles repo/git/file structure.
  Colab notebooks handle actual training/heavy compute.

## Known Issues
- rasterio/geopandas may need conda instead of pip on Windows if pip install fails.
- T1 (2024) had noticeably fewer cloud-free images (3) than T2 (2026,
  12 images) over the same-length window. Likely genuine winter
  haze/fog variation year-to-year in Delhi, not a pipeline bug — worth
  noting honestly in the eventual data validation writeup rather than
  treated as an error.
- Encountered a Google account mismatch during Drive export: ee.Authenticate()
  initially authorized a different Google account than the one used for
  Drive browsing, causing exports to succeed but be invisible in the
  expected Drive account. Fixed via ee.Authenticate(force=True) and
  explicit account selection. If resuming in a new Colab session, watch
  for this again — confirm the account picker matches your intended
  Drive account before proceeding with exports.


## Commands Already Run
- (Phase 0 commands)
- Queried Sentinel-2 L2A via GEE for T1 and T2 windows
- Built median composites, visually verified in geemap
- Exported both composites to Google Drive as GeoTIFFs (confirmed
  206 MB and 968 MB respectively)

## Last Successful Test/Build
Both T1 and T2 GeoTIFF exports confirmed present in Google Drive with
correct, non-trivial file sizes. Data acquisition pipeline verified
working end-to-end.

## Next Exact Step
Start Phase 2 — Preprocessing. Tasks: align T1/T2, pixel-level cloud
masking (not just scene-level filtering), crop to exact ROI, band
normalization, generate RGB + NIR composites, validate spatial
alignment, generate patches for model training. Create
notebooks/02_preprocessing... or continue in 01 - decide when starting
the phase. First need to download/access the two GeoTIFFs from Drive
into the Colab working session for processing.

## Anything to know before continuing
This project spans two environments: local Windows machine (repo/git/
structure) and Google Colab (training/GEE queries). Commands are always
labeled [LOCAL - PowerShell] or [COLAB - Notebook cell] to avoid confusion.
GitHub repo: https://github.com/karan02566-prog/delhi-ncr-satellite-change
GEE project ID: stellar-stream-492412-p9
IMPORTANT: watch for Google account mismatches between Drive mount and
Earth Engine auth in future Colab sessions (see Known Issues above).