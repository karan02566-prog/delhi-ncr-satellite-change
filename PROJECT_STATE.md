# PROJECT STATE

## Current Phase
PHASE 2 — Preprocessing (COMPLETE)

## Completed Phases
- Phase 0: Project Initialization
- Phase 1: Data Acquisition
- Phase 2: Preprocessing

## Current Objective
Move into Phase 3 - building ground-truth change labels from the
aligned, cloud-masked T1/T2 composites.

## Dataset Information
ROI: rectangular bounding box [76.75, 28.30, 77.60, 28.95] (Delhi NCR,
approximate - not administrative boundary).

IMPORTANT: Temporal comparison changed from the original 2024->2026 to
2022->2026, because Jan-Feb 2024 had too few cloud-free Sentinel-2
scenes (only 3) to build a usable pixel-masked composite (large nodata
gaps). Jan-Feb 2022 had 11 cloud-free scenes and produced a clean,
complete composite. This project is now a 2022 -> 2026 comparison.
Project name/README should be updated to reflect this before Phase 12
polish (noted here so it isn't forgotten).

T1 (final, used going forward): Sentinel-2 L2A, Jan 1 - Feb 15, 2022,
cloud_threshold=20% (scene-level) + SCL-based pixel-level cloud/
shadow/snow masking.
  image_count: 11
  Exported file: delhi_ncr_t1_2022_masked.tif
  CRS: EPSG:4326, width: 9463, height: 7244, count: 6 bands, dtype: float64

T2 (final, used going forward): Sentinel-2 L2A, Jan 1 - Feb 15, 2026,
cloud_threshold=20% (scene-level) + SCL-based pixel-level masking.
  image_count: 12
  Exported file: delhi_ncr_t2_2026_masked.tif
  CRS: EPSG:4326, width: 9463, height: 7244, count: 6 bands, dtype: float64

Alignment check (validate_alignment): same_crs=True, same_width=True,
same_height=True, same_transform=True, aligned=True. T1 and T2 are
pixel-for-pixel comparable.

Bands: B2, B3, B4, B8, B11, B12.
Original Phase 1 files (delhi_ncr_t1_2024.tif, delhi_ncr_t2_2026.tif,
scene-level filtering only, no pixel masking) are retained in Drive
for reference but are NOT used going forward.
Visual comparison figure: phase2_rgb_comparison.png (Drive), correctly
labeled T1 - Jan-Feb 2022 / T2 - Jan-Feb 2026.

## Model Information
Not yet built. Planned: Siamese U-Net, shared-weight encoders,
BCE + Dice loss, MC Dropout at inference. Compute: Google Colab free
GPU. Patch size TBD in Phase 3, likely 128x128 or 256x256 given
Colab VRAM constraints.

## Important Decisions (LOCKED)
- Compute: Google Colab, free-tier GPU. Checkpoints saved to Drive.
- Data source: Google Earth Engine Python API.
- Labels: spectral-index-derived (delta-NDVI / delta-NDBI threshold),
  manually verified on a representative sample.
- No political framing anywhere in this project.
- Local machine (Windows/PowerShell) handles repo/git/file structure.
  Colab notebooks handle GEE queries and heavy compute.

## Files Created
- requirements.txt, .gitignore, README.md, PROJECT_STATE.md
- Full folder structure (data/, src/, notebooks/, app/, outputs/,
  reports/, tests/) and src/**/__init__.py files
- src/data/gee_download.py (ROI, collection queries, cloud masking,
  metadata, median composite, export)
- src/data/preprocess.py (load_raster_info, validate_alignment,
  normalize_bands, generate_patches)
- notebooks/00_colab_setup.ipynb
- notebooks/01_data_acquisition.ipynb
- notebooks/02_preprocessing.ipynb

## Files Modified
- src/data/gee_download.py (added mask_s2_clouds,
  get_sentinel2_collection_masked after Phase 1)

## Known Issues
- glob.glob() with recursive=True on mounted Drive paths in Colab
  intermittently failed to find files confirmed to exist via `find`.
  Workaround: use exact hardcoded paths instead of glob when this
  happens. Not yet root-caused.
- Encountered a Google account mismatch during Phase 1 Drive export -
  resolved via ee.Authenticate(force=True) with explicit account
  selection. Watch for this every new Colab session.
- Multiple Colab runtime/session resets occurred during Phase 2,
  requiring re-running setup cells (Drive mount, GEE auth, git clone/
  pull) each time. Established pattern: always verify
  '/content/delhi-ncr-satellite-change' in sys.path before running any
  src.* imports if a session gap occurred; re-run the git clone/pull
  cell if a ModuleNotFoundError for 'src' appears.
- rasterio/geopandas may need conda instead of pip on Windows if pip
  install fails (did not occur, noted for reference).
- .ipynb files written via PowerShell Out-File -Encoding utf8 include
  a BOM that breaks Python's local json.load(); fix by rewriting with
  [System.IO.File]::WriteAllText(path, content,
  [System.Text.UTF8Encoding]::new($false)).

## Commands Already Run
- Phase 0: venv setup, git init, GEE account registration, GitHub
  repo creation and push.
- Phase 1: queried Sentinel-2 L2A via GEE for T1 (2024) and T2 (2026),
  built median composites, exported to Drive, validated.
- Phase 2: rebuilt T1 (switched to 2022) and T2 (2026) composites with
  pixel-level SCL cloud masking, exported to Drive, confirmed
  COMPLETED, validated alignment (all checks True), generated RGB
  comparison visualization.

## Last Successful Test/Build
Both masked GeoTIFFs (T1 2022, T2 2026) confirmed present in Drive,
alignment fully verified (aligned=True), RGB comparison figure
generated and visually confirms clean, near-complete coverage for
both periods.

## Next Exact Step
Start Phase 3 - Ground truth / labels. Implement spectral-index-
derived change labels (delta-NDVI, delta-NDBI thresholding) using the
two aligned, masked composites, manually verify labels on a
representative sample, then perform a geographically separated
train/val/test split (patches from the same neighborhood must not
leak across splits - use the row/col metadata from generate_patches()
for this).

## Anything to know before continuing
This project spans two environments: local Windows machine (repo/git/
structure) and Google Colab (training/GEE queries). Commands are
always labeled [LOCAL - PowerShell] or [COLAB - Notebook cell].
GitHub repo: https://github.com/karan02566-prog/delhi-ncr-satellite-change
GEE project ID: stellar-stream-492412-p9
IMPORTANT: watch for Google account mismatches between Drive mount and
Earth Engine auth in future Colab sessions.
IMPORTANT: project is now a 2022->2026 comparison (not 2024->2026).