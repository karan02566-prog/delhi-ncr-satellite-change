# PROJECT STATE

## Current Phase
PHASE 0 — Project Initialization (COMPLETE)

## Completed Phases
- Phase 0: Project Initialization
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
Not yet acquired. Source: Sentinel-2 L2A via Google Earth Engine
(`COPERNICUS/S2_SR_HARMONIZED`). Study area: Delhi NCR. Temporal window:
2024 → 2026.

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

## Commands Already Run
- python -m venv venv
- .\venv\Scripts\Activate.ps1
- python -m pip install --upgrade pip
- git init

## Last Successful Test/Build
Venv created and activated successfully. Git repo initialized. Pip upgraded
to 26.2.1. Awaiting: requirements install, GEE account setup, Colab notebook
verification.

## Next Exact Step
Start Phase 1 — Data Acquisition. Define Delhi NCR ROI precisely,
select 2024 and 2026 temporal windows (matching season/month), query
Sentinel-2 L2A via GEE for both periods, export to Drive, verify
bands/CRS/resolution, produce a data validation report.

## Anything to know before continuing
## Anything to know before continuing
This project spans two environments: local Windows machine (repo/git/structure)
and Google Colab (training/GEE queries). Commands are always labeled
[LOCAL - PowerShell] or [COLAB - Notebook cell] to avoid confusion.
GitHub repo: https://github.com/karan02566-prog/delhi-ncr-satellite-change
GEE project ID: stellar-stream-492412-p9