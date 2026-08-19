# PROJECT STATE

## Current Phase
PHASE 0 — Project Initialization (in progress)

## Completed Phases
None yet.

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
1. Run `pip install -r requirements.txt`.
2. Register a Google Earth Engine account/project (free, non-commercial).
3. Open notebooks/00_colab_setup.ipynb in Colab, run it, confirm Drive mount
   + GEE auth succeed.
4. Report back with results, then say "Start Phase 1."

## Anything to know before continuing
This project spans two environments: local Windows machine (repo/git/structure)
and Google Colab (training/GEE queries). Commands are always labeled
[LOCAL - PowerShell] or [COLAB - Notebook cell] to avoid confusion.