# Repository Update Summary (2026-04-08)

This update prepares the repository for a public refresh focused on mock/systematics workflows and new 3D masking/completeness tools.

## Added Modules

- `halfdome.py`
  - Efficient loaders for large Halfdome lightcone catalogs.
  - New completeness-based acceptance/rejection sub-sampling over 3D voxel maps.
- `mask_and_randoms.py`
  - Utilities to generate random catalogs from data footprints and redshift distributions.
  - HEALPix angular mask generation from observed data density.
- `mock_tests.py`
  - End-to-end mock testing driver for clean vs contaminated $P(k,\mu)$ and multipoles.
  - Supports Quijote and Halfdome modes, optional RSD, radial cuts, and contamination toggles.
- `nonunif_binning.py`
  - Utilities for non-uniform binning response calculations.
- `star_sim.py`
  - Gaia-based stellar-density loading plus analytic notional stellar-density models.
- `stitch_box.py`
  - Periodic-box tiling with optional randomized flips/permutations.
- `validate_3d_mask.py`
  - Validation suite for 3D masks: back-projection checks, shell-uniformity checks, and wedge/slice visual diagnostics.

## Expanded Existing Modules

- `desi_mocks.py`
  - Added dedicated base paths for DESI, Quijote, Halfdome, and DR1.
  - Added loaders:
    - `load_desi_dat`
    - `load_quijote_galpos`
    - `load_halfdome_mock`
  - Integrated imports for new random/mask/stitch/halfdome utility modules.
- `plotting_fns.py`
  - Added visualization helper `plot_hexbin_density`.
  - Added `generate_3d_angular_mask` to map HEALPix angular masks into observer-centered 3D cubes.
  - Added `generate_3d_completeness_map` to convert continuous HEALPix maps into per-voxel completeness.
- `utils.py`
  - Added redshift-distance interpolation/inversion helpers.
  - Added faster RA/Dec/distance conversion utility and effective-volume calculation helpers.
- `contamination.py`
  - Added imports and hooks for stellar simulation module usage.

## Repository Hygiene in This Update

- Added/updated `.gitignore` for common generated artifacts:
  - `.ipynb_checkpoints/`
  - `__pycache__/`
  - `*.py[cod]`
- Removed accidental empty placeholder files: `1`, `3`.
- Fixed a blocker in `mask_and_randoms.py` by adding missing `healpy` import.

## Suggested Commit Title

`Update systematics workflow: add halfdome/randoms/3D mask modules and refresh docs`

## Suggested Short Release Note

"Major repository refresh adding Halfdome loaders, random-catalog utilities, 3D angular/completeness masking, and validation/plotting pipelines for systematic-contamination studies. Documentation now reflects the expanded module architecture and workflow."
