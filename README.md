# clustering_sys_3D

Tools for simulating survey systematics in DESI-like large-scale structure analyses and measuring their impact on recovered power spectra.

The main science goal is to quantify biases from additive and multiplicative systematics, especially on large scales where PNG-like signals can appear.

## What This Repository Does

- Loads mock and data catalogs (DESI, Quijote, Halfdome).
- Builds additive (stellar) and multiplicative (dust/selection) contamination realizations.
- Applies angular and radial masking in both 2D and 3D representations.
- Generates matched random catalogs and computes FKP-like weighting utilities.
- Computes and compares $P(k,\mu)$ and multipoles with/without injected systematics.
- Provides plotting and validation tools for sky maps, wedges, and completeness volumes.

## Core Modules

- `desi_mocks.py`: Main catalog loader class (`desi_mock`) with support for DESI DR1, DESI mocks, Quijote, and Halfdome.
- `contamination.py`: Contamination generation and weighting modifications for stellar and dust-like effects.
- `pscalc.py`: Power spectrum measurements via `pypower` wrappers.
- `plotting_fns.py`: Plotting utilities and 3D angular/completeness map construction.
- `utils.py`: Coordinate transforms, interpolation helpers, shot noise, and effective-volume calculations.
- `mask_and_randoms.py`: Random catalog generation and HEALPix footprint masks.
- `halfdome.py`: Efficient Halfdome lightcone loading and completeness-based sub-sampling.
- `stitch_box.py`: Periodic box replication with optional random flips/axis permutations.
- `star_sim.py`: Gaia map loading plus analytic stellar-density models.
- `validate_3d_mask.py`: Diagnostics and visualization for validating 3D masks against source 2D masks.
- `mock_tests.py`: End-to-end mock pipelines for clean vs contaminated $P(k,\mu)$ tests.

## Typical Workflow

1. Load a catalog with `desi_mock`.
2. Generate systematics with `contam` or stellar/dust helper functions.
3. Build randoms and angular masks.
4. Compute spectra with `compute_plk`.
5. Compare clean/contaminated results and validate masks.

## Notes

- Several paths are configured for NERSC DESI environments.
- Outputs such as notebooks, figures, and `.npz` data products are intentionally excluded from git.
