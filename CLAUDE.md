# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository simulates systematic effects in galaxy surveys (both additive and multiplicative) and tests their impact on recovered power spectra from DESI mocks. The primary science goal is quantifying how systematics bias measurements of large-scale structure, especially at low-k where primordial non-Gaussianity (f_NL) signatures appear.

## Environment

This runs on NERSC Perlmutter. Key software is available via the DESI cosmodesiconda environment. The `inference.py` module hard-codes `sys.path.append` calls pointing to:
```
/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20240118-1.0.0/...
```

Key data directories (hard-coded in `desi_mocks.py`):
- DESI mocks: `/global/cfs/cdirs/desi/survey/catalogs/`
- Quijote mocks: `/pscratch/sd/r/rmfeder/quijote_dat/`
- Halfdome halos: `/global/cfs/cdirs/cmb/gsharing/halfdome/full_res/halos/`
- DESI DR1: `/global/cfs/cdirs/desi/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/`

`config.py` has a single `basepath = None` variable — set this before running scripts that need a base output path.

## Module Architecture

All modules use flat imports (no package structure). Scripts import neighbors directly (e.g., `from contamination import *`).

### Core modules

**`desi_mocks.py`** — Central class `desi_mock` for loading catalogs:
- `load_ezmock()` — EZmock galaxy catalogs (FITS)
- `load_desi_dat()` / `load_desi_mock()` — Real DESI DR1 data and AbacusSummit mocks
- `load_quijote_galpos()` — Quijote N-body galaxy positions (numpy)
- `load_halfdome_mock()` — Halfdome lightcone halo catalogs (HDF5)
- `contaminate_catalog()` — Merges clean and contaminant position arrays

**`contamination.py`** — Class `contam` for generating and applying systematics:
- Additive: `stellar_contam_gen()` draws stellar contaminants from Gaia density map
- Multiplicative: `generate_dust_selection_err()` creates δn/n map from SFD dust E(B-V)
- `modify_fkp_weights()` applies a HEALPix δn/n map to FKP weights

**`pscalc.py`** — Wraps `pypower.CatalogFFTPower` in `compute_plk()` for computing P(k,μ) multipoles from RA/DEC/distance catalogs.

**`inference.py`** — PNG/cosmological inference utilities using `desilike`:
- `desi_mock_cov()` — Gaussian covariance and mock power spectra for f_NL inference
- `pk_ratio_fnl()` — Theoretical P(k) ratio for local PNG
- Uses `MinuitProfiler` for likelihood maximization

**`dust.py`** — `gen_sfd_hp()` queries the SFD dust map via `dustmaps` and returns a HEALPix E(B-V) map with its angular power spectrum.

**`star_sim.py`** — Stellar density map generation:
- `load_gaia_stellar_density()` — Loads pre-computed Gaia map from `stars/stellar_density_map_12_lt_g_lt_17.npy`
- `simple_halo_thick_disk_stellar_density()` / `notional_radec_stellar_density()` — Analytic stellar density models for cases without the Gaia map

**`mask_and_randoms.py`** — Random catalog generation and HEALPix mask creation from data footprint.

**`halfdome.py`** — HDF5 lightcone loading utilities for Halfdome halo catalogs.

**`stitch_box.py`** — Replicates a periodic simulation box into a larger volume (`stitch_boxes_randomized()` supports random flips/rotations to reduce periodicity artifacts).

**`utils.py`** — Coordinate utilities (Cartesian → RA/DEC/comoving distance), FKP/V_eff calculations, `init_test_params()` for default run parameters.

**`plotting_fns.py`** — Visualization: P(k,μ) wedge plots, power spectrum ratio comparisons, HEALPix sky maps, 3D angular mask generation.

### Typical workflow

1. Load mock or data catalog via `desi_mock`
2. Generate contamination via `contam` class (stellar additive or dust multiplicative)
3. Build randoms with `mask_and_randoms.py`
4. Compute P(k,μ) with `pscalc.compute_plk()`
5. Compare clean vs. contaminated spectra with `plotting_fns.compare_pkmu_wsys()`

### Key conventions

- Positions are passed to `pypower` as `(RA, DEC, comoving_distance)` arrays with `position_type='rdd'`
- Comoving distances: Planck18 cosmology throughout (`astropy.cosmology.Planck18`)
- Units: Mpc/h internally; distances from `chi_interp` are in Mpc, multiplied by `cosmo.h` where Mpc/h needed
- HEALPix maps default to `nside=256`, RING ordering
- Galaxy types: `ELG`, `LRG`, `QSO` with standard DESI redshift ranges
