#!/usr/bin/env bash
# =============================================================================
# run_commands.sh  —  Reference collection of run commands for all systematic
#                     test variations implemented in this repo.
#
# Environment setup (run once before any command below):
#   source ~/.desi_bashrc
#   cd /global/homes/r/rmfeder/desi_sys
#
# Invocation template:
#   OMP_NUM_THREADS=4 PYTHONPATH=. $PYBIN <script> [args]
#
# Login-node budget: nmesh=256, n_sample=20M, target_nbar=1e-4, nmock=3-5
#
# PLOTTING OPTIONS (all runs):
# Control the y-axis scale and limits on pkmu_ratio and pkmu_ratio_per_mu plots:
#   --plot-yscale {log|linear}    Scale: 'log' (default) or 'linear'
#   --plot-ylim-min YMIN           Fixed lower y-limit (default: None = adaptive)
#   --plot-ylim-max YMAX           Fixed upper y-limit (default: None = adaptive)
#
# Examples:
#   # Keep default adaptive log scale:
#   $RUN --run-name test ... (no extra flags)
#
#   # Fixed linear scale from 0.9 to 1.1 (for small effects):
#   $RUN --run-name test ... --plot-yscale linear --plot-ylim-min 0.9 --plot-ylim-max 1.1
#
#   # Fixed log scale from 0.5 to 5 (typical):
#   $RUN --run-name test ... --plot-yscale log --plot-ylim-min 0.5 --plot-ylim-max 5.0
#
#   # Adaptive scale but force log:
#   $RUN --run-name test ... --plot-yscale log
# =============================================================================

# Shorthand used throughout
RUN="OMP_NUM_THREADS=4 $PYBIN run_transverse_sys_test.py"
HALFDOME_COMMON="--mock-type halfdome --n-sample 20000000 --target-nbar 1e-4 \
  --nmesh 256 --delta-k 0.005 --k-max 0.15 --nmock 3"
QUIJOTE_COMMON="--mock-type quijote --ds-fac 5 \
  --nmesh 256 --delta-k 0.005 --k-max 0.15 --nmock 3"


# =============================================================================
# 1.  PERIODIC BOX (Quijote)  —  null control for lightcone-specific effects
# =============================================================================

# 1a. Generic power-law GRF contaminant, additive + multiplicative
$RUN --run-name quijote_powlaw $QUIJOTE_COMMON \
  --spec-type power_law --sys-amp 0.05

# 1b. Single-scale (delta-function in ell) GRF — narrow-band template
$RUN --run-name quijote_delta_ell16 $QUIJOTE_COMMON \
  --spec-type delta --ell-delta 16 --sys-amp 0.05

# 1c. Flat-spectrum GRF — broad / white-noise-like template
$RUN --run-name quijote_flat $QUIJOTE_COMMON \
  --spec-type flat --sys-amp 0.05

# 1d. Actually demonstrated Power-law GRF with sys_amp=0.1 (stronger contamination for clearer diagnostic plots)
source ~/.desi_bashrc && $PYBIN run_transverse_sys_test.py   --run-name power_law_0p1_diagxyz_v2   --spec-type power_law   --sys-amp 0.1   --nmock 10 --clear


# =============================================================================
# 2.  LIGHTCONE (Halfdome)  —  main test-bed for angular-radial coupling
# =============================================================================

# 2a. Generic power-law GRF contaminant (baseline lightcone run)
$RUN --run-name halfdome_powlaw $HALFDOME_COMMON \
  --spec-type power_law --sys-amp 0.05

$PYBIN run_transverse_sys_test.py \
      --mock-type halfdome --n-sample 20000000 --target-nbar 1e-4 \
        --nmesh 128 --delta-k 0.005 --k-max 0.15 --nmock 1 \
          --run-name power_law_halfdome   --spec-type power_law   --sys-amp 0.1   --nmock 1


OMP_NUM_THREADS=8 $PYBIN run_transverse_sys_test.py  \
     --mock-type halfdome --n-sample 20000000 --target-nbar 1e-4   \
           --nmesh 512 --delta-k 0.005 --k-min 0.002 --k-max 0.2 --run-name power_law_halfdome_v1_lowz_evenl_firstpoint \
             --spec-type power_law   --sys-amp 0.02   --nmock 1 --plot-ylim-min 0.1 \
              --plot-ylim-max 1e2 --plot-ylim-ps-min 0 --plot-ylim-ps-max 1300 --plot-only



OMP_NUM_THREADS=4 $PYBIN run_transverse_sys_test.py  \
     --mock-type halfdome --n-sample 40000000 --target-nbar 3e-4   \
           --z-min 0.1 --z-max 0.4 --nmesh 256 --delta-k 0.005 --k-min 0.002 --k-max 0.15 --run-name power_law_halfdome_v1_lowz_snsub \
             --spec-type power_law   --sys-amp 0.02   --nmock 2 --plot-yscale log --plot-ylim-min 0.1 \
              --plot-ylim-max 1e2 --plot-ylim-ps-min 0 --plot-ylim-ps-max 10000 --plot-yscale-ps log


$PYBIN run_transverse_sys_test.py       --mock-type halfdome --n-sample 20000000 --target-nbar 1e-4              --nmesh 256 --delta-k 0.005 --k-min 0.002 --k-max 0.15 --run-name power_law_halfdome_v2_lowz_evenl_firstpoint_256              --spec-type power_law   --sys-amp 0.02   --nmock 2 
--plot-ylim-min 0.1               --plot-ylim-max 1e2 --plot-ylim-ps-min 0 --plot-ylim-ps-max 1300 --z-min 0.1 --z-max 0.4



# use JAX version

$PYBIN run_transverse_sys_test.py  \
     --mock-type halfdome --n-sample 40000000 --target-nbar 3e-4   \
           --z-min 0.1 --z-max 0.4 --nmesh 256 --delta-k 0.005 --k-min 0.002 --k-max 0.15 --run-name power_law_halfdome_v1_lowz_jax \
             --spec-type power_law   --sys-amp 0.02   --nmock 1 --plot-yscale log --plot-ylim-min 0.1 \
              --plot-ylim-max 1e2 --plot-ylim-ps-min 0 --plot-ylim-ps-max 10000 --plot-yscale-ps log --use-jax


# 2b. Same but with RSD
$RUN --run-name halfdome_powlaw_rsd $HALFDOME_COMMON \
  --spec-type power_law --sys-amp 0.05 --with-rsd

# 2c. Multiplicative-only, then apply w_sys to randoms (Phase C9 estimator check)
#     Ratio P_contam/P_clean should → 1 everywhere when both data+randoms see w_sys.
$RUN --run-name halfdome_powlaw_C9 $HALFDOME_COMMON \
  --spec-type power_law --sys-amp 0.05 \
  --modes none transverse_multiplicative --apply-sys-to-randoms


# =============================================================================
# 3.  LIGHTCONE — Gaia stellar density template (anisotropic; traces Milky Way)
# =============================================================================
# sys_spec_type='gaia_stellar': Gaia counts normalized to RMS = sys_amp
# Both additive and multiplicative use the same angular template so their
# mu-dependence can be compared directly.

# 3a. Full sky (no galactic mask) — includes Galactic-plane non-Gaussianity
$RUN --run-name halfdome_gaia $HALFDOME_COMMON \
  --spec-type gaia_stellar --sys-amp 0.05

# 3b. With |b| > 20 deg galactic latitude cut — cleaner, less non-Gaussianity
$RUN --run-name halfdome_gaia_bcut20 $HALFDOME_COMMON \
  --spec-type gaia_stellar --sys-amp 0.05 --gal-lat-cut 20

# 3c. Multiplicative + apply-sys-to-randoms (Phase C9 estimator check with Gaia map)
$RUN --run-name halfdome_gaia_bcut20_C9 $HALFDOME_COMMON \
  --spec-type gaia_stellar --sys-amp 0.05 --gal-lat-cut 20 \
  --modes none transverse_multiplicative --apply-sys-to-randoms

# 3d. Two-amplitude Gaia additive (reproduces legacy compute_pkmu_mocks; frac^2 scaling test)
#     Uses the dedicated dev script which runs frac=0.05 and frac=0.01 together.
PYTHONPATH=. $PYBIN dev/test_stellar_gaia_additive.py


# =============================================================================
# 4.  LIGHTCONE — Dust extinction error (SFD E(B-V) fluctuations)
# =============================================================================
# contamination_mode='dust': multiplicative weight via delta_n/n = alpha * delta_EBV
# Uses SFD dust maps in sfd/ directory.

# 4a. Standard sfd_std=0.01, dust_alpha=-10 (default LRG-like sensitivity)
$RUN --run-name halfdome_dust $HALFDOME_COMMON \
  --modes none transverse_multiplicative \
  --spec-type power_law --sys-amp 0.01
# NOTE: dust mode is not yet wired into --modes for run_transverse_sys_test.py.
#       Run directly via Python:
PYTHONPATH=. $PYBIN - << 'EOF'
import pipeline as pl, numpy as np
from desi_mocks import desi_mock
dm = desi_mock()
spec_clean = pl.ExperimentSpec(mock_type='halfdome', contamination_mode='none',
    redshift_sel=True, zmin=0.4, zmax=1.0, n_sample=20_000_000, target_nbar=1e-4,
    nmesh=256, delta_k=0.005, k_max=0.15, los='z',
    mu_binning_strategy='nonuniform', n_clean_bins=8,
    save_dir='data/plk/transverse_sys_test/halfdome_dust')
spec_dust  = pl.ExperimentSpec(**{**spec_clean.__dict__,
    'contamination_mode': 'dust', 'sfd_std': 0.01, 'dust_alpha': -10.0})
for spec in [spec_clean, spec_dust]:
    pl.run_experiment_grid(spec, nmock=3, dm=dm)
EOF


# =============================================================================
# 5.  DIAGNOSTIC MAPS — visualize angular template in narrow z-slices
# =============================================================================

# 5a. Power-law GRF template
PYTHONPATH=. $PYBIN diagnose_transverse_sys.py --spec-type power_law

# 5b. Gaia stellar template (full sky)
PYTHONPATH=. $PYBIN diagnose_transverse_sys.py --spec-type gaia_stellar

# 5c. Gaia stellar template with |b| > 20 deg cut
PYTHONPATH=. $PYBIN diagnose_transverse_sys.py --spec-type gaia_stellar --gal-lat-cut 20

# 5d. Custom amplitude
PYTHONPATH=. $PYBIN diagnose_transverse_sys.py --spec-type gaia_stellar --sys-amp 0.10 --gal-lat-cut 20


# =============================================================================
# 6.  AMPLITUDE SWEEP — scan sys_amp to check linearity / frac^2 scaling
# =============================================================================

for AMP in 0.01 0.05 0.10; do
  $RUN --run-name "halfdome_powlaw_amp${AMP/./p}" $HALFDOME_COMMON \
    --spec-type power_law --sys-amp $AMP
done


# =============================================================================
# 7.  ESTIMATOR CROSS-CHECKS
# =============================================================================

# 7a. Quijote control with same Gaia map (confirms flat-mu is GRF isotropy, not bug)
#     (Run via dev script; Quijote doesn't use RA/Dec maps in the same way,
#     so this is mainly a sanity check that contamination map caching works.)

# 7b. Multiplicative + apply-sys-to-randoms on Gaia template
PYTHONPATH=. $PYBIN dev/test_gaia_template_add_vs_mult.py --gal-lat-cut 20 --rerun
# (set --rerun if you already have the no-cut cache and want fresh results)

# 7c. Legacy (one-sided) vs. mean-conserving additive — shows +5% n_gal bias of legacy
$RUN --run-name halfdome_legacy_additive $HALFDOME_COMMON \
  --spec-type power_law --sys-amp 0.05 \
  --modes none transverse_additive --legacy-additive
