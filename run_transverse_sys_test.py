#!/usr/bin/env python
"""
Controlled transverse systematic test on Quijote periodic box mocks.

Runs three contamination modes — 'none', 'transverse_additive', and
'transverse_multiplicative' — using the same angular power spectrum, then
plots P(k,mu) for each mu bin and the ratio contaminated/clean.

Examples (must be run with the cosmodesi Python environment):
    PYBIN=/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250526-1.0.0/conda/bin/python

    # Power-law systematic spectrum
    $PYBIN run_transverse_sys_test.py --run-name power_law_test --spec-type power_law --sys-amp 0.1 --nmock 5

    # Delta-function spectrum at ell=10
    $PYBIN run_transverse_sys_test.py --run-name delta_ell10 --spec-type delta --ell-delta 10 --sys-amp 0.1 --nmock 5
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display on login nodes
from matplotlib import pyplot as plt
# In newer matplotlib, get_cmap is accessed via plt.get_cmap or matplotlib.colormaps
try:
    from matplotlib.cm import get_cmap
except ImportError:
    # Fallback for matplotlib >= 3.5
    from matplotlib import colormaps as cm_colormaps
    def get_cmap(name):
        return cm_colormaps[name]

# ── make sure the repo root is on the path ──────────────────────────────────
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Conditional import: only needed if using pypower (default)
# Will be imported lazily when needed
# from pypower import setup_logging

from pipeline import (
    ExperimentSpec,
    _poles_to_wedges,
    _wedges_to_poles,
    _window_matrix_diagnostics,
    _prepare_quijote_catalog,
    _build_random_catalog_periodic,
    build_kedges,
    build_mu_wedges,
    build_run_label,
    run_experiment_grid,
    save_experiment_result,
    DEFAULT_QUIJOTE_BASEDIR,
    DEFAULT_HALFDOME_BASEDIR,
)
from contamination import gen_controlled_transverse_map
from desi_mocks import desi_mock
from utils import convert_to_ra_dec_distance
from celestial_diagnostics import (
    plot_radec_distribution,
    plot_comoving_distance_distribution,
    plot_radec_distance_correlation,
    plot_radec_r_comparison,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level globals — populated by parse_args() at startup.
# ─────────────────────────────────────────────────────────────────────────────
RUN_NAME      = 'default'
NMOCK         = 5
SAVE_DIR      = 'data/plk/transverse_sys_test/default'
FIG_DIR       = 'figures/transverse_sys_test/default'
SYS_AMP       = 0.01
SYS_SPEC_TYPE = 'power_law'
SYS_ELL_MIN   = 2
SYS_ELL_MAX   = 16
SYS_ELL_DELTA = None
DS_FAC        = 20
NMESH         = 512
K_MIN         = 0.006
K_MAX         = 0.2
DELTA_K       = 0.01
Z_MIN         = 0.4        # Halfdome redshift range (lower bound)
Z_MAX         = 1.0        # Halfdome redshift range (upper bound)
ELLS          = (0, 2, 4, 6, 8, 10, 12, 14, 16)
COMPUTE_WINDOW = False
WINDOW_METHOD = 'fft'
USE_RADIAL_WINDOW = True  # Halfdome: use window-corrected mu bins; defaults True but override-able

SEED = 45

# ELLS          = tuple(range(0, 17))

# ELLS          = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
# ELLS          = (0, 2, 4, 6, 8)

N_CLEAN_BINS  = 7
NGRID_SYS     = 256
NO_CACHE      = False
CLEAR         = False
WITH_RSD      = False
MOCK_TYPE     = 'quijote'   # 'quijote' or 'halfdome'
N_SAMPLE      = 20_000_000  # galaxies to load per halfdome mock
TARGET_NBAR   = None        # target comoving nbar in (h/Mpc)^3, e.g. 1e-4; None = no nbar downsampling
PLOT_YSCALE   = 'log'  # 'log' or 'linear'
PLOT_YLIM_MIN = None   # None for adaptive, or float for fixed lower limit
PLOT_YLIM_MAX = None   # None for adaptive, or float for fixed upper limit
VERBOSE_LOGGING = False  # Enable verbose output from pypower
MODES_TO_RUN  = None   # None = run all CONTAMINATION_MODES; else list of mode names
MEAN_CONSERVING_ADDITIVE = True   # Phase H1 fix: mean-conserving additive injection
APPLY_SYS_TO_RANDOMS     = False  # Phase C9 control: apply w_sys to randoms too
GAL_LAT_CUT_DEG          = 0.0   # Galactic latitude cut (degrees); 0 = no cut
USE_JAX                  = False  # Use jax-power instead of pypower
SYS_AMP_MODE             = 'rms'  # 'rms' (default) or 'mean'; for gaia_stellar, controls interpretation of sys_amp
SYS_AMP_MULT_SCALE       = 1.0   # Scaling factor for multiplicative contamination effect (>1 = stronger)

# ─────────────────────────────────────────────────────────────────────────────
# Cached mu_wedges computed once upfront and reused throughout plotting functions
# ─────────────────────────────────────────────────────────────────────────────
COMPUTED_MU_WEDGES = None  # Will be computed in main() after parse_args()

# ─────────────────────────────────────────────────────────────────────────────

CONTAMINATION_MODES = ['none', 'transverse_additive', 'transverse_multiplicative']

# pretty labels for plots
MODE_LABELS = {
    'none':                      'No systematics',
    'transverse_additive':       'Transverse additive',
    'transverse_multiplicative': 'Transverse multiplicative',
}
MODE_COLORS = {
    'none':                      'k',
    'transverse_additive':       'C0',
    'transverse_multiplicative': 'C1',
}
MODE_LS = {
    'none':                      '-',
    'transverse_additive':       '--',
    'transverse_multiplicative': ':',
}


def get_mu_wedges() -> np.ndarray:
    """
    Get cached mu_wedges, computing them once if needed.
    
    Computes window-corrected or null-bin mu wedges based on configuration
    (mock_type, use_radial_window, etc.) and caches the result for reuse
    across all plotting functions.
    
    Returns
    -------
    np.ndarray
        Mu wedge bin edges in [0, 1].
    """
    global COMPUTED_MU_WEDGES
    if COMPUTED_MU_WEDGES is not None:
        return COMPUTED_MU_WEDGES
    
    # Compute once using build_mu_wedges from pipeline
    from pipeline import build_mu_wedges
    
    # Create a minimal spec just for mu binning computation
    spec_for_wedges = ExperimentSpec(
        mock_type=MOCK_TYPE,
        zmin=Z_MIN,
        zmax=Z_MAX,
        ells=ELLS,
        n_clean_bins=N_CLEAN_BINS,
        k_min=K_MIN,
        k_max=K_MAX,
        delta_k=DELTA_K,
        use_radial_window=USE_RADIAL_WINDOW,
    )
    COMPUTED_MU_WEDGES = build_mu_wedges(spec_for_wedges)
    return COMPUTED_MU_WEDGES


def get_adaptive_ylim_log(data_arrays, default_lower=0.6, default_upper=1e1):
    """
    Compute adaptive y-axis limits for log-scale ratio plots.
    
    If data range is small (within default limits), check if data is tightly
    clustered. If so, zoom in to reveal structure. Otherwise use defaults.
    If data exceeds defaults, expand to encompass with padding.
    """
    # Flatten all data and remove NaNs/infs
    flat = np.concatenate([np.asarray(d).flatten() for d in data_arrays])
    finite = flat[(np.isfinite(flat)) & (flat > 0)]
    
    if len(finite) == 0:
        return (default_lower, default_upper)
    
    data_min = np.min(finite)
    data_max = np.max(finite)
    
    # If data exceeds defaults, expand limits with 20% padding on log scale
    if data_min < default_lower or data_max > default_upper:
        log_pad = 0.2
        new_lower = 10 ** (np.log10(data_min) - log_pad)
        new_upper = 10 ** (np.log10(data_max) + log_pad)
        new_lower = max(new_lower, 0.1)
        return (new_lower, new_upper)
    
    # Data fits within defaults, but check if it's tightly clustered
    # If range < 20% relative spread, zoom in to show structure
    data_range_log = np.log10(data_max) - np.log10(data_min)
    if data_range_log < 0.5:  # log10(10^0.5) ~ 0.5, so ~3x spread
        # Zoom in with 15% padding on log scale
        log_pad = 0.15
        new_lower = 10 ** (np.log10(data_min) - log_pad)
        new_upper = 10 ** (np.log10(data_max) + log_pad)
        new_lower = max(new_lower, 0.1)
        return (new_lower, new_upper)
    
    # Data fits within defaults and is well-spread, use defaults
    return (default_lower, default_upper)


def build_spec(contamination_mode: str) -> ExperimentSpec:
    if MOCK_TYPE == 'halfdome':
        return ExperimentSpec(
            mock_type='halfdome',
            with_rsd=WITH_RSD,
            contamination_mode=contamination_mode,
            redshift_sel=True,
            zmin=Z_MIN,
            zmax=Z_MAX,
            k_min=K_MIN,
            k_max=K_MAX,
            delta_k=DELTA_K,
            n_sample=N_SAMPLE,
            target_nbar=TARGET_NBAR,
            nmesh=NMESH,
            ells=ELLS,
            n_clean_bins=N_CLEAN_BINS,
            mu_binning_strategy='nonuniform',
            los='firstpoint',
            sys_amp=SYS_AMP,
            sys_spec_type=SYS_SPEC_TYPE,
            sys_ell_min=SYS_ELL_MIN if SYS_SPEC_TYPE != 'gaia_stellar' else 0,
            sys_ell_max=SYS_ELL_MAX if SYS_SPEC_TYPE != 'gaia_stellar' else 0,
            sys_ell_delta=SYS_ELL_DELTA,
            sys_amp_mode=SYS_AMP_MODE,
            sys_amp_mult_scale=SYS_AMP_MULT_SCALE,
            mean_conserving_additive=MEAN_CONSERVING_ADDITIVE,
            apply_sys_to_randoms=APPLY_SYS_TO_RANDOMS,
            gal_lat_cut_deg=GAL_LAT_CUT_DEG,
            use_jax=USE_JAX,
            use_radial_window=USE_RADIAL_WINDOW,
            save_dir=SAVE_DIR,
            compute_window=COMPUTE_WINDOW,
            window_method='fft'   # or 'smooth'
        )
    # Quijote periodic box
    return ExperimentSpec(
        mock_type='quijote',
        quijote_geometry='full_cube',
        with_rsd=WITH_RSD,
        contamination_mode=contamination_mode,
        redshift_sel=False,       # periodic box — no true redshifts
        ds_fac=DS_FAC,
        nmesh=NMESH,
        ells=ELLS,
        n_clean_bins=N_CLEAN_BINS,
        mu_binning_strategy='nonuniform',
        sys_amp=SYS_AMP,
        sys_spec_type=SYS_SPEC_TYPE,
        sys_ell_min=SYS_ELL_MIN if SYS_SPEC_TYPE != 'gaia_stellar' else 0,
        sys_ell_max=SYS_ELL_MAX if SYS_SPEC_TYPE != 'gaia_stellar' else 0,
        sys_ell_delta=SYS_ELL_DELTA,
        sys_amp_mode=SYS_AMP_MODE,
        sys_amp_mult_scale=SYS_AMP_MULT_SCALE,
        mean_conserving_additive=MEAN_CONSERVING_ADDITIVE,
        apply_sys_to_randoms=APPLY_SYS_TO_RANDOMS,
        use_jax=USE_JAX,
        use_radial_window=False,  # Quijote periodic box: no radial window
        save_dir=SAVE_DIR,
    )


def compute_effective_sys_amp() -> float:
    """
    Compute the effective RMS amplitude of the systematic after masking.
    
    When GAL_LAT_CUT_DEG > 0, masked regions have zero contamination,
    which lowers the effective amplitude in the unmasked region.
    
    Returns the RMS amplitude of the systematic in the unmasked region,
    which should be used for the dilution formula instead of SYS_AMP.
    """
    if MOCK_TYPE != 'quijote':
        # For Quijote periodic box, we use the transverse map directly
        # No masking is applied in the periodic box; GAL_LAT_CUT_DEG only affects
        # the lightcone geometry (halfdome), not periodic boxes.
        return SYS_AMP
    
    # For periodic box (Quijote), GAL_LAT_CUT_DEG is not typically applied,
    # so we just use SYS_AMP. But if it were applied, compute from map:
    if GAL_LAT_CUT_DEG <= 0.0:
        return SYS_AMP
    
    # Load the systematic map to compute effective amplitude after masking
    sys_map = _load_sys_map_healpix()
    
    # The unmasked region should have been preserved, masked region zeroed
    # Compute RMS of the map (which now reflects effective contamination)
    effective_amp = np.std(sys_map)
    
    return effective_amp


def plot_pkmu_comparison(results: dict[str, dict], fig_dir: str) -> None:
    """
    Two-panel figure per mu bin:
      Left:  kP(k,mu) for each contamination mode
      Right: ratio contaminated / clean
    """
    os.makedirs(fig_dir, exist_ok=True)

    clean_res = results['none']
    kcen = clean_res['kcen'].real
    mu_wedges = clean_res['mu_wedges']
    nmu = len(mu_wedges) - 1

    cmap = get_cmap('jet')
    mu_colors = [cmap(v) for v in np.linspace(0.15, 0.9, nmu)]
    
    # Compute effective systematic amplitude accounting for masking
    effective_sys_amp = compute_effective_sys_amp()
    if GAL_LAT_CUT_DEG > 0.0 and effective_sys_amp != SYS_AMP:
        print(f'  Masking effect: SYS_AMP={SYS_AMP:.6f} → effective={effective_sys_amp:.6f}')
    else:
        print(f'  Effective contamination amplitude: {effective_sys_amp:.6f}')

    # ── Figure 1: kP(k,mu) per mode ─────────────────────────────────────────
    fig, axes = plt.subplots(1, len(CONTAMINATION_MODES), figsize=(4 * len(CONTAMINATION_MODES), 4),
                             sharey=True)
    for ax, mode in zip(axes, CONTAMINATION_MODES):
        res = results[mode]
        pkmu = res['all_pkmu'].real  # shape (nmock, nk, nmu)
        ax.set_title(MODE_LABELS[mode], fontsize=12)
        for mu_idx in range(nmu):
            mu_lab = (f'{mu_wedges[mu_idx]:.2f}' + r'$<\mu<$' +
                      f'{mu_wedges[mu_idx+1]:.2f}')
            mean_pk = np.nanmean(pkmu[:, :, mu_idx], axis=0)
            err_pk  = np.nanstd(pkmu[:, :, mu_idx], axis=0) / np.sqrt(np.isfinite(pkmu[:, :, mu_idx]).sum(axis=0).clip(1))
            
            # Only plot valid data points
            valid = np.isfinite(mean_pk) & np.isfinite(err_pk) & (mean_pk > 0)
            if np.any(valid):
                ax.errorbar(kcen[valid], (kcen * mean_pk)[valid], yerr=(kcen * err_pk)[valid],
                            label=mu_lab, color=mu_colors[mu_idx],
                            linewidth=2 if mu_idx == 0 else 1.2)
        ax.set_xlabel('k [h/Mpc]', fontsize=11)
        ax.grid(alpha=0.2)
        ax.set_xscale('log')

        # Compute y-limits
        # if PLOT_YLIM_PS_MIN is not None and PLOT_YLIM_PS_MAX is not None:
            # ylim = (PLOT_YLIM_PS_MIN, PLOT_YLIM_PS_MAX)
        # if PLOT_YSCALE_PS == 'log':
        #     ax.set_yscale('log')
        ax.set_ylim(0, 2000)
        ax.set_xlim(1e-2, 0.3)
    axes[0].set_ylabel(r'$k\,P(k,\mu)$ [$({\rm Mpc}/h)^2$]', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=5, fontsize=10, frameon=True, bbox_transform=fig.transFigure)
    plt.subplots_adjust(wspace=0)
    fpath = os.path.join(fig_dir, 'pkmu_comparison.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')

    # ── Figure 2: ratio P_sys / P_clean per mu bin, one panel per mode ──────
    sys_modes = [m for m in CONTAMINATION_MODES if m != 'none']
    fig, axes = plt.subplots(1, len(sys_modes), figsize=(4 * len(sys_modes), 3.5),
                             sharey=True)
    if len(sys_modes) == 1:
        axes = [axes]

    pkmu_clean = clean_res['all_pkmu'].real
    
    # Collect all ratio data to compute adaptive limits
    all_ratio_data = []
    for ax, mode in zip(axes, sys_modes):
        res = results[mode]
        pkmu_sys = res['all_pkmu'].real
        ax.set_title(f'{MODE_LABELS[mode]}\nvs. clean', fontsize=12)
        ax.axhline(1.0, color='grey', lw=1.5, ls='-', zorder=0)
        ax.axhline((1.0 / (1.0 + effective_sys_amp)) ** 2, color='k', lw=1.2, ls='--', zorder=0, label='$(1+f_{*})^{-2}$ dilution')
        for mu_idx in range(nmu):
            mu_lab = (f'{mu_wedges[mu_idx]:.2f}' + r'$<\mu<$' +
                      f'{mu_wedges[mu_idx+1]:.2f}')
            ratio = pkmu_sys[:, :, mu_idx] / np.where(
                np.abs(pkmu_clean[:, :, mu_idx]) > 0,
                pkmu_clean[:, :, mu_idx],
                np.nan,
            )
            all_ratio_data.append(ratio)
            mean_r = np.nanmean(ratio, axis=0)
            err_r  = np.nanstd(ratio, axis=0) / np.sqrt(ratio.shape[0])
            ax.errorbar(kcen, mean_r, yerr=err_r,
                        label=mu_lab, color=mu_colors[mu_idx],
                        linewidth=2.5 if mu_idx == 0 else 2,
                        linestyle=MODE_LS[mode])
        ax.set_xlabel('k [h/Mpc]', fontsize=11)
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        ax.set_yscale(PLOT_YSCALE)
    
    # Compute y-limits
    if PLOT_YLIM_MIN is not None and PLOT_YLIM_MAX is not None:
        ylim = (PLOT_YLIM_MIN, PLOT_YLIM_MAX)
    elif PLOT_YSCALE == 'log':
        ylim = get_adaptive_ylim_log(all_ratio_data)
    else:
        # For linear scale, use data range with padding
        flat = np.concatenate([np.asarray(d).flatten() for d in all_ratio_data])
        finite = flat[np.isfinite(flat)]
        if len(finite) > 0:
            dmin, dmax = np.min(finite), np.max(finite)
            padding = (dmax - dmin) * 0.15
            ylim = (dmin - padding, dmax + padding)
        else:
            ylim = (0.9, 1.1)
    
    for ax in axes:
        ax.set_ylim(*ylim)
    
    axes[0].set_ylabel(r'$P(k,\mu)^{\rm sys}\,/\,P(k,\mu)^{\rm clean}$', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),
               ncol=4, fontsize=10, frameon=True, bbox_transform=fig.transFigure)
    plt.subplots_adjust(wspace=0)
    fpath = os.path.join(fig_dir, 'pkmu_ratio.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')

    # ── Figure 3: ratio overlay, all modes in one panel per mu bin ──────────
    ncols = min(nmu, 4)
    nrows = (nmu + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten()
    for mu_idx in range(nmu):
        ax = axes_flat[mu_idx]
        mu_lab = (f'{mu_wedges[mu_idx]:.2f}' + r'$<\mu<$' +
                  f'{mu_wedges[mu_idx+1]:.2f}')
        ax.set_title(mu_lab, fontsize=10)
        ax.axhline(1.0, color='grey', lw=1.5, ls='-', zorder=0)
        ax.axhline((1.0 / (1.0 + SYS_AMP)) ** 2, color='k', lw=1.2, ls='--', zorder=0)
        mu_ratio_data = []
        for mode in sys_modes:
            pkmu_sys = results[mode]['all_pkmu'].real
            ratio = pkmu_sys[:, :, mu_idx] / np.where(
                np.abs(pkmu_clean[:, :, mu_idx]) > 0,
                pkmu_clean[:, :, mu_idx],
                np.nan,
            )
            mu_ratio_data.append(ratio)
            mean_r = np.nanmean(ratio, axis=0)
            err_r  = np.nanstd(ratio, axis=0) / np.sqrt(ratio.shape[0])
            ax.errorbar(kcen, mean_r, yerr=err_r,
                        label=MODE_LABELS[mode], color=MODE_COLORS[mode],
                        linewidth=1.8, linestyle=MODE_LS[mode])
        ax.set_xscale('log')
        ax.set_yscale(PLOT_YSCALE)
        
        # Compute y-limits for this mu bin
        if PLOT_YLIM_MIN is not None and PLOT_YLIM_MAX is not None:
            ylim = (PLOT_YLIM_MIN, PLOT_YLIM_MAX)
        elif PLOT_YSCALE == 'log':
            ylim = get_adaptive_ylim_log(mu_ratio_data)
        else:
            # For linear scale, use data range with padding
            flat = np.concatenate([np.asarray(d).flatten() for d in mu_ratio_data])
            finite = flat[np.isfinite(flat)]
            if len(finite) > 0:
                dmin, dmax = np.min(finite), np.max(finite)
                padding = (dmax - dmin) * 0.15
                ylim = (dmin - padding, dmax + padding)
            else:
                ylim = (0.9, 1.1)
        
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    for ax in axes_flat[nmu:]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=min(5, len(sys_modes)), fontsize=10, frameon=True,
               bbox_transform=fig.transFigure)
    fig.supxlabel('k [h/Mpc]', fontsize=11)
    fig.supylabel(r'$P(k,\mu)^{\rm sys}\,/\,P(k,\mu)^{\rm clean}$', fontsize=11)
    title_parts = [f'Quijote periodic box  |  {NMOCK} mocks  |  run={RUN_NAME}',
                   f'spec_type={SYS_SPEC_TYPE}, sys_amp={SYS_AMP}' +
                   (f', ell_delta={SYS_ELL_DELTA}' if SYS_ELL_DELTA is not None else
                    f', ell_min={SYS_ELL_MIN}, ell_max={SYS_ELL_MAX}')]
    fig.suptitle('\n'.join(title_parts), fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fpath = os.path.join(fig_dir, 'pkmu_ratio_per_mu.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_power_spectrum_multipoles(results: dict[str, dict], fig_dir: str) -> None:
    """
    Plot power spectrum multipoles (monopole, quadrupole, hexadecapole) 
    reconstructed from wedges via wedges-to-poles inversion.
    
    Three panels: one per contamination mode (none, additive, multiplicative).
    Each panel shows ℓ=0, 2, 4 with error bars.
    """
    from pipeline import _wedges_to_poles
    
    os.makedirs(fig_dir, exist_ok=True)
    
    clean_res = results['none']
    kcen = clean_res['kcen'].real
    mu_wedges = clean_res['mu_wedges']
    
    # Target multipoles
    # target_ells = [0, 2, 4]
    # ell_labels = {0: r'$P_0(k)$ (monopole)', 
    #               2: r'$P_2(k)$ (quadrupole)', 
    #               4: r'$P_4(k)$ (hexadecapole)'}
    # ell_colors = {0: 'C0', 2: 'C1', 4: 'C2'}

    target_ells = [0, 2, 4, 6, 8, 10, 12, 14, 16]

    ell_labels = {0: r'$P_0(k)$ (monopole)', 
                  2: r'$P_2(k)$ (quadrupole)', 
                  4: r'$P_4(k)$ (hexadecapole)', 
                  6: r'$P_6(k)$', 
                  8: r'$P_8(k)$',
                  10: r'$P_{10}(k)$',
                  12: r'$P_{12}(k)$',
                  14: r'$P_{14}(k)$',
                  16: r'$P_{16}(k)$'}
    ell_colors = {0: 'C0', 2: 'C1', 4: 'C2', 6: 'C3', 8: 'C4', 10: 'C5', 12: 'C6', 14: 'C7', 16: 'C8'}
    
    # target_ells = [1, 3, 5, 7, 9, 11, 13, 15]
    target_ells = [0, 2, 4, 6, 8, 10, 12, 14, 16]

    # ell_labels = {1: r'$P_1(k)$', 3: r'$P_3(k)$', 5: r'$P_5(k)$', 7: r'$P_7(k)$',
                  # 9: r'$P_9(k)$', 11: r'$P_{11}(k)$', 13: r'$P_{13}(k)$', 15: r'$P_{15}(k)$'}
    # ell_colors = {1: 'C0', 3: 'C1', 5: 'C2', 7: 'C3', 9: 'C4', 11: 'C5', 13: 'C6', 15: 'C7'}

    # Build figure with one panel per contamination mode
    fig, axes = plt.subplots(1, len(CONTAMINATION_MODES), figsize=(4 * len(CONTAMINATION_MODES), 4.5),
                             sharey=True)
    if len(CONTAMINATION_MODES) == 1:
        axes = [axes]
    
    for ax, mode in zip(axes, CONTAMINATION_MODES):
        res = results[mode]
        pkmu_all = res['all_pkmu'].real  # shape (nmock, nk, nmu)
        
        ax.set_title(MODE_LABELS[mode], fontsize=12)
        
        # For each mock, reconstruct multipoles from wedges
        plk_all = []
        for mock_idx in range(pkmu_all.shape[0]):
            pkmu = pkmu_all[mock_idx]  # shape (nk, nmu)
            plk = _wedges_to_poles(ELLS, pkmu, mu_wedges)  # shape (nk, nell)
            plk_all.append(plk.real)
        plk_all = np.array(plk_all)  # shape (nmock, nk, nell)
        
        # Plot each target multipole with robustness to NaN values
        valid_data_ranges = []
        for ell in target_ells:
            if ell in ELLS:
                ell_idx = list(ELLS).index(ell)
                mean_ell = np.nanmean(plk_all[:, :, ell_idx], axis=0)
                err_ell = np.nanstd(plk_all[:, :, ell_idx], axis=0) / np.sqrt(np.isfinite(plk_all[:, :, ell_idx]).sum(axis=0).clip(1))
                
                # Only plot valid data points
                valid = np.isfinite(mean_ell)
                if np.any(valid):
                    ax.errorbar(kcen[valid], mean_ell[valid], yerr=err_ell[valid],
                               label=ell_labels[ell], color=ell_colors[ell],
                               linewidth=2, marker='o', markersize=4, capsize=3)
                    valid_data_ranges.append(mean_ell[valid])
        
        ax.set_xlabel('k [h/Mpc]', fontsize=11)

        if mode == 'none':
            ax.set_ylabel(r'$P_\ell(k)$ [$({\rm Mpc}/h)^3$]', fontsize=11)
        ax.set_xscale('log')
        ax.set_yscale('log')
        

        ax.set_ylim(1e3, 2e5)

        # # Set y-limits adaptively based on valid data
        # if valid_data_ranges:
        #     all_valid = np.concatenate(valid_data_ranges)
        #     if len(all_valid) > 0:
        #         ymin, ymax = np.min(all_valid), np.max(all_valid)
        #         ymin_padded = ymin / 3
        #         ymax_padded = ymax * 3
        #         ax.set_ylim(ymin_padded, ymax_padded)
        #     else:
        #         ax.set_ylim(1e2, 1e5)
        # else:
        #     ax.set_ylim(1e2, 1e5)
        
        ax.grid(alpha=0.3, which='both')
        if mode == 'none':
            ax.legend(fontsize=12, ncol=4, bbox_to_anchor=(1.5, 1.5), loc='upper center', frameon=True)
    
    fig.suptitle(f'Power Spectrum Multipoles | {NMOCK} mocks | run={RUN_NAME}\n'
                 f'spec_type={SYS_SPEC_TYPE}, sys_amp={SYS_AMP}',
                 fontsize=11, y=1.02)
    # plt.tight_layout()
    fpath = os.path.join(fig_dir, 'pkell_multipoles.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def _load_sys_map_healpix(seed: int | None = None) -> np.ndarray:
    """Return the HEALPix contamination map matching the current globals.

    Handles both GRF-based (power_law / flat / delta) and gaia_stellar maps,
    including the galactic latitude cut, so all plotting functions stay in sync
    with the pipeline.
    
    For gaia_stellar:
    - Preserves inherent Gaia coverage mask (where counts are zero)
    - If SYS_AMP_MODE == 'rms': scales map to have RMS = sys_amp (original behavior)
    - If SYS_AMP_MODE == 'mean': scales map to have mean = sys_amp (more intuitive for contamination fraction)
    """
    import healpy as hp
    if SYS_SPEC_TYPE == 'gaia_stellar':
        from star_sim import load_gaia_stellar_density
        from pipeline import _gal_lat_b
        stellar_map = load_gaia_stellar_density(plot=False)
        m = stellar_map.astype(float)
        
        # Preserve the inherent Gaia coverage mask (where Gaia has zero counts).
        # These regions should remain zero in the final contamination map.
        gaia_coverage_mask = (stellar_map == 0)
        
        # Create combined mask: Gaia no-coverage + galactic latitude cut
        mask_pixels = gaia_coverage_mask.copy()
        if GAL_LAT_CUT_DEG > 0.0:
            nside_m = hp.npix2nside(len(m))
            pix_theta, pix_phi = hp.pix2ang(nside_m, np.arange(len(m)))
            pix_b = _gal_lat_b(np.degrees(pix_phi), 90.0 - np.degrees(pix_theta))
            plane_mask = np.abs(pix_b) < GAL_LAT_CUT_DEG
            mask_pixels |= plane_mask  # Combine with OR
        
        # Apply combined mask
        m[mask_pixels] = 0.0
        
        # Get unmasked region for normalization
        unmasked = m[~mask_pixels]
        
        # Scale map according to sys_amp_mode
        if SYS_AMP_MODE == 'mean':
            # Scale to target mean contamination level (interpretation: fraction of added stars)
            current_mean = unmasked.mean()
            if np.abs(current_mean) > 1e-10:
                sys_map = (SYS_AMP / current_mean) * m
            else:
                sys_map = m
        else:  # 'rms' (default)
            # Scale to target RMS (original behavior), centering only on unmasked region
            m_centered = m - unmasked.mean()
            std = m_centered[~mask_pixels].std()
            sys_map = (SYS_AMP / std) * m_centered if std > 0 else m_centered
        
        # Ensure masked regions stay at zero
        sys_map[mask_pixels] = 0.0
        return sys_map

    if SEED is not None:
        seed = SEED
    if seed is None:
        seed = 42 + 0 * 10_000 + 2  # mock_idx=0, same as pipeline seed_dust

    print('seed before gen_controlled_transverse_map is ', seed)
    return gen_controlled_transverse_map(
        amp=SYS_AMP,
        seed=seed,
        spec_type=SYS_SPEC_TYPE,
        ell_max=SYS_ELL_MAX,
        ell_min=SYS_ELL_MIN,
        ell_delta=SYS_ELL_DELTA,
        periodic=False,
        nside=256,
    )

def plot_null_mu_bin_effect(results: dict[str, dict], fig_dir: str) -> None:
    """
    Plot the impact of nulling the lowest μ bin on recovered P_ℓ(k).

    For each contaminated mode and ℓ ∈ {0, 2, 4}, shows:
    - Top row (taller): clean, contaminated (regular), and contaminated with lowest μ nulled
    - Bottom row (shorter): fractional deviation (nulled / clean), fixed y-range [1e-1, 1e1]
    """
    if 'none' not in results or results['none'].get('all_plk_null_lowest_mu') is None:
        print('Skipping null μ-bin effect plot: clean mode missing or no nulled multipoles.')
        return

    plk_clean = results['none']['all_plk']  # (nmock, nell, nk)
    plk_clean_avg = np.mean(plk_clean, axis=0)  # (nell, nk)
    kcen = results['none']['kcen']
    ells = results['none']['ells']


    os.makedirs(fig_dir, exist_ok=True)

    contam_modes = [
        mode for mode in results
        if mode != 'none' and results[mode].get('all_plk_null_lowest_mu') is not None
    ]
    if not contam_modes:
        print('Skipping null μ-bin effect plot: no contaminated modes with nulled multipoles.')
        return

    target_ells = [ell for ell in ells if ell in [0, 2, 4]]
    target_ell_indices = [list(ells).index(ell) for ell in target_ells]

    n_ells = len(target_ells)
    n_modes = len(contam_modes)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(5 * n_ells, 5.4 * n_modes))
    # For each mode: 2 rows (top taller, bottom shorter)
    gs = GridSpec(
        nrows=2 * n_modes,
        ncols=n_ells,
        figure=fig,
        height_ratios=[3, 1] * n_modes,
        hspace=0.18,
        wspace=0.25,
    )

    for mode_row_idx, mode in enumerate(contam_modes):
        plk_contam = results[mode]['all_plk']  # (nmock, nell, nk)
        plk_null = results[mode]['all_plk_null_lowest_mu']  # (nmock, nell, nk)

        plk_contam_avg = np.nanmean(plk_contam, axis=0)
        plk_null_avg = np.nanmean(plk_null, axis=0)

        top_row = 2 * mode_row_idx
        bot_row = top_row + 1

        for col_idx, (ell, ell_idx) in enumerate(zip(target_ells, target_ell_indices)):
            ax_top = fig.add_subplot(gs[top_row, col_idx])
            ax_bot = fig.add_subplot(gs[bot_row, col_idx], sharex=ax_top)

            pk_clean = np.abs(plk_clean_avg[ell_idx, :])
            pk_contam = np.abs(plk_contam_avg[ell_idx, :])
            pk_null = np.abs(plk_null_avg[ell_idx, :])

            valid_clean = np.isfinite(pk_clean) & (pk_clean > 0)
            valid_contam = np.isfinite(pk_contam) & (pk_contam > 0)
            valid_null = np.isfinite(pk_null) & (pk_null > 0)


            clean_samples  = np.abs(plk_clean[:, ell_idx, :])    # (nmock, nk)
            contam_samples = np.abs(plk_contam[:, ell_idx, :])   # (nmock, nk)
            null_samples   = np.abs(plk_null[:, ell_idx, :])     # (nmock, nk)

            # Means
            pk_clean  = np.nanmean(clean_samples, axis=0)
            pk_contam = np.nanmean(contam_samples, axis=0)
            pk_null   = np.nanmean(null_samples, axis=0)

            n_clean_eff  = np.sum(np.isfinite(clean_samples), axis=0)
            n_contam_eff = np.sum(np.isfinite(contam_samples), axis=0)
            n_null_eff   = np.sum(np.isfinite(null_samples), axis=0)

            err_clean  = np.nanstd(clean_samples, axis=0, ddof=1) / np.sqrt(np.maximum(n_clean_eff, 1))
            err_contam = np.nanstd(contam_samples, axis=0, ddof=1) / np.sqrt(np.maximum(n_contam_eff, 1))
            err_null   = np.nanstd(null_samples, axis=0, ddof=1) / np.sqrt(np.maximum(n_null_eff, 1))

            # --- Top panel ---
            # if np.any(valid_clean):
            #     ax_top.loglog(kcen[valid_clean], pk_clean[valid_clean], 'k-', linewidth=2, label='Clean', alpha=0.8)
            # if np.any(valid_contam):
            #     ax_top.loglog(kcen[valid_contam], pk_contam[valid_contam], 'C0--', linewidth=1.8, label='Contaminated', alpha=0.75)
            # if np.any(valid_null):
            #     ax_top.loglog(kcen[valid_null], pk_null[valid_null], 'C1:', linewidth=2.2, label='Contam (null lowest μ)', alpha=0.85)

            valid_clean = np.isfinite(pk_clean) & (pk_clean > 0) & np.isfinite(err_clean)
            ax_top.errorbar(kcen[valid_clean], pk_clean[valid_clean], yerr=err_clean[valid_clean],
                            fmt='k-', lw=2, elinewidth=1, capsize=2, alpha=0.8, label='Clean')

            valid_contam = np.isfinite(pk_contam) & (pk_contam > 0) & np.isfinite(err_contam)
            ax_top.errorbar(kcen[valid_contam], pk_contam[valid_contam], yerr=err_contam[valid_contam],
                            fmt='C0--', lw=1.8, elinewidth=1, capsize=2, alpha=0.75, label='Contaminated')

            valid_null = np.isfinite(pk_null) & (pk_null > 0) & np.isfinite(err_null)
            ax_top.errorbar(kcen[valid_null], pk_null[valid_null], yerr=err_null[valid_null],
                            fmt='C1:', lw=2.2, elinewidth=1, capsize=2, alpha=0.85, label='Contam (null lowest μ)')


            ax_top.set_xscale('log')

            if mode_row_idx == 0:
                ax_top.set_title(f'ℓ={ell}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax_top.text(
                    -0.32, 0.5, f'{mode}',
                    transform=ax_top.transAxes, fontsize=10, fontweight='bold',
                    rotation=90, va='center'
                )

            ax_top.set_ylabel(fr'$|P_{{{ell}}}(k)|$ [$(h/\mathrm{{Mpc}})^3$]', fontsize=10)
            ax_top.grid(alpha=0.3, which='both')
            if mode_row_idx == 0 and col_idx == 0:
                ax_top.legend(fontsize=8, loc='best')

            ax_top.set_yscale('log')

            ax_top.set_ylim(1e3, 1e6)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            # --- Bottom panel: fractional deviation (nulled/clean) ---
            ratio_null = np.where(
                np.isfinite(pk_clean) & (pk_clean > 1e-20),
                pk_null / pk_clean,
                np.nan
            )

            ratio_contam = np.where(
                np.isfinite(pk_clean) & (pk_clean > 1e-20),
                pk_contam / pk_clean,
                np.nan
            )


            valid_ratio = np.isfinite(ratio_null) & (ratio_null > 0)

            valid_ratio_contam = np.isfinite(ratio_contam) & (ratio_contam > 0)

            if np.any(valid_ratio):

                err_null_rel = err_null[valid_ratio] / pk_clean[valid_ratio]
                err_clean_rel = err_clean[valid_ratio] / pk_clean[valid_ratio]
                err_ratio = ratio_null[valid_ratio] * np.sqrt(err_null_rel**2 + err_clean_rel**2)

                err_contam_rel = err_contam[valid_ratio_contam] / pk_clean[valid_ratio_contam]
                err_ratio_contam = (pk_contam[valid_ratio_contam] / pk_clean[valid_ratio_contam]) * np.sqrt(err_contam_rel**2 + err_clean_rel**2)     

                ax_bot.errorbar(kcen[valid_ratio], ratio_null[valid_ratio], yerr=err_ratio, fmt='C1', linewidth=1.7)

                ax_bot.errorbar(kcen[valid_ratio_contam], ratio_contam[valid_ratio_contam], yerr=err_ratio_contam, fmt='C0', linewidth=1.7)

                # ax_bot.semilogx(kcen[valid_ratio], ratio_null[valid_ratio], color='C1', linewidth=1.7)

            ax_bot.axhline(1.0, color='red', linestyle='--', linewidth=1.0, alpha=0.65)
            ax_bot.set_yscale('log')
            ax_bot.set_ylim(1e-1, 1e1)  # fixed range requested
            ax_bot.grid(alpha=0.25, which='both')
            ax_bot.set_xlabel('k [h/Mpc]', fontsize=10)
            if col_idx == 0:
                ax_bot.set_ylabel('nulled / clean', fontsize=9)

    fig.suptitle('Impact of Nulling Lowest μ Bin on Recovered P_ℓ(k)', fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    fpath = os.path.join(fig_dir, 'null_mu_bin_effect.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')

    # Summary stats (unchanged)
    print("\nNull μ-bin effect summary (ℓ ∈ {0, 2, 4}):")
    print("-" * 70)
    for mode in contam_modes:
        plk_null = results[mode]['all_plk_null_lowest_mu']
        plk_null_avg = np.nanmean(plk_null, axis=0)
        plk_contam = results[mode]['all_plk']
        plk_contam_avg = np.nanmean(plk_contam, axis=0)

        print(f"\n{mode}:")
        print(f"  {'ℓ':<4} {'Mean ratio':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        for ell, ell_idx in zip(target_ells, target_ell_indices):
            ratio = np.abs(plk_null_avg[ell_idx, :]) / (np.abs(plk_clean_avg[ell_idx, :]) + 1e-20)
            ratio_valid = ratio[np.isfinite(ratio) & (ratio > 0)]
            if len(ratio_valid) > 0:
                mean_r = np.mean(ratio_valid)
                std_r = np.std(ratio_valid)
                min_r = np.min(ratio_valid)
                max_r = np.max(ratio_valid)
                n_valid = len(ratio_valid)
                n_total = len(ratio)
                print(f"  {ell:<4} {mean_r:<12.4f} {std_r:<12.4f} {min_r:<12.4f} {max_r:<12.4f} ({n_valid}/{n_total} valid)")
            else:
                print(f"  {ell:<4} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} (no valid data)")
    print("-" * 70)

# def plot_null_mu_bin_effect(results: dict[str, dict], fig_dir: str) -> None:
#     """
#     Plot the impact of nulling the lowest μ bin on recovered P_ℓ(k).
    
#     For each contaminated mode and ℓ ∈ {0, 2, 4}, shows:
#     - Main panel: clean, contaminated (regular), and contaminated with lowest μ nulled
#     - Inset: ratio of (nulled / clean)
    
#     Parameters
#     ----------
#     results : dict[str, dict]
#         Dict mapping mode -> result dict with keys: kcen, all_plk, all_plk_null_lowest_mu, ells, label
#     fig_dir : str
#         Output directory for figures
#     """
#     if 'none' not in results or results['none'].get('all_plk_null_lowest_mu') is None:
#         print('Skipping null μ-bin effect plot: clean mode missing or no nulled multipoles.')
#         return
    
#     plk_clean = results['none']['all_plk']  # (nmock, nell, nk)
#     plk_clean_avg = np.mean(plk_clean, axis=0)  # (nell, nk)
#     kcen = results['none']['kcen']
#     ells = results['none']['ells']
    
#     os.makedirs(fig_dir, exist_ok=True)
    
#     # Determine which contaminated modes have nulled multipoles
#     contam_modes = [mode for mode in results if mode != 'none' 
#                     and results[mode].get('all_plk_null_lowest_mu') is not None]
    
#     if not contam_modes:
#         print('Skipping null μ-bin effect plot: no contaminated modes with nulled multipoles.')
#         return
    
#     # Target only ℓ=0, 2, 4
#     target_ells = [ell for ell in ells if ell in [0, 2, 4]]
#     target_ell_indices = [list(ells).index(ell) for ell in target_ells]
    
#     n_ells = len(target_ells)
#     n_modes = len(contam_modes)
    
#     # Create figure with n_modes rows × n_ells columns
#     fig, axes = plt.subplots(n_modes, n_ells, figsize=(5*n_ells, 4.5*n_modes), sharey='col')
#     if n_modes == 1:
#         axes = axes.reshape(1, -1)
#     elif n_ells == 1:
#         axes = axes.reshape(-1, 1)
    
#     ell_colors = {0: 'C0', 2: 'C1', 4: 'C2'}
    
#     for mode_row_idx, mode in enumerate(contam_modes):
#         plk_contam = results[mode]['all_plk']  # (nmock, nell, nk)
#         plk_null = results[mode]['all_plk_null_lowest_mu']  # (nmock, nell, nk)
        
#         plk_contam_avg = np.nanmean(plk_contam, axis=0)  # (nell, nk)
#         plk_null_avg = np.nanmean(plk_null, axis=0)  # (nell, nk)
        
#         for col_idx, (ell, ell_idx) in enumerate(zip(target_ells, target_ell_indices)):
#             ax = axes[mode_row_idx, col_idx]
            
#             pk_clean = np.abs(plk_clean_avg[ell_idx, :])
#             pk_contam = np.abs(plk_contam_avg[ell_idx, :])
#             pk_null = np.abs(plk_null_avg[ell_idx, :])
            
#             # Filter out NaN and inf values for robustness with small nmesh
#             valid_clean = np.isfinite(pk_clean) & (pk_clean > 0)
#             valid_contam = np.isfinite(pk_contam) & (pk_contam > 0)
#             valid_null = np.isfinite(pk_null) & (pk_null > 0)
            
#             # Main panel: three curves (only plot valid data)
#             if np.any(valid_clean):
#                 ax.loglog(kcen[valid_clean], pk_clean[valid_clean], 'k-', linewidth=2, label='Clean', alpha=0.8)
#             if np.any(valid_contam):
#                 ax.loglog(kcen[valid_contam], pk_contam[valid_contam], 'C0--', linewidth=1.8, label='Contaminated', alpha=0.7)
#             if np.any(valid_null):
#                 ax.loglog(kcen[valid_null], pk_null[valid_null], 'C1:', linewidth=2.2, label='Contaminated (null lowest μ)', alpha=0.8)
            
#             ax.set_xlabel('k [h/Mpc]', fontsize=10)
#             ax.set_ylabel(f'$|P_{ell}(k)|$ [$(h/\\mathrm{{Mpc}})^3$]', fontsize=10)
            
#             if mode_row_idx == 0:
#                 ax.set_title(f'ℓ={ell}', fontsize=11, fontweight='bold')
            
#             if col_idx == 0:
#                 ax.text(-0.35, 0.5, f'{mode}', transform=ax.transAxes,
#                        fontsize=10, fontweight='bold', rotation=90, va='center')
            
#             ax.grid(alpha=0.3, which='both')
#             if mode_row_idx == 0 and col_idx == 0:
#                 ax.legend(fontsize=8, loc='best')
            
#             # Inset: ratio panel (nulled / clean)
#             from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#             axins = inset_axes(ax, width='35%', height='35%', loc='lower left', borderpad=0.8)
            
#             # Compute ratio, handling division by zero robustly
#             ratio_null = np.where(
#                 np.isfinite(pk_clean) & (pk_clean > 1e-20),
#                 pk_null / pk_clean,
#                 np.nan
#             )
            
#             # Plot only valid ratio values
#             valid_ratio = np.isfinite(ratio_null) & (ratio_null > 0)
#             if np.any(valid_ratio):
#                 axins.semilogx(kcen[valid_ratio], ratio_null[valid_ratio], 'C1-', linewidth=1.5, alpha=0.8)
            
#             axins.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
#             axins.set_xlabel('k', fontsize=8)
#             axins.set_ylabel('Ratio', fontsize=8)
#             axins.tick_params(labelsize=7)
#             axins.grid(alpha=0.2, which='both')
            
#             # Set reasonable y-limits for inset, handling all-NaN case
#             valid_ratio_filtered = ratio_null[(valid_ratio) & (ratio_null >= 0.1) & (ratio_null <= 10)]
#             if len(valid_ratio_filtered) > 0:
#                 ratio_min = np.min(valid_ratio_filtered)
#                 ratio_max = np.max(valid_ratio_filtered)
#                 axins.set_ylim(max(0.5, ratio_min * 0.8), min(2.0, ratio_max * 1.2))
#             else:
#                 axins.set_ylim(0.5, 2.0)
    
#     fig.suptitle('Impact of Nulling Lowest μ Bin on Recovered P_ℓ(k)', fontsize=13, y=0.995)
#     plt.tight_layout()
    
#     fpath = os.path.join(fig_dir, 'null_mu_bin_effect.png')
#     fig.savefig(fpath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f'Saved: {fpath}')
    
#     # Print summary stats per mode and ℓ
#     print("\nNull μ-bin effect summary (ℓ ∈ {0, 2, 4}):")
#     print("-" * 70)
#     for mode in contam_modes:
#         plk_null = results[mode]['all_plk_null_lowest_mu']
#         plk_null_avg = np.nanmean(plk_null, axis=0)
#         plk_contam = results[mode]['all_plk']
#         plk_contam_avg = np.nanmean(plk_contam, axis=0)
        
#         print(f"\n{mode}:")
#         print(f"  {'ℓ':<4} {'Mean ratio':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
#         for ell, ell_idx in zip(target_ells, target_ell_indices):
#             ratio = np.abs(plk_null_avg[ell_idx, :]) / (np.abs(plk_clean_avg[ell_idx, :]) + 1e-20)
#             # Filter out invalid values for stats
#             ratio_valid = ratio[np.isfinite(ratio) & (ratio > 0)]
#             if len(ratio_valid) > 0:
#                 mean_r = np.mean(ratio_valid)
#                 std_r = np.std(ratio_valid)
#                 min_r = np.min(ratio_valid)
#                 max_r = np.max(ratio_valid)
#                 n_valid = len(ratio_valid)
#                 n_total = len(ratio)
#                 print(f"  {ell:<4} {mean_r:<12.4f} {std_r:<12.4f} {min_r:<12.4f} {max_r:<12.4f} ({n_valid}/{n_total} valid)")
#             else:
#                 print(f"  {ell:<4} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} (no valid data)")
#     print("-" * 70)


def plot_sys_map(fig_dir: str) -> None:
    """Plot the 2D transverse contamination field in the periodic box.
    
    Uses the same seed as the pipeline for mock 0 to ensure consistency
    with the injected galaxy density slices visualization.
    """
    os.makedirs(fig_dir, exist_ok=True)
    boxsize = 1000.0
    mock_idx = 0

    # Use the same seed as the pipeline for mock 0
    # _stage_seeds: base = spec.seed + mock_idx * 10_000; seed = base + 2
    # ExperimentSpec default seed is 42

    seed_dust = None
    if SEED is not None:
        seed_dust = SEED

    if seed_dust is None:
        seed_dust = 42 + mock_idx * 10_000 + 2

    sys_map = gen_controlled_transverse_map(
        amp=SYS_AMP,
        seed=seed_dust,
        spec_type=SYS_SPEC_TYPE,
        ell_max=SYS_ELL_MAX,
        ell_min=SYS_ELL_MIN,
        ell_delta=SYS_ELL_DELTA,
        periodic=True,
        boxsize=boxsize,
        ngrid=NGRID_SYS,
    )

    extent = [0, boxsize, 0, boxsize]
    vmax = np.abs(sys_map).max()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(sys_map.T, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\delta_{\rm sys}(x,y)$', fontsize=11)
    ax.set_xlabel('x [Mpc/h]', fontsize=11)
    ax.set_ylabel('y [Mpc/h]', fontsize=11)

    # ax.set_title(
    #     f'Transverse contamination field\n'
    #     f'spec_type={SYS_SPEC_TYPE}, '
    #     f'$\\ell\\in[{SYS_ELL_MIN},{SYS_ELL_MAX}]$, amp={SYS_AMP}',
    #     fontsize=11,
    # )
    ax.set_title(
        f'Transverse contamination field\n'
        f'spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}',
        fontsize=11,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'sys_map_2d.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_sys_map_healpix(fig_dir: str) -> None:
    """Plot the full-sky transverse contamination field for halfdome in Mollweide projection."""
    import healpy as hp
    
    os.makedirs(fig_dir, exist_ok=True)

    sys_map = _load_sys_map_healpix()
    
    vmax = np.abs(sys_map).max()
    
    fig = plt.figure(figsize=(10, 6))
    hp.mollview(sys_map, fig=fig, title=None, min=-vmax, max=vmax, cmap='RdBu_r', format='%.3f')
    plt.title(
        f'Transverse contamination field (Mollweide)\n'
        f'spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}',
        fontsize=11,
        pad=20,
    )
    fpath = os.path.join(fig_dir, 'sys_map_healpix.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_density_distribution_healpix(fig_dir: str) -> None:
    """Visualize galaxy angular distribution on the sky for halfdome mocks."""
    import healpy as hp
    
    os.makedirs(fig_dir, exist_ok=True)
    mock_idx = 0
    seed_dust = 42 + mock_idx * 10_000 + 2
    
    # Load a sample of halfdome galaxies
    dm = desi_mock()
    dm.halfdome_mock_basedir = DEFAULT_HALFDOME_BASEDIR
    galpos, redshift = dm.load_halfdome_mock(mock_idx, n_sample=500_000)
    
    # Convert to RA/Dec
    boxsize = 1000.0
    ra, dec, r = convert_to_ra_dec_distance(galpos, boxsize, center_offset_mpc=0.0)
    
    # Generate the contamination map
    sys_map = _load_sys_map_healpix()
    
    nside = hp.npix2nside(len(sys_map))
    
    # Convert RA/Dec to HEALPix pixel indices
    theta = np.radians(90.0 - dec)  # colatitude from Dec
    phi = np.radians(ra)             # azimuth from RA
    pix = hp.ang2pix(nside, theta, phi)
    
    # Count galaxies per HEALPix pixel
    npix = hp.nside2npix(nside)
    gal_density = np.zeros(npix)
    np.add.at(gal_density, pix, 1)
    
    # Normalize by pixel area to get surface density
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    gal_density /= pixel_area
    
    # Get multiplicative weights from sys_map
    weights_mult = 1.0 + sys_map[pix]
    
    # Create weighted density map
    gal_density_mult = np.zeros(npix)
    np.add.at(gal_density_mult, pix, weights_mult)
    gal_density_mult /= pixel_area
    
    # Create figure with three Mollweide projections
    fig = plt.figure(figsize=(16, 5))
    
    # Clean
    ax1 = fig.add_subplot(131, projection='mollweide')
    vmax_gal = np.percentile(gal_density[gal_density > 0], 95)
    hp.mollview(gal_density, fig=fig, sub=131, title=None, min=0, max=vmax_gal * 1.1, 
                cmap='viridis', format='%.3f')
    plt.title('Clean — galaxy density', fontsize=11, pad=10)
    
    # Multiplicative systematic
    ax2 = fig.add_subplot(132, projection='mollweide')
    hp.mollview(gal_density_mult, fig=fig, sub=132, title=None, min=0, max=vmax_gal * 1.1, 
                cmap='viridis', format='%.3f')
    plt.title('Transverse multiplicative — galaxy density', fontsize=11, pad=10)
    
    # Difference
    ax3 = fig.add_subplot(133, projection='mollweide')
    diff_map = gal_density_mult - gal_density
    vd = np.percentile(np.abs(diff_map), 95)
    hp.mollview(diff_map, fig=fig, sub=133, title=None, min=-vd, max=vd, 
                cmap='RdBu_r', format='%.3f')
    plt.title(r'$\Delta N$ vs. clean', fontsize=11, pad=10)
    
    fig.suptitle(
        f'Galaxy density distribution (mock #{mock_idx})  |  run={RUN_NAME}\n'
        f'spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}' +
        (f', ell_delta={SYS_ELL_DELTA}' if SYS_ELL_DELTA is not None else
         f', ell_min={SYS_ELL_MIN}, ell_max={SYS_ELL_MAX}'),
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'density_distribution_healpix.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_mask_and_contamination_diagnostics(fig_dir: str) -> None:
    """
    Diagnostic figure to verify mask application and contamination field properties.
    
    Shows:
    1. The mask itself (where 1=unmasked, 0=masked)
    2. The contamination field overlaid with mask boundary
    3. Histograms of contamination values in masked vs unmasked regions
    4. Numerical statistics for verification
    """
    import healpy as hp
    
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load the contamination field
    sys_map = _load_sys_map_healpix()
    nside = hp.npix2nside(len(sys_map))
    
    # Build mask: 1 where unmasked, 0 where masked
    mask = np.ones(len(sys_map), dtype=float)
    
    if GAL_LAT_CUT_DEG > 0.0 and SYS_SPEC_TYPE == 'gaia_stellar':
        from pipeline import _gal_lat_b
        pix_theta, pix_phi = hp.pix2ang(nside, np.arange(len(sys_map)))
        pix_b = _gal_lat_b(np.degrees(pix_phi), 90.0 - np.degrees(pix_theta))
        mask[np.abs(pix_b) < GAL_LAT_CUT_DEG] = 0.0
    
    # Figure 1: Mask and contamination field
    fig = plt.figure(figsize=(16, 6))
    
    # Panel 1: Mask
    ax1 = fig.add_subplot(131, projection='mollweide')
    hp.mollview(mask, fig=fig, sub=131, title=None, min=0, max=1, 
                cmap='RdYlGn', format='%.1f')
    plt.title(f'Mask (1=unmasked, 0=masked)\n|b| < {GAL_LAT_CUT_DEG}°' if GAL_LAT_CUT_DEG > 0 
              else 'No mask applied', fontsize=11, pad=10)
    
    # Panel 2: Contamination field
    ax2 = fig.add_subplot(132, projection='mollweide')
    vmax = np.abs(sys_map).max()
    hp.mollview(sys_map, fig=fig, sub=132, title=None, min=-vmax, max=vmax, 
                cmap='RdBu_r', format='%.3f')
    plt.title('Contamination field', fontsize=11, pad=10)
    
    # Panel 3: Contamination field with mask overlaid (masked regions set to zero)
    ax3 = fig.add_subplot(133, projection='mollweide')
    sys_map_masked = sys_map * mask  # zero out masked regions for visualization
    hp.mollview(sys_map_masked, fig=fig, sub=133, title=None, min=-vmax, max=vmax, 
                cmap='RdBu_r', format='%.3f')
    plt.title('Contamination field (masked=0)', fontsize=11, pad=10)
    
    fig.suptitle(
        f'Mask diagnostics  |  spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}',
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'mask_and_contamination.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')
    
    # Figure 2: Histograms and statistics
    unmasked_vals = sys_map[mask > 0.5]
    masked_vals = sys_map[mask < 0.5]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    if len(unmasked_vals) > 0:
        ax.hist(unmasked_vals, bins=50, alpha=0.7, label=f'Unmasked (N={len(unmasked_vals)})', color='C0', density=True)
    if len(masked_vals) > 0:
        ax.hist(masked_vals, bins=50, alpha=0.7, label=f'Masked (N={len(masked_vals)})', color='C1', density=True)
    ax.set_xlabel('Contamination field value', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('Distribution of contamination values', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Statistics table as text
    ax = axes[1]
    ax.axis('off')
    
    stats_text = (
        f'STATISTICS\n'
        f'{"="*50}\n\n'
        f'Unmasked region:\n'
        f'  Count: {len(unmasked_vals)}\n'
        f'  Mean: {unmasked_vals.mean():.6f}\n'
        f'  Std: {unmasked_vals.std():.6f}\n'
        f'  Min: {unmasked_vals.min():.6f}\n'
        f'  Max: {unmasked_vals.max():.6f}\n\n'
    )
    
    if len(masked_vals) > 0:
        stats_text += (
            f'Masked region (|b| < {GAL_LAT_CUT_DEG}°):\n'
            f'  Count: {len(masked_vals)}\n'
            f'  Mean: {masked_vals.mean():.6f}\n'
            f'  Std: {masked_vals.std():.6f}\n'
            f'  Min: {masked_vals.min():.6f}\n'
            f'  Max: {masked_vals.max():.6f}\n\n'
        )
    
    stats_text += (
        f'Global:\n'
        f'  Total pixels: {len(sys_map)}\n'
        f'  Mean (all): {sys_map.mean():.6f}\n'
        f'  Std (all): {sys_map.std():.6f}\n'
        f'  Expected amp: {SYS_AMP:.6f}\n'
        f'  Actual amp (unmasked): {unmasked_vals.std():.6f}\n'
    )
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(
        f'Contamination field statistics  |  spec_type={SYS_SPEC_TYPE}',
        fontsize=12,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'contamination_statistics.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')
    
    # Print to console for quick reference
    print('\n' + '='*60)
    print('MASK AND CONTAMINATION DIAGNOSTICS')
    print('='*60)
    print(f'spec_type: {SYS_SPEC_TYPE}, sys_amp: {SYS_AMP}')
    if GAL_LAT_CUT_DEG > 0:
        print(f'Galactic latitude cut: |b| < {GAL_LAT_CUT_DEG}°')
    print('\nUnmasked region:')
    print(f'  Count: {len(unmasked_vals)} pixels')
    print(f'  Mean: {unmasked_vals.mean():.6f}')
    print(f'  Std: {unmasked_vals.std():.6f}')
    if len(masked_vals) > 0:
        print(f'\nMasked region:')
        print(f'  Count: {len(masked_vals)} pixels')
        print(f'  Mean: {masked_vals.mean():.6f} (should be ≈0 if properly masked)')
        print(f'  Std: {masked_vals.std():.6f} (should be ≈0 if properly masked)')
    print('='*60 + '\n')


def plot_catalog_positions_with_masking(fig_dir: str) -> None:
    """
    Diagnostic visualization of galaxy and random catalog positions with masking applied.
    
    Shows subsampled 3D positions in multiple slices to verify that:
    - Galactic latitude masking is correctly removing galaxies/randoms
    - Gaia coverage mask regions are properly excluded
    - Catalog geometry and depth look reasonable
    """
    import healpy as hp
    from pipeline import _gal_lat_b, _stage_seeds
    
    os.makedirs(fig_dir, exist_ok=True)
    
    print('Loading halfdome mock catalog for position diagnostics...')
    mock_idx = 0
    dm = desi_mock()
    dm.halfdome_mock_basedir = DEFAULT_HALFDOME_BASEDIR
    if not dm.halfdome_mock_basedir.endswith('/'):
        dm.halfdome_mock_basedir += '/'
    
    # Load halfdome catalog following pipeline.py logic
    seeds = _stage_seeds(ExperimentSpec(), mock_idx)
    galpos, z_array = dm.load_halfdome_mock(mock_idx, n_sample=N_SAMPLE, seed=seeds['quijote'])
    boxsize = 1000.0
    ra, dec, r = convert_to_ra_dec_distance(galpos, boxsize, center_offset_mpc=0.0)
    r_values = np.asarray(r.value if hasattr(r, 'value') else r)
    
    # Apply redshift selection (same as pipeline)
    redshift_mask = np.ones_like(z_array, dtype=bool)
    redshift_mask &= z_array > Z_MIN
    redshift_mask &= z_array < Z_MAX
    ra = ra[redshift_mask]
    dec = dec[redshift_mask]
    r_values = r_values[redshift_mask]
    z_array = z_array[redshift_mask]
    print(f'  After redshift selection [{Z_MIN}, {Z_MAX}]: {len(ra):,} galaxies')
    
    # Cache original z-distribution BEFORE masking (same as pipeline does)
    # This ensures randoms maintain constant comoving density even when data is masked
    z_orig = z_array.copy()
    
    ra_orig = ra.copy()
    dec_orig = dec.copy()
    r_orig = r_values.copy()
    
    # Apply galactic latitude masking
    if GAL_LAT_CUT_DEG > 0.0:
        pix_b = _gal_lat_b(ra, dec)
        keep_mask = np.abs(pix_b) >= GAL_LAT_CUT_DEG
        n_before = len(ra)
        ra = ra[keep_mask]
        dec = dec[keep_mask]
        r_values = r_values[keep_mask]
        z_array = z_array[keep_mask]
        print(f'  After gal_lat_cut |b|>{GAL_LAT_CUT_DEG}°: {len(ra):,} galaxies '
              f'(removed {n_before - len(ra):,})')
    
    ra_masked = ra
    dec_masked = dec
    r_masked = r_values
    
    # Generate randoms using ORIGINAL z-distribution (same as pipeline)
    # This maintains constant comoving density even when data is masked
    n_randoms = int(5 * len(ra_masked))
    rng = np.random.default_rng(seed=42)
    
    # Sample redshifts from the original (pre-mask) distribution
    rand_indices = rng.choice(len(z_orig), size=n_randoms, replace=True)
    z_rand = z_orig[rand_indices]
    
    # Convert redshifts to comoving distance (same as pipeline.generate_uniform_randoms)
    from utils import grab_chi_interp
    from astropy.cosmology import Planck18 as cosmo_diag
    chi_interp = grab_chi_interp()
    r_rand_mpc = chi_interp(z_rand)
    r_rand = r_rand_mpc * cosmo_diag.h  # convert to Mpc/h
    
    # Generate uniform RA/Dec
    ra_rand = rng.uniform(0.0, 360.0, size=n_randoms)
    dec_rand = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n_randoms)))
    
    # Apply same masking to randoms
    if GAL_LAT_CUT_DEG > 0.0:
        pix_b_rand = _gal_lat_b(ra_rand, dec_rand)
        keep_mask_rand = np.abs(pix_b_rand) >= GAL_LAT_CUT_DEG
        ra_rand = ra_rand[keep_mask_rand]
        dec_rand = dec_rand[keep_mask_rand]
        r_rand = r_rand[keep_mask_rand]
        print(f'  Randoms after gal_lat_cut: {len(ra_rand):,}')
    
    # Subsample for visualization (keep only ~10k for clarity)
    max_points = 10000
    if len(ra_masked) > max_points:
        idx_subsample = rng.choice(len(ra_masked), size=max_points, replace=False)
        ra_vis = ra_masked[idx_subsample]
        dec_vis = dec_masked[idx_subsample]
        r_vis = r_masked[idx_subsample]
    else:
        ra_vis = ra_masked
        dec_vis = dec_masked
        r_vis = r_masked
    
    if len(ra_rand) > max_points:
        idx_subsample_rand = rng.choice(len(ra_rand), size=max_points, replace=False)
        ra_rand_vis = ra_rand[idx_subsample_rand]
        dec_rand_vis = dec_rand[idx_subsample_rand]
        r_rand_vis = r_rand[idx_subsample_rand]
    else:
        ra_rand_vis = ra_rand
        dec_rand_vis = dec_rand
        r_rand_vis = r_rand
    
    # Create figure with multiple slices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: RA vs Dec (full angular sky)
    ax = axes[0, 0]
    ax.scatter(ra_vis, dec_vis, s=1, alpha=0.3, c='C0', label=f'Data ({len(ra_vis):,})')
    ax.scatter(ra_rand_vis, dec_rand_vis, s=1, alpha=0.2, c='C1', label=f'Randoms ({len(ra_rand_vis):,})')
    if GAL_LAT_CUT_DEG > 0.0:
        # Show the Galactic plane mask region
        ax.axhline(-GAL_LAT_CUT_DEG, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label=f'|Dec| = {GAL_LAT_CUT_DEG}°')
        ax.axhline(GAL_LAT_CUT_DEG, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('RA (degrees)', fontsize=11)
    ax.set_ylabel('Dec (degrees)', fontsize=11)
    ax.set_title('Angular positions (RA vs Dec)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)
    
    # Panel 2: RA vs r (depth along RA)
    ax = axes[0, 1]
    ax.scatter(ra_vis, r_vis, s=1, alpha=0.3, c='C0', label='Data')
    ax.scatter(ra_rand_vis, r_rand_vis, s=1, alpha=0.2, c='C1', label='Randoms')
    ax.set_xlabel('RA (degrees)', fontsize=11)
    ax.set_ylabel('Comoving distance (Mpc/h)', fontsize=11)
    ax.set_title('Radial distribution vs RA', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 3: Dec vs r (depth along Dec)
    ax = axes[1, 0]
    ax.scatter(dec_vis, r_vis, s=1, alpha=0.3, c='C0', label='Data')
    ax.scatter(dec_rand_vis, r_rand_vis, s=1, alpha=0.2, c='C1', label='Randoms')
    if GAL_LAT_CUT_DEG > 0.0:
        ax.axvline(-GAL_LAT_CUT_DEG, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(GAL_LAT_CUT_DEG, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Dec (degrees)', fontsize=11)
    ax.set_ylabel('Comoving distance (Mpc/h)', fontsize=11)
    ax.set_title('Radial distribution vs Dec', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 4: Radial distribution histogram
    ax = axes[1, 1]
    ax.hist(r_vis, bins=50, alpha=0.6, label=f'Data (N={len(r_vis):,})', density=False, color='C0')
    ax.hist(r_rand_vis, bins=50, alpha=0.4, label=f'Randoms (N={len(r_rand_vis):,})', density=False, color='C1')
    ax.set_xlabel('Comoving distance (Mpc/h)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Radial distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    fig.suptitle(
        f'Catalog positions with masking  |  mock_idx={mock_idx}' +
        (f', gal_lat_cut={GAL_LAT_CUT_DEG}°' if GAL_LAT_CUT_DEG > 0.0 else ''),
        fontsize=12,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'catalog_positions_with_masking.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}\n')


def plot_angular_power_spectrum_healpix(fig_dir: str) -> None:
    """Plot the angular power spectrum C_ell of the halfdome contamination field."""
    import healpy as hp

    os.makedirs(fig_dir, exist_ok=True)

    sys_map = _load_sys_map_healpix()

    # Compute power spectrum via healpy anafast.
    # For power_law we need lmax >> SYS_ELL_MAX since the map has no hard upper cutoff.
    nside = hp.npix2nside(len(sys_map))
    if SYS_SPEC_TYPE in ('gaia_stellar', 'power_law'):
        lmax_compute = min(3 * nside - 1, 1024)
    else:
        lmax_compute = min(4 * nside, SYS_ELL_MAX + 50)
    cl = hp.anafast(sys_map, pol=False, lmax=lmax_compute)
    ells = np.arange(len(cl))

    # Expected power spectrum based on spec_type (not applicable for gaia_stellar)
    cl_expected = np.zeros_like(ells, dtype=float)
    if SYS_SPEC_TYPE == 'flat':
        mask = (ells >= SYS_ELL_MIN) & (ells <= SYS_ELL_MAX)
        cl_expected[mask] = 1.0
    elif SYS_SPEC_TYPE == 'power_law':
        # No hard upper cutoff — same as contamination.py
        mask = ells >= max(SYS_ELL_MIN, 1)
        cl_expected[mask] = 1.0 / (ells[mask] ** 2)
    elif SYS_SPEC_TYPE == 'delta':
        if SYS_ELL_DELTA is not None and SYS_ELL_DELTA < len(cl_expected):
            cl_expected[SYS_ELL_DELTA] = 1.0
    
    # Normalize both to peak value so they sit on the same scale
    cl_peak = cl[max(SYS_ELL_MIN, 1):].max() if cl[max(SYS_ELL_MIN, 1):].max() > 0 else 1.0
    cl_norm = cl / cl_peak
    if cl_expected.max() > 0:
        cl_expected /= cl_expected.max()
    
    # x-axis: show from 0 to where measured Cl drops to 1e-4 of peak, or lmax_compute
    significant = np.where(cl_norm > 1e-4)[0]
    xlim_max = int(significant[-1] * 1.05) + 5 if len(significant) > 0 else lmax_compute
    xlim_max = min(xlim_max, lmax_compute)
    
    # y-axis: peak-relative, floor at 1e-4 of peak
    ylim_lo = 3e-5
    ylim_hi = cl_norm.max() * 3.0
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(ells, cl_norm, 'o-', linewidth=1.5, markersize=3, label='Measured $C_\ell$', color='C0')
    if np.any(cl_expected > 0):
        ax.semilogy(ells, cl_expected, '--', linewidth=2, label='Expected spectrum', color='C1')
    ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax.set_ylabel(r'$C_\ell$ [normalized to peak]', fontsize=12)
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_title(
        f'Angular power spectrum of contamination field\n'
        f'spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}' +
        (f', $\ell_\delta$={SYS_ELL_DELTA}' if SYS_ELL_DELTA is not None else
         (f', $|b|>{GAL_LAT_CUT_DEG:.0f}\\deg$ mask' if SYS_SPEC_TYPE == 'gaia_stellar' and GAL_LAT_CUT_DEG > 0 else
          ('' if SYS_SPEC_TYPE == 'gaia_stellar' else
           f', $\ell \in [{SYS_ELL_MIN}, {SYS_ELL_MAX}]$'))),
        fontsize=11,
    )
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'angular_power_spectrum.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_angular_power_spectrum_periodic(fig_dir: str) -> None:
    """Plot the angular power spectrum of the periodic contamination field (2D FFT)."""
    os.makedirs(fig_dir, exist_ok=True)
    mock_idx = 0
    boxsize = 1000.0
    ngrid = NGRID_SYS

    seed_dust = None

    if SEED is not None:
        seed_dust = SEED

    if seed_dust is None:
        seed_dust = 42 + mock_idx * 10_000 + 2
    
    sys_map = gen_controlled_transverse_map(
        amp=SYS_AMP,
        seed=seed_dust,
        spec_type=SYS_SPEC_TYPE,
        ell_max=SYS_ELL_MAX,
        ell_min=SYS_ELL_MIN,
        ell_delta=SYS_ELL_DELTA,
        periodic=True,
        boxsize=boxsize,
        ngrid=ngrid,
    )
    
    # Compute 2D power spectrum
    pk2d = np.abs(np.fft.rfft2(sys_map)) ** 2
    pk2d /= (ngrid ** 4)  # Normalize by grid size
    
    # Bin by radial k-mode
    nr = pk2d.shape[1]
    nk = pk2d.shape[0]
    kx = np.fft.fftfreq(ngrid, d=boxsize / ngrid)[:nk]
    ky = np.fft.rfftfreq(ngrid, d=boxsize / ngrid)[:nr]
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(KX ** 2 + KY ** 2)
    
    # Radial binning
    k_edges = np.linspace(0, np.sqrt(2) * np.pi * ngrid / boxsize, 50)
    k_centers = (k_edges[:-1] + k_edges[1:]) / 2
    pk_radial = np.zeros_like(k_centers)
    for i, (k_lo, k_hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        mask = (k_mag >= k_lo) & (k_mag < k_hi)
        if mask.sum() > 0:
            pk_radial[i] = pk2d[mask].mean()
    
    # Filter out any zero/NaN values for plotting
    valid = pk_radial > 0
    k_centers = k_centers[valid]
    pk_radial = pk_radial[valid]
    
    # Normalize to peak and set ylim to peak-relative floor
    pk_peak = pk_radial.max() if pk_radial.max() > 0 else 1.0
    pk_norm = pk_radial / pk_peak
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(k_centers, pk_norm, 'o-', linewidth=1.5, markersize=5, label='Measured $P(k)$', color='C0')
    
    # ylim: show down to 1e-4 of peak
    significant = pk_norm > 1e-4
    if significant.any():
        ax.set_ylim(3e-5, pk_norm.max() * 3.0)
    
    ax.set_xlabel(r'Wavenumber $k$ [Mpc$^{-1}$ h]', fontsize=12)
    ax.set_ylabel(r'$P(k)$ [normalized]', fontsize=12)
    ax.set_title(
        f'Angular power spectrum of contamination field (periodic box)\n'
        f'spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}, ngrid={ngrid}' +
        (f', $\ell_\delta$={SYS_ELL_DELTA}' if SYS_ELL_DELTA is not None else ''),
        fontsize=11,
    )
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'angular_power_spectrum_periodic.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_density_slices(fig_dir: str) -> None:
    """
    Visualize galaxy density in the transverse (x-y) plane for each contamination
    mode using a single Quijote mock.

    For 'none': 2D histogram of raw galaxy positions.
    For 'transverse_additive': same + injected galaxies shown separately.
    For 'transverse_multiplicative': 2D histogram weighted by (1 + delta_sys).
    """
    os.makedirs(fig_dir, exist_ok=True)
    mock_idx = 0
    boxsize = 1000.0
    ngrid = NGRID_SYS
    nbins = 128   # histogram bins per axis

    # Load raw Cartesian positions for one mock
    dm = desi_mock()
    dm.quijote_mock_basedir = DEFAULT_QUIJOTE_BASEDIR
    galpos = dm.load_quijote_galpos(mock_idx, with_RSD=False, ds_fac=DS_FAC)
    # galpos: (N, 3) array in [0, boxsize]
    x, y, z = galpos[:, 0], galpos[:, 1], galpos[:, 2]
    n_gal = len(x)

    # Generate the contamination map using the same seed as the pipeline for mock 0.
    # _stage_seeds: base = spec.seed + mock_idx * 10_000; seeds['dust'] = base + 2
    # ExperimentSpec default seed is 42
    seed_dust = 42 + mock_idx * 10_000 + 2
    sys_map = gen_controlled_transverse_map(
        amp=SYS_AMP,
        seed=seed_dust,
        spec_type=SYS_SPEC_TYPE,
        ell_max=SYS_ELL_MAX,
        ell_min=SYS_ELL_MIN,
        ell_delta=SYS_ELL_DELTA,
        periodic=True,
        boxsize=boxsize,
        ngrid=ngrid,
    )

    # Map each galaxy to the sys_map pixel
    ix = (x / boxsize * ngrid).astype(int) % ngrid
    iy = (y / boxsize * ngrid).astype(int) % ngrid
    weights_mult = 1.0 + sys_map[ix, iy]   # multiplicative weight per galaxy

    # Additive: rejection-sample n_inject new positions from positive lobe
    rng = np.random.default_rng(seed_dust + 1)
    n_inject = max(1, int(SYS_AMP * n_gal))
    positive_map = np.clip(sys_map, 0.0, None)
    max_val = positive_map.max()
    accepted_xy = []
    batch = n_inject * 5
    while len(accepted_xy) < n_inject:
        xs = rng.uniform(0.0, boxsize, size=batch)
        ys = rng.uniform(0.0, boxsize, size=batch)
        ixs = (xs / boxsize * ngrid).astype(int) % ngrid
        iys = (ys / boxsize * ngrid).astype(int) % ngrid
        vals = positive_map[ixs, iys]
        keep = rng.uniform(0.0, max_val, size=batch) < vals
        accepted_xy.extend(zip(xs[keep], ys[keep]))
    accepted_xy = accepted_xy[:n_inject]
    x_inj = np.array([p[0] for p in accepted_xy])
    y_inj = np.array([p[1] for p in accepted_xy])

    # ── Build 2D histograms ──────────────────────────────────────────────────
    edges = np.linspace(0, boxsize, nbins + 1)

    def make_hist(xv, yv, w=None):
        h, _, _ = np.histogram2d(xv, yv, bins=edges, weights=w)
        return h

    h_clean  = make_hist(x, y)
    h_add    = make_hist(np.concatenate([x, x_inj]), np.concatenate([y, y_inj]))
    h_mult   = make_hist(x, y, w=weights_mult)
    h_inj    = make_hist(x_inj, y_inj)   # injected galaxies only

    extent = [0, boxsize, 0, boxsize]

    # ── One row per mode: density map | difference vs. clean ────────────────
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    datasets = [
        ('Clean',                   h_clean, None),
        ('Transverse additive',     h_add,   h_inj),
        ('Transverse multiplicative', h_mult, h_mult - h_clean),
    ]

    for row, (title, h_map, h_diff) in enumerate(datasets):
        ax_map  = axes[row, 0]
        ax_diff = axes[row, 1]

        vmax_map = np.percentile(h_clean, 99)
        im1 = ax_map.imshow(h_map.T, origin='lower', extent=extent,
                            cmap='viridis', vmin=0, vmax=vmax_map * 1.3,
                            interpolation='nearest')
        ax_map.set_title(f'{title} — galaxy density', fontsize=10)
        ax_map.set_xlabel('x [Mpc/h]', fontsize=9)
        ax_map.set_ylabel('y [Mpc/h]', fontsize=9)
        fig.colorbar(im1, ax=ax_map, label='N / pixel')

        if h_diff is not None:
            vd = np.percentile(np.abs(h_diff), 99)
            im2 = ax_diff.imshow(h_diff.T, origin='lower', extent=extent,
                                 cmap='RdBu_r', vmin=-vd, vmax=vd,
                                 interpolation='nearest')
            if row == 1:
                ax_diff.set_title('Injected galaxies', fontsize=10)
            else:
                ax_diff.set_title(r'$\Delta N$ vs. clean', fontsize=10)
            ax_diff.set_xlabel('x [Mpc/h]', fontsize=9)
            ax_diff.set_ylabel('y [Mpc/h]', fontsize=9)
            fig.colorbar(im2, ax=ax_diff, label='ΔN / pixel')
        else:
            ax_diff.set_visible(False)

    fig.suptitle(
        f'Galaxy density maps (mock #{mock_idx}, DS_FAC={DS_FAC})  |  run={RUN_NAME}\n'
        f'spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}' +
        (f', ell_delta={SYS_ELL_DELTA}' if SYS_ELL_DELTA is not None else
         f', ell_min={SYS_ELL_MIN}, ell_max={SYS_ELL_MAX}'),
        fontsize=11,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'density_slices.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def _compute_pkmu_with_jax(
    positions: np.ndarray,
    weights: np.ndarray,
    randoms_positions: np.ndarray,
    randoms_weights: np.ndarray,
    los: str,
    position_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute P(k,μ) using jax-power.

    Returns kcen, pkmu arrays.
    """
    from jaxpower import (
        MeshAttrs, ParticleField, BinMesh2SpectrumPoles,
        compute_mesh2_spectrum
    )
    from nonunif_binning import compute_null_bins

    print(f'[jax-power] Computing diagnostic P(k,μ) (position_type={position_type}, los={los})')

    kedges = np.arange(K_MIN, K_MAX + DELTA_K, DELTA_K)
    mu_wedges = compute_null_bins(np.max(ELLS), N_CLEAN_BINS)

    # positions is shape (3, N); jaxpower expects (N, 3)
    data_pos = positions.T
    data_w = weights
    rand_pos = randoms_positions.T
    rand_w = randoms_weights

    # Create mesh geometry
    if position_type == 'xyz':
        # Periodic box
        mattrs = MeshAttrs(meshsize=NMESH, boxsize=1000.0)
    else:
        # Lightcone (rdd) — not used in this diagnostic, but keep for completeness
        mattrs = MeshAttrs(meshsize=NMESH)

    # Create binning configuration
    bin_config = BinMesh2SpectrumPoles(mattrs, edges=kedges, ells=ELLS)

    # Paint data and randoms to mesh
    data_field = ParticleField(data_pos, data_w, attrs=mattrs)
    rand_field = ParticleField(rand_pos, rand_w, attrs=mattrs)

    # Paint to real space mesh
    data_mesh = data_field.paint(resampler='tsc', interlacing=3, compensate=True)
    rand_mesh = rand_field.paint(resampler='tsc', interlacing=3, compensate=True)

    # FKP estimator: compute data - randoms mesh (element-wise subtraction)
    fkp_mesh = data_mesh - rand_mesh

    # Compute power spectrum
    spectrum = compute_mesh2_spectrum(fkp_mesh, bin=bin_config, los=los)

    kcen = np.asarray(spectrum.coords('k'))
    plk = np.stack(
        [np.asarray(spectrum.get(ell).power) for ell in ELLS],
        axis=0,
    )
    plk_kell = np.moveaxis(plk, 0, 1)
    pkmu = _poles_to_wedges(ELLS, plk_kell, mu_wedges)

    return kcen, pkmu


def plot_contaminant_pkmu(fig_dir: str) -> None:
    """
    Compute and plot the P(k,μ) power spectrum of the pure contamination field
    (no clustering signal), for a periodic box.
    
    This shows the intrinsic power spectrum structure of the contaminant field,
    useful for understanding its coupling to different μ modes.
    
    Method: 
    1. Generate the contamination field sys_map (2D in transverse plane)
    2. Create a uniform random catalog
    3. Weight galaxies by local contamination field value  
    4. Compute P(k,μ) using standard estimator
    """ 
    from pypower import CatalogFFTPower
    from nonunif_binning import compute_null_bins, compute_window_corrected_mu_bins
    
    os.makedirs(fig_dir, exist_ok=True)
    mock_idx = 0
    boxsize = 1000.0
    ngrid_sys = NGRID_SYS
    
    # Generate the contamination field (2D transverse plane)
    # seed_dust = 42 + mock_idx * 10_000 + 2

    seed_dust = None
    if SEED is not None:
        seed_dust = SEED

    if seed_dust is None:
        seed_dust = 42 + mock_idx * 10_000 + 2
        
    sys_map = gen_controlled_transverse_map(
        amp=SYS_AMP,
        seed=seed_dust,
        spec_type=SYS_SPEC_TYPE,
        ell_max=SYS_ELL_MAX,
        ell_min=SYS_ELL_MIN,
        ell_delta=SYS_ELL_DELTA,
        periodic=True,
        boxsize=boxsize,
        ngrid=ngrid_sys,
    )
    
    # Create uniform random catalog of galaxy positions (Cartesian)
    rng = np.random.default_rng(seed=42)
    n_gal = int(0.1 * NMESH ** 3)  # ~10% of mesh volume
    positions = rng.uniform(0.0, boxsize, size=(n_gal, 3))
    
    # Interpolate contamination field values at galaxy positions
    # Map (x, y, z) → grid indices, use only transverse (x, y) for sys_map
    ix = (positions[:, 0] / boxsize * ngrid_sys).astype(int) % ngrid_sys
    iy = (positions[:, 1] / boxsize * ngrid_sys).astype(int) % ngrid_sys
    
    # For multiplicative interpretation: w_gal = 1 + δ_sys
    weights = 1.0 + sys_map[ix, iy]
    weights = np.clip(weights, 0.01, 10.0)
    
    # Keep positions in Cartesian (x, y, z) coordinates for periodic box
    # This ensures los='z' correctly refers to the box z-axis
    pos_array = positions.T  # Transpose to (3, N) format for CatalogFFTPower

    print("pos array has shape", pos_array.shape)
    
    # Generate random catalog (uniform, no weights) in Cartesian coordinates
    n_randoms = int(10 * n_gal)
    rand_positions = rng.uniform(0.0, boxsize, size=(n_randoms, 3))
    rand_array = rand_positions.T  # Transpose to (3, N) format
    rand_weights = np.ones(n_randoms)
    
    # Compute power spectrum
    mu_wedges = get_mu_wedges()
    print('Mu wedges:', mu_wedges)
    kedges = np.arange(K_MIN, K_MAX + DELTA_K, DELTA_K)
    print('k edges in plot contaminant Pkmu:', kedges)

    if USE_JAX:
        print('[jax-power] Computing contamination field P(k,μ) (periodic box)')
        kcen, pkmu = _compute_pkmu_with_jax(
            pos_array,
            weights,
            rand_array,
            rand_weights,
            los='z',
            position_type='xyz',
        )
        pkmu = pkmu.real
    else:
        print('[pypower] Computing contamination field P(k,μ) (periodic box)')
        edges = (kedges, mu_wedges)
        result = CatalogFFTPower(
            data_positions1=pos_array,
            data_weights1=weights,
            randoms_positions1=rand_array,
            randoms_weights1=rand_weights,
            nmesh=NMESH,
            los='z',
            position_type='xyz',
            resampler='tsc',
            dtype='f8',
            ells=ELLS,
            edges=edges,
        )
        pkmu = result.wedges.get_power().real
        kcen = result.wedges.k[:, 0].real
    nmu = len(mu_wedges) - 1
    
    fig = plt.figure(figsize=(6, 5))
    if nmu == 1:
        axes = [axes]

    cmap = plt.get_cmap('jet')
    
    for mu_idx in range(nmu):
        # ax = axes[mu_idx]
        kp = kcen * pkmu[:, mu_idx]
        plt.plot(kcen, kp, 'o-', linewidth=2, markersize=4, color=cmap(mu_idx / nmu),
                label=f'μ ∈ [{mu_wedges[mu_idx]:.2f}, {mu_wedges[mu_idx+1]:.2f}]')

    plt.ylabel(r'$k P(k,\mu)$ [$({\rm Mpc}/h)^2$]', fontsize=11)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=10, loc=3, ncol=2, facecolor='white', edgecolor='gray')
    plt.grid(alpha=0.3, which='both')
    
    plt.xlabel('k [h/Mpc]', fontsize=11)
    plt.title(
        f'Contamination field P(k,μ) (periodic box)\n'
        f'spec_type={SYS_SPEC_TYPE}, sys_amp={SYS_AMP}',
        fontsize=12
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'contaminant_pkmu.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_contaminant_pkmu_halfdome(fig_dir: str) -> None:
    """
    Compute and plot the P(k,μ) power spectrum of the pure contamination field
    for halfdome lightcones.
    
    This shows the intrinsic power spectrum structure of the contaminant field
    in the lightcone geometry with realistic N(z) from halfdome.
    
    Method:
    1. Generate the contamination field sys_map (HEALPix full-sky map)
    2. Load halfdome mock to extract empirical N(z)
    3. Sample RA/Dec positions according to contamination field distribution
    4. All galaxy weights = 1 (contamination is in the positions, not weights)
    5. Sample redshifts from halfdome N(z) to get comoving distances
    6. Compute P(k,μ) using CatalogFFTPower with same config as data
    """
    from pypower import CatalogFFTPower
    from nonunif_binning import compute_null_bins
    from astropy.cosmology import Planck18 as cosmo
    import healpy as hp
    
    os.makedirs(fig_dir, exist_ok=True)
    mock_idx = 0
    nside_sys = 256  # HEALPix resolution for contamination field
    
    print('Loading halfdome mock to extract N(z)...')
    dm = desi_mock()
    dm.halfdome_mock_basedir = DEFAULT_HALFDOME_BASEDIR
    galpos, redshift = dm.load_halfdome_mock(mock_idx, n_sample=1_000_000)

    redshift_sel = (redshift > Z_MIN) & (redshift < Z_MAX)
    galpos = galpos[redshift_sel]
    redshift = redshift[redshift_sel]
    
    # Generate the contamination field (HEALPix full-sky map)
    # seed_dust = 42 + mock_idx * 10_000 + 2

    seed_dust = None
    if SEED is not None:
        seed_dust = SEED

    if seed_dust is None:
        seed_dust = 42 + mock_idx * 10_000 + 2

    print('seed dust in plot_contaminant_pkmu_halfdome is ', seed_dust)
    sys_map = gen_controlled_transverse_map(
        amp=SYS_AMP,
        seed=seed_dust,
        spec_type=SYS_SPEC_TYPE,
        ell_max=SYS_ELL_MAX,
        ell_min=SYS_ELL_MIN,
        ell_delta=SYS_ELL_DELTA,
        periodic=False,
        nside=nside_sys,
    )
    
    # Sample RA/Dec positions according to contamination field distribution
    # Use the field values (normalized) as pixel weights, then sample RA/Dec from those pixels
    rng = np.random.default_rng(seed=42)
    n_gal = int(0.1 * NMESH ** 3)  # ~10% of mesh volume, ~1.3M for NMESH=512
    
    # Normalize contamination map to use as sampling probabilities
    # Use absolute value to handle both positive and negative deviations
    prob_map = np.abs(sys_map)
    prob_map = prob_map / np.sum(prob_map)  # Normalize to sum to 1
    
    # Sample pixel indices according to probability distribution
    npix = len(sys_map)
    pixel_indices = rng.choice(npix, size=n_gal, replace=True, p=prob_map)
    
    # Convert pixel indices to RA/Dec
    # Get pixel centers
    theta_pix, phi_pix = hp.pix2ang(nside_sys, pixel_indices, nest=False)
    ra_gal = np.degrees(phi_pix)
    dec_gal = 90.0 - np.degrees(theta_pix)
    
    # Add small random offsets within pixels for smoother distribution
    pixel_size_deg = np.degrees(hp.nside2resol(nside_sys))
    ra_gal += rng.uniform(-pixel_size_deg/2, pixel_size_deg/2, size=n_gal)
    dec_gal += rng.uniform(-pixel_size_deg/2, pixel_size_deg/2, size=n_gal)
    
    # Wrap RA to [0, 360) and clip Dec to [-90, 90]
    ra_gal = ra_gal % 360.0
    dec_gal = np.clip(dec_gal, -90.0, 90.0)
    
    # Sample redshifts from halfdome N(z) distribution
    # Use empirical CDF from loaded redshifts
    z_sorted = np.sort(redshift)
    z_min, z_max = z_sorted[0], z_sorted[-1]
    
    # Draw uniform samples and interpolate into sorted z array
    uniform_samples = rng.uniform(0, len(redshift) - 1, size=n_gal)
    z_sampled = np.interp(uniform_samples, np.arange(len(redshift)), z_sorted)
    
    # Convert redshifts to comoving distances
    r_sampled_mpc = cosmo.comoving_distance(z_sampled).value  # Mpc (physical)
    r_sampled = r_sampled_mpc * cosmo.h  # Convert to Mpc/h
    
    # Arrange as (3, N) for CatalogFFTPower with position_type='rdd'
    pos_array = np.vstack([ra_gal, dec_gal, r_sampled])
    
    # All data weights are 1 (contamination is in spatial distribution only)
    weights = np.ones(n_gal)
    
    print(f'Contamination field catalog: {n_gal:,} galaxies (sampled from field)')
    print(f'  RA range: [{ra_gal.min():.2f}, {ra_gal.max():.2f}]°')
    print(f'  Dec range: [{dec_gal.min():.2f}, {dec_gal.max():.2f}]°')
    print(f'  z range: [{z_sampled.min():.4f}, {z_sampled.max():.4f}]')
    print(f'  r range: [{r_sampled.min():.2f}, {r_sampled.max():.2f}] Mpc/h')
    
    # Generate random catalog (uniform RA/Dec, unit weights)
    n_randoms = int(10 * n_gal)
    ra_rand = rng.uniform(0.0, 360.0, size=n_randoms)
    dec_rand = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n_randoms)))
    
    # Sample redshifts from same distribution
    uniform_samples_rand = rng.uniform(0, len(redshift) - 1, size=n_randoms)
    z_rand = np.interp(uniform_samples_rand, np.arange(len(redshift)), z_sorted)
    r_rand_mpc = cosmo.comoving_distance(z_rand).value
    r_rand = r_rand_mpc * cosmo.h
    
    rand_array = np.vstack([ra_rand, dec_rand, r_rand])
    rand_weights = np.ones(n_randoms)
    
    # Build k and mu bins using same config as data
    mu_wedges = get_mu_wedges()
    kedges = np.arange(K_MIN, K_MAX + DELTA_K, DELTA_K)

    print(f'Mu wedges: {mu_wedges}')
    print(f'K edges: {kedges[:5]}...{kedges[-5:]}')

    if USE_JAX:
        print('[jax-power] Computing contamination field P(k,μ) (halfdome lightcone)')
        kcen, pkmu = _compute_pkmu_with_jax(
            pos_array,
            weights,
            rand_array,
            rand_weights,
            los='firstpoint',
            position_type='rdd',
        )
        pkmu_from_pypower = None  # Not computed for jax version
    else:
        print('[pypower] Computing contamination field P(k,μ) (halfdome lightcone)')
        result = CatalogFFTPower(
            data_positions1=pos_array,
            data_weights1=weights,
            randoms_positions1=rand_array,
            randoms_weights1=rand_weights,
            nmesh=NMESH,
            los='firstpoint',
            position_type='rdd',
            resampler='tsc',
            dtype='f8',
            ells=ELLS,
            edges=kedges,
            interlacing=3,
            shotnoise=None,
            mpiroot=0,
        )
        plk = result.poles.get_power()
        kcen = result.poles.k
        plk_kell = np.moveaxis(plk, 0, 1)
        pkmu_from_pypower = result.poles.to_wedges(mu_wedges).get_power()
        pkmu = _poles_to_wedges(ELLS, plk_kell, mu_wedges)
    

    nmu = len(mu_wedges) - 1
    # Plot k*P(k,μ) for each mu wedge
    fig = plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap('jet')
    
    for mu_idx in range(nmu):
        kp = kcen * pkmu[:, mu_idx]
        plt.plot(kcen, kp, 'o-', linewidth=2, markersize=4, color=cmap(mu_idx / nmu),
                label=f'μ ∈ [{mu_wedges[mu_idx]:.2f}, {mu_wedges[mu_idx+1]:.2f}]')

        # if pkmu_from_pypower is not None:
        #     print('pkmu_from_pypower / pkmu my own: for mu idx', mu_idx, pkmu_from_pypower[:,mu_idx]/pkmu[:,mu_idx])
        #     kp_pypower = kcen * pkmu_from_pypower[:,mu_idx]
        #     plt.plot(kcen, kp_pypower, '--', linewidth=2, markersize=4, color=cmap(mu_idx / nmu),
        #             label=f'μ ∈ [{mu_wedges[mu_idx]:.2f}, {mu_wedges[mu_idx+1]:.2f}]')
        
    
    plt.ylabel(r'$k P(k,\mu)$ [$({\rm Mpc}/h)^2$]', fontsize=11)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=10, loc=3, ncol=2, facecolor='white', edgecolor='gray')
    plt.grid(alpha=0.3, which='both')
    
    plt.xlabel('k [h/Mpc]', fontsize=11)
    plt.title(
        f'Contamination field P(k,μ) (halfdome lightcone)\n'
        f'spec_type={SYS_SPEC_TYPE}, sys_amp={SYS_AMP}',
        fontsize=12
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'contaminant_pkmu_halfdome.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def parse_args():
    """Parse CLI arguments and update module globals."""
    global RUN_NAME, NMOCK, SAVE_DIR, FIG_DIR, SYS_AMP, SYS_SPEC_TYPE, SYS_AMP_MODE, SYS_AMP_MULT_SCALE
    global SYS_ELL_MIN, SYS_ELL_MAX, SYS_ELL_DELTA, DS_FAC, NMESH, NO_CACHE, CLEAR
    global WITH_RSD, MOCK_TYPE, N_SAMPLE, TARGET_NBAR, PLOT_YSCALE, PLOT_YLIM_MIN, PLOT_YLIM_MAX, PLOT_YSCALE_PS, PLOT_YLIM_PS_MIN, PLOT_YLIM_PS_MAX, VERBOSE_LOGGING
    global K_MIN, K_MAX, DELTA_K, Z_MIN, Z_MAX, MODES_TO_RUN, MEAN_CONSERVING_ADDITIVE, APPLY_SYS_TO_RANDOMS, GAL_LAT_CUT_DEG, USE_JAX, USE_RADIAL_WINDOW
    global COMPUTE_WINDOW, WINDOW_METHOD

    p = argparse.ArgumentParser(
        description='Controlled transverse systematic test on Quijote mocks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--run-name',  required=True,
                   help='Name for this run; used as sub-directory under data/plk/ and figures/.')
    p.add_argument('--spec-type', default='power_law',
                   choices=['flat', 'power_law', 'delta', 'gaia_stellar'],
                   help='Spectral shape of the systematic map. '
                        'power_law/flat/delta: GRF with that spectrum. '
                        'gaia_stellar: real Gaia stellar density map normalized to sys_amp (rms or mean).')
    p.add_argument('--ell-delta', type=int, default=None,
                   help='Annular scale (in grid modes) when --spec-type=delta.')
    p.add_argument('--sys-amp',   type=float, default=0.05,
                   help='Amplitude of the systematic map. For gaia_stellar, interpretation depends on --sys-amp-mode.')
    p.add_argument('--sys-amp-mode', default='mean', choices=['rms', 'mean'],
                   help='For gaia_stellar: "rms"=scale to RMS amplitude (default), "mean"=scale to mean value (contamination fraction).')
    p.add_argument('--sys-amp-mult-scale', type=float, default=1.0,
                   help='Scaling factor for multiplicative contamination effect. >1 makes multiplicative effect stronger at fixed sys_amp.')
    p.add_argument('--nmock',     type=int,   default=5,
                   help='Number of Quijote mocks to average over.')
    p.add_argument('--ds-fac',    type=int,   default=20,
                   help='Downsampling factor (larger = faster, fewer galaxies).')
    p.add_argument('--nmesh',     type=int,   default=128,
                   help='FFT mesh size for power-spectrum estimation.')
    p.add_argument('--k-min',     type=float, default=0.006,
                   help='Minimum k for power spectrum binning [h/Mpc].')
    p.add_argument('--k-max',     type=float, default=0.2,
                   help='Maximum k for power spectrum binning [h/Mpc].')
    p.add_argument('--delta-k',   type=float, default=0.01,
                   help='k bin width for power spectrum binning [h/Mpc].')
    p.add_argument('--z-min',     type=float, default=0.4,
                   help='Minimum redshift for halfdome mock (ignored for quijote).')
    p.add_argument('--z-max',     type=float, default=1.0,
                   help='Maximum redshift for halfdome mock (ignored for quijote).')
    p.add_argument('--no-cache',  action='store_true',
                   help='Force recompute even if a cached npz already exists.')
    p.add_argument('--clear',  action='store_true',
                   help='Remove all cached results for this run before starting.')
    p.add_argument('--plot-only', action='store_true',
                   help='skips some calculations in plot-only mode.')
    p.add_argument('--with-rsd', action='store_true',
                   help='Use RSD galaxy positions (z-axis LOS). Default is real-space positions.')
    p.add_argument('--mock-type', default='quijote', choices=['quijote', 'halfdome'],
                   help='Mock type to use. quijote=periodic box, halfdome=lightcone.')
    p.add_argument('--n-sample', type=int, default=2_000_000,
                   help='Number of galaxies to load per halfdome mock (ignored for quijote).')
    p.add_argument('--plot-yscale', default='log', choices=['log', 'linear'],

                   help='Y-axis scale for ratio plots.')
    p.add_argument('--plot-ylim-min', type=float, default=None,
                   help='Fixed lower y-limit for ratio plots (None = adaptive).')
    p.add_argument('--plot-ylim-max', type=float, default=None,
                   help='Fixed upper y-limit for ratio plots (None = adaptive).')

    p.add_argument('--plot-ylim-ps-min', type=float, default=None,
                   help='Fixed lower y-limit for power spectrum plots (None = adaptive).')
    p.add_argument('--plot-ylim-ps-max', type=float, default=None,
                   help='Fixed upper y-limit for power spectrum plots (None = adaptive).')
    
    p.add_argument('--plot-yscale-ps', default='log', choices=['log', 'linear'],
                   help='Y-axis scale for power spectrum plots.')
    p.add_argument('--verbose-logging', action='store_true',
                   help='Enable verbose logging from pypower (useful for debugging power spectrum computation).')
    p.add_argument('--target-nbar', type=float, default=None,
                   help='Target comoving number density in (h/Mpc)^3 for downsampling halfdome mocks '
                        '(e.g. 1e-4). Overrides n_sample-based density. None = no nbar downsampling.')
    p.add_argument('--modes', nargs='+',
                   choices=['none', 'transverse_additive', 'transverse_multiplicative'],
                   default=None,
                   help='Which contamination modes to run. Defaults to all three. '
                        'Example: --modes none transverse_multiplicative')
    p.add_argument('--legacy-additive', action='store_true',
                   help='Use legacy one-sided positive-lobe-only additive injection '
                        '(biases n_gal by +sys_amp and distorts angular C_l). '
                        'Default is mean-conserving injection.')
    p.add_argument('--apply-sys-to-randoms', action='store_true',
                   help='Phase C9 debug: also apply the multiplicative w_sys to randoms. '
                        'Unbiased estimator must yield ratio=1 at all (k, mu).')
    p.add_argument('--gal-lat-cut', type=float, default=0.0, metavar='DEG',
                   help='Remove galaxies with |b| < DEG from data, randoms, and contamination map. '
                        '0 = no cut (default). Recommended: 20 for Gaia stellar template.')
    p.add_argument('--test-poles-to-wedges', action='store_true',
                   help='Run diagnostic test comparing _poles_to_wedges() with result.wedges.get_power() '
                        'for periodic box geometry. Exits after test completes.')
    p.add_argument('--diagnose-matrix', action='store_true',
                   help='Analyze the poles→wedges transformation matrix and its numerical properties. '
                        'Exits after diagnosis completes.')
    p.add_argument('--use-jax', action='store_true',
                   help='Use jax-power instead of pypower for power spectrum estimation.')
    p.add_argument('--no-radial-window', action='store_true',
                   help='Disable window-corrected mu bins for halfdome. '
                        'Default: True for halfdome (False for quijote). '
                        'NOTE: Ignored for quijote mock_type and for delta-function systematics '
                        '(which always use radial window corrections to match analytical binning).')
    p.add_argument('--compute-window', action='store_true')
    p.add_argument('--window-method', default='fft', choices=['fft', 'smooth'])
    
    args = p.parse_args()

    if args.spec_type == 'delta' and args.ell_delta is None:
        p.error('--ell-delta is required when --spec-type=delta')

    # Update module globals
    RUN_NAME           = args.run_name
    NMOCK              = args.nmock
    SYS_AMP            = args.sys_amp
    SYS_SPEC_TYPE      = args.spec_type
    SYS_AMP_MODE       = args.sys_amp_mode
    SYS_AMP_MULT_SCALE = args.sys_amp_mult_scale
    SYS_ELL_DELTA      = args.ell_delta
    DS_FAC        = args.ds_fac
    NMESH         = args.nmesh
    K_MIN         = args.k_min
    K_MAX         = args.k_max
    DELTA_K       = args.delta_k
    Z_MIN         = args.z_min
    Z_MAX         = args.z_max
    NO_CACHE      = args.no_cache
    CLEAR         = args.clear
    WITH_RSD      = args.with_rsd
    MOCK_TYPE     = args.mock_type
    N_SAMPLE      = args.n_sample
    TARGET_NBAR   = args.target_nbar
    PLOT_YSCALE   = args.plot_yscale
    PLOT_YLIM_MIN = args.plot_ylim_min
    PLOT_YLIM_MAX = args.plot_ylim_max
    PLOT_YSCALE_PS = args.plot_yscale_ps
    PLOT_YLIM_PS_MIN = args.plot_ylim_ps_min
    PLOT_YLIM_PS_MAX = args.plot_ylim_ps_max
    VERBOSE_LOGGING = args.verbose_logging
    MODES_TO_RUN  = args.modes  # None means all
    MEAN_CONSERVING_ADDITIVE = not args.legacy_additive
    APPLY_SYS_TO_RANDOMS     = args.apply_sys_to_randoms
    GAL_LAT_CUT_DEG          = args.gal_lat_cut
    USE_JAX                  = args.use_jax
    # For delta-function systematics with halfdome, always use radial window corrections
    # (matching the analytical approach in delta_ell_systematic_analysis.py)
    USE_RADIAL_WINDOW = (MOCK_TYPE == 'halfdome') and (SYS_SPEC_TYPE == 'delta' or not args.no_radial_window)
    COMPUTE_WINDOW = args.compute_window
    WINDOW_METHOD = args.window_method
    SAVE_DIR      = os.path.join('data/plk/transverse_sys_test', RUN_NAME)
    FIG_DIR       = os.path.join('figures/transverse_sys_test', RUN_NAME)
    if VERBOSE_LOGGING and not USE_JAX:
        # Only load pypower if we're using verbose logging with pypower
        from pypower import setup_logging
        setup_logging()
    return args


def run_or_load(spec: ExperimentSpec, nmock: int) -> dict:
    """Run experiment grid (or load cached result), with optional NO_CACHE override."""
    label = build_run_label(spec)
    out_path = os.path.join(spec.save_dir, f'{label}.npz')

    if os.path.exists(out_path) and not NO_CACHE:
        print(f'[{label}] Loading cached result from {out_path}')
        dat = np.load(out_path)
        result_dict = {
            'kcen': dat['kcen'],
            'all_pkmu': dat['all_pkmu'],
            'all_plk': dat['all_plk'],
            'mu_wedges': dat['mu_wedges'],
            'ells': np.atleast_1d(dat['ells']),
            'label': label,
        }
        # Include nulled multipoles if available
        if 'all_plk_null_lowest_mu' in dat:
            result_dict['all_plk_null_lowest_mu'] = dat['all_plk_null_lowest_mu']
        return result_dict

    print(f'\n[{label}] Running {nmock} mocks...')
    result = run_experiment_grid(spec, nmock=nmock)
    save_experiment_result(result)
    print(f'[{label}] Saved to {out_path}')
    result_dict = {
        'kcen': result.kcen,
        'all_pkmu': result.all_pkmu,
        'all_plk': result.all_plk,
        'mu_wedges': result.mu_wedges,
        'ells': np.asarray(result.spec.ells),
        'label': result.label,
    }
    if result.all_plk_null_lowest_mu is not None:
        result_dict['all_plk_null_lowest_mu'] = result.all_plk_null_lowest_mu
    return result_dict


def plot_halfdome_celestial_diagnostics(fig_dir: str) -> None:
    """
    Generate celestial coordinate (RA/DEC/r) diagnostics for halfdome catalog.
    
    Loads and visualizes the RA/DEC/r coordinates (position_type='rdd') that are 
    actually used in power spectrum calculations, showing:
    - RA/DEC distributions on the celestial sphere
    - Radial distance distribution
    - Correlations between coordinates
    - Sky coverage pattern
    """
    if MOCK_TYPE != 'halfdome':
        return  # Only for halfdome
    
    os.makedirs(fig_dir, exist_ok=True)
    
    print('\n[Celestial diagnostics] Generating RA/DEC/r visualizations...')
    
    try:
        dm = desi_mock()
        dm.halfdome_mock_basedir = DEFAULT_HALFDOME_BASEDIR
        
        # ── Load clean halfdome catalog for mock 0 ────────────────────────────
        mock_idx = 0
        galpos_clean, redshift_clean = dm.load_halfdome_mock(mock_idx, n_sample=N_SAMPLE, seed=42)
        
        # Use nominal boxsize initially to compute RA/DEC/r
        nominal_boxsize = 1000.0
        ra_clean, dec_clean, r_clean = convert_to_ra_dec_distance(galpos_clean, nominal_boxsize, center_offset_mpc=0.0)
        r_clean_values = np.asarray(r_clean.value if hasattr(r_clean, 'value') else r_clean)
        
        # Use maximum r value as the effective boxsize for reference
        max_r = np.max(r_clean_values)
        print(f'  Clean catalog: {len(ra_clean):,} objects, max distance: {max_r:.1f} Mpc/h')
        
        # ── Generate diagnostic plots ─────────────────────────────────────────
        plot_radec_distribution(ra_clean, dec_clean, fig_dir, '_halfdome_clean', 
                               title_suffix='(halfdome clean catalog)')
        print(f'    Saved RA/DEC distribution')
        
        plot_comoving_distance_distribution(r_clean_values, fig_dir, '_halfdome_clean',
                                           title_suffix='(halfdome clean catalog)')
        print(f'    Saved distance distribution')
        
        plot_radec_distance_correlation(ra_clean, dec_clean, r_clean_values, fig_dir, 
                                        '_halfdome_clean', title_suffix='(halfdome clean catalog)')
        print(f'    Saved RA/DEC/r correlation plot')
        
        print('  ✓ Celestial coordinate diagnostics complete')
        
    except Exception as e:
        print(f'  ✗ Error generating celestial diagnostics: {e}')
        import traceback
        traceback.print_exc()


def main():
    global COMPUTED_MU_WEDGES
    args = parse_args()
    
    # Compute and cache mu_wedges upfront so all plotting functions use the same binning
    COMPUTED_MU_WEDGES = get_mu_wedges()
    print(f'[main] Computed mu_wedges: {COMPUTED_MU_WEDGES}')
    print()
    
    # Run diagnostic tests if requested
    if args.diagnose_matrix:
        print('\nRunning matrix diagnostics...')
        diagnose_poles_to_wedges_matrix()
        print('\nDiagnosis completed.')
        sys.exit(0)
    
    if args.test_poles_to_wedges:
        print('\nRunning diagnostic test: _poles_to_wedges vs result.wedges.get_power()...')
        test_result = test_poles_to_wedges_periodic()
        exit_code = 0 if test_result else 1
        print(f'\nTest completed with exit code {exit_code}.')
        sys.exit(exit_code)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Clear cached results if requested
    if CLEAR:
        import glob
        cached_files = glob.glob(os.path.join(SAVE_DIR, '*.npz'))
        for fpath in cached_files:
            os.remove(fpath)
            print(f'Cleared: {fpath}')
        if cached_files:
            print()

    print(f'Run name  : {RUN_NAME}')
    print(f'mock_type : {MOCK_TYPE}' + (f', n_sample={N_SAMPLE}' if MOCK_TYPE == 'halfdome' else f', ds_fac={DS_FAC}'))
    print(f'spec_type : {SYS_SPEC_TYPE}, sys_amp={SYS_AMP}' + 
          (f' ({SYS_AMP_MODE} mode)' if SYS_SPEC_TYPE == 'gaia_stellar' else ''))
    if SYS_SPEC_TYPE == 'delta':
        print(f'ell_delta : {SYS_ELL_DELTA}')
    print(f'nmock={NMOCK}, ds_fac={DS_FAC}, nmesh={NMESH}')
    print(f'Output dirs: {SAVE_DIR} | {FIG_DIR}')
    print()

    print('Generating contamination field visualization...')
    if MOCK_TYPE == 'quijote':
        plot_sys_map(FIG_DIR)
    elif MOCK_TYPE == 'halfdome':
        plot_sys_map_healpix(FIG_DIR)

    print('Generating angular power spectrum...')
    if MOCK_TYPE == 'quijote':
        plot_angular_power_spectrum_periodic(FIG_DIR)
    elif MOCK_TYPE == 'halfdome':
        plot_angular_power_spectrum_healpix(FIG_DIR)

    print('Generating contamination field P(k,μ) power spectrum...')

    if not args.plot_only:
        if MOCK_TYPE == 'quijote':
            plot_contaminant_pkmu(FIG_DIR)
        elif MOCK_TYPE == 'halfdome':
            plot_contaminant_pkmu_halfdome(FIG_DIR)

    print('Generating galaxy density visualization...')
    if MOCK_TYPE == 'quijote':
        plot_density_slices(FIG_DIR)
    elif MOCK_TYPE == 'halfdome':
        plot_density_distribution_healpix(FIG_DIR)
        plot_mask_and_contamination_diagnostics(FIG_DIR)
        # plot_catalog_positions_with_masking(FIG_DIR)
        # plot_halfdome_celestial_diagnostics(FIG_DIR)

    active_modes = MODES_TO_RUN if MODES_TO_RUN is not None else CONTAMINATION_MODES
    results = {}
    for mode in active_modes:
        spec = build_spec(mode)
        results[mode] = run_or_load(spec, nmock=NMOCK)

    print('\nAll modes computed. Generating plots...')
    if 'none' not in results:
        print('Skipping ratio plots (--modes did not include "none").')
    else:
        plot_pkmu_comparison(results, FIG_DIR)
        plot_power_spectrum_multipoles(results, FIG_DIR)
        plot_null_mu_bin_effect(results, FIG_DIR)
    print(f'\nDone. Figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
