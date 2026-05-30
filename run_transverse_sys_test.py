#!/usr/bin/env python
"""
Controlled transverse systematic test on Quijote periodic box mocks.

Runs three contamination modes — 'none', 'transverse_additive', and
'transverse_multiplicative' — using the same angular power spectrum, then
plots P(k,mu) for each mu bin and the ratio contaminated/clean.

Usage (must be run with the cosmodesi Python environment):
    /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20250526-1.0.0/conda/bin/python \
        run_transverse_sys_test.py

Optional arguments (edit the CONFIG block below).
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display on login nodes
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

# ── make sure the repo root is on the path ──────────────────────────────────
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from pipeline import (
    ExperimentSpec,
    build_run_label,
    run_experiment_grid,
    save_experiment_result,
    DEFAULT_QUIJOTE_BASEDIR,
)
from contamination import gen_controlled_transverse_map
from desi_mocks import desi_mock

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here
# ─────────────────────────────────────────────────────────────────────────────
NMOCK        = 5            # number of Quijote mocks to average over
SAVE_DIR     = 'data/plk/transverse_sys_test'
FIG_DIR      = 'figures/transverse_sys_test'
SYS_AMP      = 0.10        # rms amplitude of systematic map (10%)
SYS_SPEC_TYPE = 'power_law' # 'flat', 'power_law', or 'delta'
SYS_ELL_MIN  = 6           # fundamental mode of Quijote box at z~0.5
SYS_ELL_MAX  = 16          # max ell for systematic map (also drives mu binning)
SYS_ELL_DELTA = None        # only used if SYS_SPEC_TYPE='delta'
DS_FAC       = 20           # downsampling factor (higher = faster; increase for quick tests)
NMESH        = 128          # FFT mesh size
ELLS         = (0, 2, 4, 16)  # multipoles; max(ELLS)=16 drives nonuniform mu bins
N_CLEAN_BINS = 6            # number of clean mu bins for nonuniform binning
NGRID_SYS    = 256          # resolution of 2D contamination map
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


def build_spec(contamination_mode: str) -> ExperimentSpec:
    return ExperimentSpec(
        mock_type='quijote',
        quijote_geometry='full_cube',
        with_rsd=False,
        contamination_mode=contamination_mode,
        redshift_sel=False,       # periodic box — no true redshifts
        ds_fac=DS_FAC,
        nmesh=NMESH,
        ells=ELLS,
        n_clean_bins=N_CLEAN_BINS,
        mu_binning_strategy='nonuniform',
        sys_amp=SYS_AMP,
        sys_spec_type=SYS_SPEC_TYPE,
        sys_ell_min=SYS_ELL_MIN,
        sys_ell_max=SYS_ELL_MAX,
        sys_ell_delta=SYS_ELL_DELTA,
        save_dir=SAVE_DIR,
    )


def run_or_load(spec: ExperimentSpec, nmock: int) -> dict:
    """Run experiment grid, save, and return {'kcen', 'all_pkmu', 'mu_wedges'}."""
    label = build_run_label(spec)
    out_path = os.path.join(spec.save_dir, f'{label}.npz')

    if os.path.exists(out_path):
        print(f'[{label}] Loading cached result from {out_path}')
        dat = np.load(out_path)
        return {'kcen': dat['kcen'], 'all_pkmu': dat['all_pkmu'], 'mu_wedges': dat['mu_wedges'], 'label': label}

    print(f'\n[{label}] Running {nmock} mocks...')
    result = run_experiment_grid(spec, nmock=nmock)
    save_experiment_result(result)
    print(f'[{label}] Saved to {out_path}')
    return {
        'kcen': result.kcen,
        'all_pkmu': result.all_pkmu,
        'mu_wedges': result.mu_wedges,
        'label': label,
    }


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

    # ── Figure 1: kP(k,mu) per mode ─────────────────────────────────────────
    fig, axes = plt.subplots(1, len(CONTAMINATION_MODES), figsize=(5 * len(CONTAMINATION_MODES), 4.5),
                             sharey=True)
    for ax, mode in zip(axes, CONTAMINATION_MODES):
        res = results[mode]
        pkmu = res['all_pkmu'].real  # shape (nmock, nk, nmu)
        ax.set_title(MODE_LABELS[mode], fontsize=12)
        for mu_idx in range(nmu):
            mu_lab = (f'{mu_wedges[mu_idx]:.2f}' + r'$<\mu<$' +
                      f'{mu_wedges[mu_idx+1]:.2f}')
            mean_pk = np.mean(pkmu[:, :, mu_idx], axis=0)
            err_pk  = np.std(pkmu[:, :, mu_idx], axis=0) / np.sqrt(pkmu.shape[0])
            ax.errorbar(kcen, kcen * mean_pk, yerr=kcen * err_pk,
                        label=mu_lab, color=mu_colors[mu_idx],
                        linewidth=2 if mu_idx == 0 else 1.2)
        ax.set_xlabel('k [h/Mpc]', fontsize=11)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r'$k\,P(k,\mu)$ [$({\rm Mpc}/h)^2$]', fontsize=11)
    axes[0].legend(ncol=2, fontsize=8, loc='upper right')
    plt.subplots_adjust(wspace=0)
    fpath = os.path.join(fig_dir, 'pkmu_comparison.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')

    # ── Figure 2: ratio P_sys / P_clean per mu bin, one panel per mode ──────
    sys_modes = [m for m in CONTAMINATION_MODES if m != 'none']
    fig, axes = plt.subplots(1, len(sys_modes), figsize=(5.5 * len(sys_modes), 4.5),
                             sharey=True)
    if len(sys_modes) == 1:
        axes = [axes]

    pkmu_clean = clean_res['all_pkmu'].real
    for ax, mode in zip(axes, sys_modes):
        res = results[mode]
        pkmu_sys = res['all_pkmu'].real
        ax.set_title(f'{MODE_LABELS[mode]}\nvs. clean', fontsize=12)
        ax.axhline(1.0, color='k', lw=1, ls='--')
        for mu_idx in range(nmu):
            mu_lab = (f'{mu_wedges[mu_idx]:.2f}' + r'$<\mu<$' +
                      f'{mu_wedges[mu_idx+1]:.2f}')
            ratio = pkmu_sys[:, :, mu_idx] / np.where(
                np.abs(pkmu_clean[:, :, mu_idx]) > 0,
                pkmu_clean[:, :, mu_idx],
                np.nan,
            )
            mean_r = np.nanmean(ratio, axis=0)
            err_r  = np.nanstd(ratio, axis=0) / np.sqrt(ratio.shape[0])
            ax.errorbar(kcen, mean_r, yerr=err_r,
                        label=mu_lab, color=mu_colors[mu_idx],
                        linewidth=2 if mu_idx == 0 else 1.2,
                        linestyle=MODE_LS[mode])
        ax.set_xlabel('k [h/Mpc]', fontsize=11)
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        ax.set_ylim(0.7, 1.5)
    axes[0].set_ylabel(r'$P(k,\mu)^{\rm sys}\,/\,P(k,\mu)^{\rm clean}$', fontsize=12)
    axes[0].legend(ncol=2, fontsize=8, loc='upper right')
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
        ax.axhline(1.0, color='k', lw=1, ls='--')
        for mode in sys_modes:
            pkmu_sys = results[mode]['all_pkmu'].real
            ratio = pkmu_sys[:, :, mu_idx] / np.where(
                np.abs(pkmu_clean[:, :, mu_idx]) > 0,
                pkmu_clean[:, :, mu_idx],
                np.nan,
            )
            mean_r = np.nanmean(ratio, axis=0)
            err_r  = np.nanstd(ratio, axis=0) / np.sqrt(ratio.shape[0])
            ax.errorbar(kcen, mean_r, yerr=err_r,
                        label=MODE_LABELS[mode], color=MODE_COLORS[mode],
                        linewidth=1.8, linestyle=MODE_LS[mode])
        ax.set_xscale('log')
        ax.grid(alpha=0.25)
        ax.set_ylim(0.7, 1.5)
    for ax in axes_flat[nmu:]:
        ax.set_visible(False)
    axes_flat[0].legend(fontsize=8)
    fig.supxlabel('k [h/Mpc]', fontsize=11)
    fig.supylabel(r'$P(k,\mu)^{\rm sys}\,/\,P(k,\mu)^{\rm clean}$', fontsize=11)
    title_parts = [f'Quijote periodic box  |  {NMOCK} mocks  |  sys_amp={SYS_AMP}',
                   f'spec_type={SYS_SPEC_TYPE}, ell_min={SYS_ELL_MIN}, ell_max={SYS_ELL_MAX}']
    fig.suptitle('\n'.join(title_parts), fontsize=10)
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'pkmu_ratio_per_mu.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_sys_map(fig_dir: str) -> None:
    """Plot the 2D transverse contamination field in the periodic box."""
    os.makedirs(fig_dir, exist_ok=True)
    boxsize = 1000.0

    sys_map = gen_controlled_transverse_map(
        amp=SYS_AMP,
        seed=12345,
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
    ax.set_title(
        f'Transverse contamination field\n'
        f'spec_type={SYS_SPEC_TYPE}, '
        f'$\\ell\\in[{SYS_ELL_MIN},{SYS_ELL_MAX}]$, amp={SYS_AMP}',
        fontsize=11,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'sys_map_2d.png')
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
        f'Galaxy density maps (mock #{mock_idx}, DS_FAC={DS_FAC})\n'
        f'spec_type={SYS_SPEC_TYPE}, amp={SYS_AMP}, '
        f'$\\ell\\in[{SYS_ELL_MIN},{SYS_ELL_MAX}]$',
        fontsize=11,
    )
    plt.tight_layout()
    fpath = os.path.join(fig_dir, 'density_slices.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print('Generating contamination field visualization...')
    plot_sys_map(FIG_DIR)

    print('Generating galaxy density slice visualization...')
    plot_density_slices(FIG_DIR)

    results = {}
    for mode in CONTAMINATION_MODES:
        spec = build_spec(mode)
        results[mode] = run_or_load(spec, nmock=NMOCK)

    print('\nAll modes computed. Generating plots...')
    plot_pkmu_comparison(results, FIG_DIR)
    print(f'\nDone. Figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
