from __future__ import annotations

import json
import os
from itertools import product
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import healpy as hp
from astropy.cosmology import Planck18 as cosmo

from nonunif_binning import compute_null_bins

from contamination import (
    gen_controlled_transverse_map,
    gen_delta_ebv_map_uncorr,
    gen_dn_n_map,
    generate_poisson_radec_from_map,
    modify_fkp_weights,
    poisson_star_map_from_fraction,
)
from desi_mocks import desi_mock
from mask_and_randoms import generate_uniform_randoms
from star_sim import load_gaia_stellar_density, simple_halo_thick_disk_stellar_density
from utils import (
    comoving_distance_to_redshift,
    convert_to_ra_dec_distance,
    gen_interp_fn_dcom_z,
    grab_chi_interp,
)


DEFAULT_QUIJOTE_BASEDIR = '/global/cfs/cdirs/desi/users/rmfeder/quijote/fiducial/systematics/'
DEFAULT_HALFDOME_BASEDIR = '/global/cfs/cdirs/cmb/gsharing/halfdome/full_res/halos/'


@dataclass(frozen=True)
class ExperimentSpec:
    mock_type: str = 'halfdome'
    with_rsd: bool = False
    contamination_mode: str = 'none'
    frac_stellar_contam: float = 0.1
    use_gaia: bool = False
    sfd_std: float = 0.01
    dust_alpha: float = -10.0
    redshift_sel: bool = True
    zmin: float = 0.4
    zmax: float = 1.0
    replicate: bool = False
    rep_fac: int = 1
    ds_fac: int = 5
    randomize: bool = False
    n_sample: int = 20_000_000
    k_min: float = 0.002
    k_max: float = 0.2
    delta_k: float = 0.01
    mu_min: float = 0.0
    mu_max: float = 1.0
    nwedge: int = 6
    mu_wedges: tuple[float, ...] | None = None
    mu_binning_strategy: str = 'nonuniform'
    n_clean_bins: int = 8
    ells: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16)
    nmesh: int = 512
    boxsize: float = 1000.0
    n_random_factor: int = 5
    seed: int = 42
    save_dir: str = 'data/plk'
    output_name: str | None = None
    plot: bool = False
    nplot: int = 0
    quijote_basedir: str = DEFAULT_QUIJOTE_BASEDIR
    halfdome_basedir: str = DEFAULT_HALFDOME_BASEDIR
    quijote_geometry: str = 'replicated'
    # Controlled transverse systematic parameters
    sys_amp: float = 0.01
    sys_spec_type: str = 'power_law'
    sys_ell_min: int = 6
    sys_ell_max: int = 64
    sys_ell_delta: int | None = None


@dataclass
class PreparedCatalog:
    positions_rdd: np.ndarray
    weights: np.ndarray
    base_redshifts: np.ndarray | None
    base_r: np.ndarray
    mock_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    kcen: np.ndarray
    all_pkmu: np.ndarray
    all_plk: np.ndarray
    mu_wedges: np.ndarray
    output_path: str | None = None
    label: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)


def _slug_number(value: float | int) -> str:
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


def _spec_to_jsonable(spec: ExperimentSpec) -> dict[str, Any]:
    data = asdict(spec)
    if data['mu_wedges'] is not None:
        data['mu_wedges'] = list(data['mu_wedges'])
    return data


def build_kedges(spec: ExperimentSpec) -> np.ndarray:
    return np.arange(spec.k_min, spec.k_max + spec.delta_k, spec.delta_k)


def build_mu_wedges(spec: ExperimentSpec) -> np.ndarray:
    if spec.mu_wedges is not None:
        return np.asarray(spec.mu_wedges, dtype=float)

    strategy = spec.mu_binning_strategy.lower()
    if strategy == 'nonuniform':
        ell_max = max(spec.ells)
        return np.asarray(
            compute_null_bins(ell_max=ell_max, n_clean_bins=spec.n_clean_bins),
            dtype=float,
        )

    if strategy == 'uniform':
        return np.linspace(spec.mu_min, spec.mu_max, spec.nwedge)

    raise ValueError(f'Unknown mu_binning_strategy: {spec.mu_binning_strategy}')


def build_run_label(spec: ExperimentSpec) -> str:
    parts = [spec.mock_type]
    if spec.mock_type == 'quijote':
        parts.append(spec.quijote_geometry)
    parts.append('rsd' if spec.with_rsd else 'noRSD')
    parts.append(spec.contamination_mode)
    if spec.contamination_mode in {'stellar', 'both'}:
        parts.append(f'star{_slug_number(spec.frac_stellar_contam)}')
        parts.append('gaia' if spec.use_gaia else 'notional')
    if spec.contamination_mode in {'dust', 'both'}:
        parts.append(f'dust{_slug_number(spec.sfd_std)}')
        parts.append(f'a{_slug_number(spec.dust_alpha)}')
    if spec.contamination_mode in {'transverse_additive', 'transverse_multiplicative'}:
        parts.append(spec.sys_spec_type)
        parts.append(f'amp{_slug_number(spec.sys_amp)}')
        parts.append(f'lmin{spec.sys_ell_min}')
        if spec.sys_spec_type == 'delta':
            parts.append(f'ldelta{spec.sys_ell_delta}')
        else:
            parts.append(f'lmax{spec.sys_ell_max}')
    if spec.redshift_sel:
        parts.append(f'z{_slug_number(spec.zmin)}-{_slug_number(spec.zmax)}')
    if spec.replicate:
        parts.append(f'rep{spec.rep_fac}')
    if spec.ds_fac != 1:
        parts.append(f'ds{spec.ds_fac}')
    if spec.mu_wedges is not None:
        parts.append('manualmu')
    elif spec.mu_binning_strategy.lower() == 'nonuniform':
        parts.append(f'nonunifmu{spec.n_clean_bins}')
    else:
        parts.append('unifmu')

    parts.append(f'lmax{max(spec.ells)}')
    return '_'.join(parts)


def recommended_base_kwargs(mock_type: str = 'halfdome') -> dict[str, Any]:
    """Return recommended base kwargs for the current pipeline defaults."""
    mock_type_norm = mock_type.lower()
    if mock_type_norm not in {'halfdome', 'quijote'}:
        raise ValueError(
            f"Unsupported mock_type='{mock_type}'. Supported options are 'halfdome' and 'quijote'."
        )

    return {
        'mock_type': mock_type_norm,
        'with_rsd': False,
        'contamination_mode': 'none',
        'frac_stellar_contam': 0.1,
        'use_gaia': False,
        'sfd_std': 0.01,
        'dust_alpha': -10.0,
        'redshift_sel': True,
        'zmin': 0.4,
        'zmax': 1.0,
        'replicate': False,
        'rep_fac': 1,
        'ds_fac': 5,
        'randomize': False,
        'n_sample': 20_000_000,
        'k_min': 0.002,
        'k_max': 0.2,
        'delta_k': 0.01,
        'mu_binning_strategy': 'nonuniform',
        'n_clean_bins': 8,
        'ells': (0, 2, 4, 6, 8, 10, 12, 14, 16),
        'nmesh': 512,
        'seed': 42,
        'quijote_geometry': 'full_cube' if mock_type_norm == 'quijote' else 'replicated',
    }


def append_run_ledger(save_dir: str, entry: dict[str, Any]) -> str:
    ledger_path = os.path.join(save_dir, 'run_ledger.jsonl')
    os.makedirs(save_dir, exist_ok=True)
    with open(ledger_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, sort_keys=True) + '\n')
    return ledger_path


def _stage_seeds(spec: ExperimentSpec, mock_idx: int) -> dict[str, int]:
    base = int(spec.seed) + mock_idx * 10_000
    return {
        'quijote': base + 1,
        'dust': base + 2,
        'stellar_map': base + 3,
        'stellar_radec': base + 4,
        'stellar_redshift': base + 5,
        'randoms': base + 6,
    }


def _prepare_quijote_catalog(spec: ExperimentSpec, mock_idx: int, dm: desi_mock) -> PreparedCatalog:
    if not spec.quijote_basedir.endswith('/'):
        quijote_basedir = spec.quijote_basedir + '/'
    else:
        quijote_basedir = spec.quijote_basedir

    dm.quijote_mock_basedir = quijote_basedir

    galpos = dm.load_quijote_galpos(
        mock_idx,
        with_RSD=spec.with_rsd,
        replicate=spec.replicate,
        rep_fac=spec.rep_fac,
        ds_fac=spec.ds_fac,
        randomize=spec.randomize,
        seed=_stage_seeds(spec, mock_idx)['quijote'],
    )

    boxsize_use = spec.boxsize * spec.rep_fac if spec.replicate else spec.boxsize
    center_offset_mpc = None
    ra, dec, r = convert_to_ra_dec_distance(galpos, boxsize_use, center_offset_mpc=center_offset_mpc)
    r_values = np.asarray(r.value if hasattr(r, 'value') else r)
    positions_rdd = np.vstack([ra, dec, r_values])
    weights = np.ones_like(ra, dtype=float)

    base_redshifts = None
    if spec.redshift_sel:
        dcom_to_z_interp = gen_interp_fn_dcom_z()
        base_redshifts = comoving_distance_to_redshift(r_values / cosmo.h, dcom_to_z_interp)

    return PreparedCatalog(
        positions_rdd=positions_rdd,
        weights=weights,
        base_redshifts=base_redshifts,
        base_r=r_values,
        mock_idx=mock_idx,
        metadata={'boxsize_use': boxsize_use, 'quijote_basedir': quijote_basedir},
    )


def _prepare_halfdome_catalog(spec: ExperimentSpec, mock_idx: int, dm: desi_mock) -> PreparedCatalog:
    if not spec.halfdome_basedir.endswith('/'):
        halfdome_basedir = spec.halfdome_basedir + '/'
    else:
        halfdome_basedir = spec.halfdome_basedir

    dm.halfdome_mock_basedir = halfdome_basedir
    galpos, redshift = dm.load_halfdome_mock(mock_idx, n_sample=spec.n_sample)

    ra, dec, r = convert_to_ra_dec_distance(galpos, spec.boxsize, center_offset_mpc=0.0)
    r_values = np.asarray(r.value if hasattr(r, 'value') else r)
    positions_rdd = np.vstack([ra, dec, r_values])
    weights = np.ones_like(ra, dtype=float)

    return PreparedCatalog(
        positions_rdd=positions_rdd,
        weights=weights,
        base_redshifts=np.asarray(redshift),
        base_r=r_values,
        mock_idx=mock_idx,
        metadata={'boxsize_use': spec.boxsize, 'halfdome_basedir': halfdome_basedir},
    )


def _apply_dust_systematic(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    seeds = _stage_seeds(spec, mock_idx)
    from dust import gen_sfd_hp

    _, cl_sfd = gen_sfd_hp()
    delta_ebv = gen_delta_ebv_map_uncorr(cl_sfd, std=spec.sfd_std, seed=seeds['dust'])
    delta_n_over_n = gen_dn_n_map(grf_map=delta_ebv, alpha=spec.dust_alpha)
    weights, sys_weights = modify_fkp_weights(
        catalog.positions_rdd[0],
        catalog.positions_rdd[1],
        catalog.weights,
        delta_n_over_n,
    )
    catalog.weights = weights
    catalog.metadata['dust_sys_weights'] = sys_weights
    return catalog


def _apply_stellar_systematic(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    seeds = _stage_seeds(spec, mock_idx)
    n_gal = catalog.positions_rdd.shape[1]

    if spec.use_gaia:
        stellar_map = load_gaia_stellar_density(plot=False)
        star_counts = poisson_star_map_from_fraction(
            stellar_map,
            n_gal,
            frac=spec.frac_stellar_contam,
            mask=None,
            seed=seeds['stellar_map'],
        )
        ra_star, dec_star = generate_poisson_radec_from_map(star_counts, seed=seeds['stellar_radec'])
    else:
        ra_star, dec_star, _ = simple_halo_thick_disk_stellar_density(
            N_gal=n_gal,
            frac=spec.frac_stellar_contam,
            seed=seeds['stellar_map'],
        )

    if len(ra_star) == 0:
        catalog.metadata['stellar_added'] = 0
        return catalog

    rng = np.random.default_rng(seeds['stellar_redshift'])
    rand_indices = rng.choice(len(catalog.base_r), size=len(ra_star), replace=True)
    r_star = catalog.base_r[rand_indices]

    star_positions = np.vstack([ra_star, dec_star, r_star])
    star_weights = np.ones_like(ra_star, dtype=float)
    catalog.positions_rdd = np.concatenate([catalog.positions_rdd, star_positions], axis=1)
    catalog.weights = np.concatenate([catalog.weights, star_weights])
    catalog.metadata['stellar_added'] = len(ra_star)
    return catalog


def _apply_transverse_additive_sys(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    seeds = _stage_seeds(spec, mock_idx)
    periodic = (spec.mock_type == 'quijote')
    boxsize_use = catalog.metadata.get('boxsize_use', spec.boxsize)

    sys_map = gen_controlled_transverse_map(
        amp=spec.sys_amp,
        nside=256,
        seed=seeds['dust'],
        spec_type=spec.sys_spec_type,
        ell_max=spec.sys_ell_max,
        ell_min=spec.sys_ell_min,
        ell_delta=spec.sys_ell_delta,
        periodic=periodic,
        boxsize=boxsize_use,
    )

    n_gal = catalog.positions_rdd.shape[1]
    rng = np.random.default_rng(seeds['stellar_map'])

    if not periodic:
        # HEALPix path: Poisson-sample from positive lobe of full-sky map
        nside_map = hp.npix2nside(len(sys_map))
        positive_map = np.clip(sys_map, 0.0, None)
        total_weight = positive_map.sum()
        if total_weight > 0:
            n_inject = max(1, int(spec.sys_amp * n_gal))
            expected_counts = positive_map / total_weight * n_inject
            ra_inject, dec_inject = generate_poisson_radec_from_map(
                expected_counts, seed=seeds['stellar_radec']
            )
        else:
            ra_inject, dec_inject = np.array([]), np.array([])
    else:
        # Periodic 2D FFT path: sample positions uniformly in transverse plane,
        # then accept/reject with probability proportional to positive part of map.
        # This preserves periodicity: the injection density is periodic in (x,y).
        ngrid = sys_map.shape[0]
        positive_map = np.clip(sys_map, 0.0, None)
        max_val = positive_map.max()
        if max_val <= 0:
            ra_inject, dec_inject = np.array([]), np.array([])
        else:
            n_inject = max(1, int(spec.sys_amp * n_gal))
            # Rejection-sample uniform (x,y) with envelope = max_val
            accepted_xy = []
            batch = n_inject * 4
            while len(accepted_xy) < n_inject:
                xs = rng.uniform(0.0, boxsize_use, size=batch)
                ys = rng.uniform(0.0, boxsize_use, size=batch)
                ix = np.floor(xs / boxsize_use * ngrid).astype(int) % ngrid
                iy = np.floor(ys / boxsize_use * ngrid).astype(int) % ngrid
                vals = positive_map[ix, iy]
                u = rng.uniform(0.0, max_val, size=batch)
                keep = u < vals
                accepted_xy.extend(zip(xs[keep], ys[keep]))
            accepted_xy = accepted_xy[:n_inject]
            xs_inj = np.array([p[0] for p in accepted_xy])
            ys_inj = np.array([p[1] for p in accepted_xy])
            # Convert (x, y) in Cartesian box coords to (RA, Dec)
            # The box z-axis is the LOS; x and y are transverse.
            # Place injected galaxies at box center in z, sample r from catalog.
            z_inj = np.full(n_inject, boxsize_use / 2.0)
            galpos_inj = np.stack([xs_inj, ys_inj, z_inj], axis=1)
            ra_inject, dec_inject, _ = convert_to_ra_dec_distance(galpos_inj, boxsize_use)

    if len(ra_inject) == 0:
        catalog.metadata['transverse_additive_injected'] = 0
        return catalog

    rng2 = np.random.default_rng(seeds['stellar_redshift'])
    rand_idx = rng2.choice(len(catalog.base_r), size=len(ra_inject), replace=True)
    r_inject = catalog.base_r[rand_idx]

    inject_positions = np.vstack([ra_inject, dec_inject, r_inject])
    inject_weights = np.ones(len(ra_inject), dtype=float)
    catalog.positions_rdd = np.concatenate([catalog.positions_rdd, inject_positions], axis=1)
    catalog.weights = np.concatenate([catalog.weights, inject_weights])
    catalog.metadata['transverse_additive_injected'] = len(ra_inject)
    return catalog


def _apply_transverse_multiplicative_sys(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    seeds = _stage_seeds(spec, mock_idx)
    periodic = (spec.mock_type == 'quijote')
    boxsize_use = catalog.metadata.get('boxsize_use', spec.boxsize)

    sys_map = gen_controlled_transverse_map(
        amp=spec.sys_amp,
        nside=256,
        seed=seeds['dust'],
        spec_type=spec.sys_spec_type,
        ell_max=spec.sys_ell_max,
        ell_min=spec.sys_ell_min,
        ell_delta=spec.sys_ell_delta,
        periodic=periodic,
        boxsize=boxsize_use,
    )

    ra = catalog.positions_rdd[0]
    dec = catalog.positions_rdd[1]

    if not periodic:
        # HEALPix map: use existing modify_fkp_weights (samples by RA/Dec)
        nside_map = hp.npix2nside(len(sys_map))
        weights, sys_weights = modify_fkp_weights(ra, dec, catalog.weights, sys_map, nside=nside_map)
    else:
        # 2D periodic map: sample by Cartesian (x, y) transverse position.
        # catalog positions are already in (RA, Dec, r) form; we recover transverse
        # Cartesian coords by reversing the box projection used during catalog prep.
        ngrid = sys_map.shape[0]
        # Convert RA/Dec back to Cartesian box transverse coords
        # RA in [0,360) maps to x in [0, boxsize), Dec in [~-45,~45] maps to y.
        # The exact mapping depends on convert_to_ra_dec_distance; we reproduce it:
        #   ra  = atan2(y - L/2, x - L/2) in degrees (plus offset)
        #   dec = asin(z_normalized) ... but here z is LOS, not the third Cartesian axis.
        # Simplest robust approach: treat RA/Dec as angular coords of a 2D map
        # evaluated in the periodic box frame by wrapping.
        # Map RA [0,360) -> x index, Dec angle -> y index via linear wrap.
        x_frac = (ra % 360.0) / 360.0
        # Dec ranges approximately over the angular size of the box; wrap to [0,1)
        dec_range = np.degrees(np.arctan(boxsize_use / 2.0 / (boxsize_use / 2.0)))  # ~45 deg
        y_frac = ((dec + dec_range) / (2 * dec_range)) % 1.0
        ix = np.floor(x_frac * ngrid).astype(int) % ngrid
        iy = np.floor(y_frac * ngrid).astype(int) % ngrid
        dn_over_n = sys_map[ix, iy]
        w_sys = np.clip(1.0 + dn_over_n, 0.01, 10.0)
        weights = catalog.weights * w_sys
        sys_weights = w_sys

    catalog.weights = weights
    catalog.metadata['transverse_multiplicative_sys_weights'] = sys_weights
    return catalog


def _apply_contamination(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    valid_modes = {'none', 'stellar', 'dust', 'both', 'transverse_additive', 'transverse_multiplicative'}
    if spec.contamination_mode not in valid_modes:
        raise ValueError(f'Unknown contamination_mode: {spec.contamination_mode}')

    if spec.contamination_mode in {'dust', 'both'}:
        catalog = _apply_dust_systematic(spec, catalog, mock_idx)

    if spec.contamination_mode in {'stellar', 'both'}:
        catalog = _apply_stellar_systematic(spec, catalog, mock_idx)

    if spec.contamination_mode == 'transverse_additive':
        catalog = _apply_transverse_additive_sys(spec, catalog, mock_idx)

    if spec.contamination_mode == 'transverse_multiplicative':
        catalog = _apply_transverse_multiplicative_sys(spec, catalog, mock_idx)

    return catalog


def _build_random_catalog(spec: ExperimentSpec, catalog: PreparedCatalog) -> tuple[np.ndarray, np.ndarray]:
    seeds = _stage_seeds(spec, catalog.mock_idx)
    n_randoms = int(spec.n_random_factor * catalog.positions_rdd.shape[1])
    boxsize_use = catalog.metadata['boxsize_use']

    if spec.redshift_sel:
        chi_interp = grab_chi_interp()
        redshift_source = catalog.base_redshifts
        if redshift_source is None:
            raise ValueError('Redshift selection requested but no base redshifts were prepared.')
        if len(redshift_source) == 0:
            print('Warning: redshift-selected data is empty; falling back to uniform random z sampling in [zmin, zmax].')
            redshift_source = None
        ra_rand, dec_rand, r_rand, _ = generate_uniform_randoms(
            chi_interp,
            n_randoms,
            zmin=spec.zmin,
            zmax=spec.zmax,
            data_z=redshift_source,
            seed=seeds['randoms'],
        )
        rand_positions = np.array([ra_rand, dec_rand, r_rand], dtype=float)
        rand_weights = np.ones_like(ra_rand, dtype=float)
        return rand_positions, rand_weights

    rng = np.random.default_rng(seeds['randoms'])
    randoms_positions = rng.uniform(0.0, boxsize_use, size=(n_randoms, 3))
    ra_rand, dec_rand, r_rand = convert_to_ra_dec_distance(randoms_positions, boxsize_use)
    r_values = np.asarray(r_rand.value if hasattr(r_rand, 'value') else r_rand)
    rand_positions = np.array([ra_rand, dec_rand, r_values], dtype=float)
    rand_weights = np.ones_like(ra_rand, dtype=float)
    return rand_positions, rand_weights


def _compute_power_spectra(
    spec: ExperimentSpec,
    catalog: PreparedCatalog,
    rand_positions: np.ndarray,
    rand_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pypower import CatalogFFTPower

    kedges = build_kedges(spec)
    mu_wedges = build_mu_wedges(spec)
    result = CatalogFFTPower(
        data_positions1=catalog.positions_rdd,
        data_weights1=catalog.weights,
        randoms_positions1=rand_positions,
        randoms_weights1=rand_weights,
        nmesh=spec.nmesh,
        los='z',
        position_type='rdd',
        resampler='tsc',
        dtype='f8',
        ells=spec.ells,
        edges=(kedges, mu_wedges),
    )
    pkmu = result.wedges.get_power()
    plk = result.poles.power
    kcen = result.wedges.k[:, 0]
    return kcen, pkmu, plk


def run_single_experiment(spec: ExperimentSpec, mock_idx: int, dm: desi_mock | None = None) -> ExperimentResult:
    if dm is None:
        dm = desi_mock()

    if spec.mock_type == 'quijote':
        catalog = _prepare_quijote_catalog(spec, mock_idx, dm)
    elif spec.mock_type == 'halfdome':
        catalog = _prepare_halfdome_catalog(spec, mock_idx, dm)
    else:
        raise NotImplementedError(
            f"Unsupported mock_type='{spec.mock_type}'. Supported options are 'quijote' and 'halfdome'."
        )

    if spec.redshift_sel:
        redshift_mask = np.ones_like(catalog.base_redshifts, dtype=bool)
        if spec.zmin is not None:
            redshift_mask &= catalog.base_redshifts > spec.zmin
        if spec.zmax is not None:
            redshift_mask &= catalog.base_redshifts < spec.zmax
        catalog.positions_rdd = catalog.positions_rdd[:, redshift_mask]
        catalog.weights = catalog.weights[redshift_mask]
        catalog.base_redshifts = catalog.base_redshifts[redshift_mask]
        catalog.base_r = catalog.base_r[redshift_mask]
        if catalog.positions_rdd.shape[1] == 0:
            raise ValueError(
                f'Redshift selection z=[{spec.zmin}, {spec.zmax}] removed all objects for mock_idx={mock_idx}. '
                'This can happen for Quijote variants depending on box geometry. '
                'Try lowering zmin, enabling replication (replicate=True), or disabling redshift_sel.'
            )

    catalog = _apply_contamination(spec, catalog, mock_idx)
    rand_positions, rand_weights = _build_random_catalog(spec, catalog)
    kcen, pkmu, plk = _compute_power_spectra(spec, catalog, rand_positions, rand_weights)

    return ExperimentResult(
        spec=spec,
        kcen=kcen,
        all_pkmu=pkmu[None, ...],
        all_plk=plk[None, ...],
        mu_wedges=build_mu_wedges(spec),
        run_metadata={
            'mock_idx': mock_idx,
            'n_data': int(catalog.positions_rdd.shape[1]),
            'n_randoms': int(rand_positions.shape[1]),
            'contamination_mode': spec.contamination_mode,
        },
    )


def run_experiment_grid(spec: ExperimentSpec, nmock: int, dm: desi_mock | None = None) -> ExperimentResult:
    if dm is None:
        dm = desi_mock()

    mu_wedges = build_mu_wedges(spec)
    kedges = build_kedges(spec)
    nkbin = len(kedges) - 1
    nwedge = len(mu_wedges) - 1
    all_pkmu = np.zeros((nmock, nkbin, nwedge), dtype=complex)
    all_plk = np.zeros((nmock, len(spec.ells), nkbin), dtype=complex)
    kcen = None
    run_records: list[dict[str, Any]] = []

    for mock_idx in range(nmock):
        result = run_single_experiment(spec, mock_idx=mock_idx, dm=dm)
        all_pkmu[mock_idx] = result.all_pkmu[0]
        all_plk[mock_idx] = result.all_plk[0]
        if kcen is None:
            kcen = result.kcen
        run_records.append(result.run_metadata)

    if kcen is None:
        raise RuntimeError('No experiments were run.')

    return ExperimentResult(
        spec=spec,
        kcen=kcen,
        all_pkmu=all_pkmu,
        all_plk=all_plk,
        mu_wedges=mu_wedges,
        run_metadata={
            'records': run_records,
            'nmock': nmock,
            'kedges': kedges.tolist(),
            'label': build_run_label(spec),
        },
    )


def run_variant_collection(
    specs: list[ExperimentSpec],
    nmock: int,
    dm: desi_mock | None = None,
) -> dict[str, ExperimentResult]:
    results: dict[str, ExperimentResult] = {}
    for spec in specs:
        result = run_experiment_grid(spec, nmock=nmock, dm=dm)
        results[build_run_label(spec)] = result
    return results


def _normalize_options(values: Any, *, name: str) -> tuple[Any, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        return (values,)
    if np.isscalar(values):
        return (values,)
    return tuple(values)


def _coerce_bool_options(values: Any, *, name: str) -> tuple[bool, ...]:
    normalized = _normalize_options(values, name=name)
    coerced: list[bool] = []
    for value in normalized:
        if isinstance(value, bool):
            coerced.append(value)
            continue

        if isinstance(value, str):
            text = value.strip().lower()
            if text in {'true', 't', '1', 'yes', 'y'}:
                coerced.append(True)
                continue
            if text in {'false', 'f', '0', 'no', 'n'}:
                coerced.append(False)
                continue

        raise ValueError(f"Unsupported boolean value in {name}: {value!r}")

    return tuple(coerced)


def build_quijote_variant_specs(
    with_rsd_options: tuple[bool, ...] = (False, True),
    contamination_modes: tuple[str, ...] = ('none', 'stellar', 'dust', 'both'),
    stellar_fracs: tuple[float, ...] = (0.01,),
    sfd_stds: tuple[float, ...] = (0.01,),
    quijote_geometry: str = 'full_cube',
    base_kwargs: dict[str, Any] | None = None,
) -> list[ExperimentSpec]:
    base_kwargs = dict(base_kwargs or {})
    specs: list[ExperimentSpec] = []
    with_rsd_options = _coerce_bool_options(with_rsd_options, name='with_rsd_options')
    contamination_modes = _normalize_options(contamination_modes, name='contamination_modes')
    stellar_fracs = _normalize_options(stellar_fracs, name='stellar_fracs')
    sfd_stds = _normalize_options(sfd_stds, name='sfd_stds')

    if quijote_geometry not in {'full_cube', 'replicated'}:
        raise ValueError(
            f"Unsupported quijote_geometry='{quijote_geometry}'. Supported options are 'full_cube' and 'replicated'."
        )

    # Use explicitly provided sweep values as the canonical defaults, even for
    # contamination modes where that parameter is not active.
    default_stellar = stellar_fracs[0] if len(stellar_fracs) > 0 else base_kwargs.get('frac_stellar_contam', 0.01)
    default_dust = sfd_stds[0] if len(sfd_stds) > 0 else base_kwargs.get('sfd_std', 0.01)

    for with_rsd, contamination_mode in product(with_rsd_options, contamination_modes):
        if contamination_mode in {'stellar', 'both'}:
            stellar_values = stellar_fracs
        else:
            stellar_values = (default_stellar,)

        if contamination_mode in {'dust', 'both'}:
            dust_values = sfd_stds
        else:
            dust_values = (default_dust,)

        for frac_stellar_contam, sfd_std in product(stellar_values, dust_values):
            spec_kwargs = dict(base_kwargs)
            spec_kwargs.update(
                {
                    'mock_type': 'quijote',
                    'with_rsd': with_rsd,
                    'contamination_mode': contamination_mode,
                    'frac_stellar_contam': frac_stellar_contam,
                    'sfd_std': sfd_std,
                    'quijote_geometry': quijote_geometry,
                }
            )
            if quijote_geometry == 'full_cube' and not spec_kwargs.get('replicate', False):
                spec_kwargs['redshift_sel'] = False
            specs.append(ExperimentSpec(**spec_kwargs))

    return specs


def build_halfdome_variant_specs(
    contamination_modes: tuple[str, ...] = ('none', 'stellar', 'dust', 'both'),
    stellar_fracs: tuple[float, ...] = (0.01,),
    sfd_stds: tuple[float, ...] = (0.01,),
    base_kwargs: dict[str, Any] | None = None,
) -> list[ExperimentSpec]:
    base_kwargs = dict(base_kwargs or {})
    specs: list[ExperimentSpec] = []
    contamination_modes = _normalize_options(contamination_modes, name='contamination_modes')
    stellar_fracs = _normalize_options(stellar_fracs, name='stellar_fracs')
    sfd_stds = _normalize_options(sfd_stds, name='sfd_stds')

    # Use explicitly provided sweep values as the canonical defaults, even for
    # contamination modes where that parameter is not active.
    default_stellar = stellar_fracs[0] if len(stellar_fracs) > 0 else base_kwargs.get('frac_stellar_contam', 0.01)
    default_dust = sfd_stds[0] if len(sfd_stds) > 0 else base_kwargs.get('sfd_std', 0.01)

    for contamination_mode in contamination_modes:
        if contamination_mode in {'stellar', 'both'}:
            stellar_values = stellar_fracs
        else:
            stellar_values = (default_stellar,)

        if contamination_mode in {'dust', 'both'}:
            dust_values = sfd_stds
        else:
            dust_values = (default_dust,)

        for frac_stellar_contam, sfd_std in product(stellar_values, dust_values):
            spec_kwargs = dict(base_kwargs)
            spec_kwargs.update(
                {
                    'mock_type': 'halfdome',
                    'with_rsd': False,
                    'contamination_mode': contamination_mode,
                    'frac_stellar_contam': frac_stellar_contam,
                    'sfd_std': sfd_std,
                }
            )
            specs.append(ExperimentSpec(**spec_kwargs))

    return specs


def build_mock_variant_specs(
    mock_type: str,
    contamination_modes: tuple[str, ...] = ('none', 'stellar', 'dust', 'both'),
    stellar_fracs: tuple[float, ...] = (0.01,),
    sfd_stds: tuple[float, ...] = (0.01,),
    with_rsd_options: tuple[bool, ...] = (False, True),
    quijote_geometry: str = 'full_cube',
    base_kwargs: dict[str, Any] | None = None,
) -> list[ExperimentSpec]:
    """Build variant specs for either halfdome or quijote using one entry point."""
    mock_type_norm = mock_type.lower()
    contamination_modes = _normalize_options(contamination_modes, name='contamination_modes')
    stellar_fracs = _normalize_options(stellar_fracs, name='stellar_fracs')
    sfd_stds = _normalize_options(sfd_stds, name='sfd_stds')
    with_rsd_options = _coerce_bool_options(with_rsd_options, name='with_rsd_options')

    if mock_type_norm == 'quijote':
        return build_quijote_variant_specs(
            with_rsd_options=with_rsd_options,
            contamination_modes=contamination_modes,
            stellar_fracs=stellar_fracs,
            sfd_stds=sfd_stds,
            quijote_geometry=quijote_geometry,
            base_kwargs=base_kwargs,
        )

    if mock_type_norm == 'halfdome':
        return build_halfdome_variant_specs(
            contamination_modes=contamination_modes,
            stellar_fracs=stellar_fracs,
            sfd_stds=sfd_stds,
            base_kwargs=base_kwargs,
        )

    raise ValueError(
        f"Unsupported mock_type='{mock_type}'. Supported options are 'halfdome' and 'quijote'."
    )


def save_experiment_result(result: ExperimentResult) -> str:
    os.makedirs(result.spec.save_dir, exist_ok=True)
    label = result.spec.output_name or build_run_label(result.spec)
    output_path = os.path.join(result.spec.save_dir, f'{label}.npz')
    np.savez(
        output_path,
        kcen=result.kcen,
        all_pkmu=result.all_pkmu,
        all_plk=result.all_plk,
        ells=np.asarray(result.spec.ells),
        mu_wedges=result.mu_wedges,
        run_config_json=json.dumps(_spec_to_jsonable(result.spec), sort_keys=True),
        run_metadata_json=json.dumps(result.run_metadata, default=str, sort_keys=True),
        run_label=label,
    )
    append_run_ledger(
        result.spec.save_dir,
        {
            'label': label,
            'output_path': output_path,
            'config': _spec_to_jsonable(result.spec),
            'metadata': result.run_metadata,
        },
    )
    return output_path
