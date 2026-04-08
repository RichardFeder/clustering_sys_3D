from __future__ import annotations

import json
import os
from itertools import product
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from astropy.cosmology import Planck18 as cosmo

from contamination import (
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


@dataclass(frozen=True)
class ExperimentSpec:
    mock_type: str = 'quijote'
    with_rsd: bool = False
    contamination_mode: str = 'none'
    frac_stellar_contam: float = 0.01
    use_gaia: bool = True
    sfd_std: float = 0.01
    dust_alpha: float = -10.0
    redshift_sel: bool = False
    zmin: float = 0.4
    zmax: float = 1.0
    replicate: bool = False
    rep_fac: int = 3
    ds_fac: int = 100
    randomize: bool = False
    n_sample: int = 20_000_000
    k_min: float = 0.005
    k_max: float = 0.2
    delta_k: float = 0.01
    mu_min: float = 0.0
    mu_max: float = 1.0
    nwedge: int = 6
    mu_wedges: tuple[float, ...] | None = None
    ells: tuple[int, ...] = (0, 2, 4, 6, 8)
    nmesh: int = 256
    boxsize: float = 1000.0
    n_random_factor: int = 5
    seed: int = 42
    save_dir: str = 'data/plk'
    output_name: str | None = None
    plot: bool = False
    nplot: int = 0


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
    return np.linspace(spec.mu_min, spec.mu_max, spec.nwedge)


def build_run_label(spec: ExperimentSpec) -> str:
    parts = [spec.mock_type]
    parts.append('rsd' if spec.with_rsd else 'noRSD')
    parts.append(spec.contamination_mode)
    if spec.contamination_mode in {'stellar', 'both'}:
        parts.append(f'star{_slug_number(spec.frac_stellar_contam)}')
        parts.append('gaia' if spec.use_gaia else 'notional')
    if spec.contamination_mode in {'dust', 'both'}:
        parts.append(f'dust{_slug_number(spec.sfd_std)}')
        parts.append(f'a{_slug_number(spec.dust_alpha)}')
    if spec.redshift_sel:
        parts.append(f'z{_slug_number(spec.zmin)}-{_slug_number(spec.zmax)}')
    if spec.replicate:
        parts.append(f'rep{spec.rep_fac}')
    if spec.ds_fac != 1:
        parts.append(f'ds{spec.ds_fac}')
    if spec.mu_wedges is not None:
        parts.append('nonunifmu')
    else:
        parts.append(f'lmax{max(spec.ells)}')
    return '_'.join(parts)


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
        metadata={'boxsize_use': boxsize_use},
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


def _apply_contamination(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    if spec.contamination_mode not in {'none', 'stellar', 'dust', 'both'}:
        raise ValueError(f'Unknown contamination_mode: {spec.contamination_mode}')

    if spec.contamination_mode in {'dust', 'both'}:
        catalog = _apply_dust_systematic(spec, catalog, mock_idx)

    if spec.contamination_mode in {'stellar', 'both'}:
        catalog = _apply_stellar_systematic(spec, catalog, mock_idx)

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
    if spec.mock_type != 'quijote':
        raise NotImplementedError('Current runner only implements quijote in the first pass.')

    if dm is None:
        dm = desi_mock()

    catalog = _prepare_quijote_catalog(spec, mock_idx, dm)

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
    all_pkmu = np.zeros((nmock, nkbin, nwedge))
    all_plk = np.zeros((nmock, len(spec.ells), nkbin))
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


def build_quijote_variant_specs(
    with_rsd_options: tuple[bool, ...] = (False, True),
    contamination_modes: tuple[str, ...] = ('none', 'stellar', 'dust', 'both'),
    stellar_fracs: tuple[float, ...] = (0.01,),
    sfd_stds: tuple[float, ...] = (0.01,),
    base_kwargs: dict[str, Any] | None = None,
) -> list[ExperimentSpec]:
    base_kwargs = dict(base_kwargs or {})
    specs: list[ExperimentSpec] = []

    for with_rsd, contamination_mode in product(with_rsd_options, contamination_modes):
        if contamination_mode in {'stellar', 'both'}:
            stellar_values = stellar_fracs
        else:
            stellar_values = (base_kwargs.get('frac_stellar_contam', 0.01),)

        if contamination_mode in {'dust', 'both'}:
            dust_values = sfd_stds
        else:
            dust_values = (base_kwargs.get('sfd_std', 0.01),)

        for frac_stellar_contam, sfd_std in product(stellar_values, dust_values):
            spec_kwargs = dict(base_kwargs)
            spec_kwargs.update(
                {
                    'mock_type': 'quijote',
                    'with_rsd': with_rsd,
                    'contamination_mode': contamination_mode,
                    'frac_stellar_contam': frac_stellar_contam,
                    'sfd_std': sfd_std,
                }
            )
            specs.append(ExperimentSpec(**spec_kwargs))

    return specs


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
