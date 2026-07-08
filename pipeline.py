import json
import os
from itertools import product
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import healpy as hp
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
import astropy.units as u

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

# ─────────────────────────────────────────────────────────────────────────────
# Contamination map cache: stores computed sys_maps by (spec_label, mock_idx)
# to avoid recomputing the same field for each mode (additive, multiplicative).
# ─────────────────────────────────────────────────────────────────────────────
_contamination_map_cache: Dict[Tuple[str, int], np.ndarray] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Gaia HEALPix mask cache: stores the extragalactic Gaia coverage mask by mock_idx.
# The mask is a HEALPix array where True = coverage (counts > 0), False = no coverage.
# ─────────────────────────────────────────────────────────────────────────────
_gaia_healpix_mask_cache: Dict[int, np.ndarray] = {}


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
    zmin: float = 0.1
    zmax: float = 0.4
    replicate: bool = False
    rep_fac: int = 1
    ds_fac: int = 5
    randomize: bool = False
    n_sample: int = 20_000_000
    k_min: float = 0.006
    k_max: float = 0.2
    delta_k: float = 0.01
    mu_min: float = 0.0
    mu_max: float = 1.0
    nwedge: int = 6
    mu_wedges: Optional[Tuple[float, ...]] = None
    mu_binning_strategy: str = 'nonuniform'
    n_clean_bins: int = 8
    ells: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    nmesh: int = 512
    boxsize: float = 1000.0
    n_random_factor: int = 5
    seed: int = 42
    save_dir: str = 'data/plk'
    output_name: Optional[str] = None
    plot: bool = False
    nplot: int = 0
    quijote_basedir: str = DEFAULT_QUIJOTE_BASEDIR
    halfdome_basedir: str = DEFAULT_HALFDOME_BASEDIR
    quijote_geometry: str = 'replicated'
    # Power spectrum line-of-sight direction
    # 'z': plane-parallel along z-axis (correct for Quijote periodic boxes)
    # 'endpoint': per-galaxy LOS vector from observer (correct for wide-angle lightcones like Halfdome)
    los: str = 'z'
    # Target comoving number density for downsampling (h/Mpc)^3.
    # If None, no nbar-based downsampling is applied (use n_sample instead).
    target_nbar: Optional[float] = None
    # Controlled transverse systematic parameters
    sys_amp: float = 0.01
    sys_spec_type: str = 'power_law'
    sys_ell_min: int = 6
    sys_ell_max: int = 64
    sys_ell_delta: Optional[int] = None
    # For gaia_stellar spec_type: 'rms'=scale to RMS amplitude (default),
    # 'mean'=scale to mean value (contamination fraction)
    sys_amp_mode: str = 'rms'
    # Multiplicative scaling factor: scales the weight effect relative to additive.
    # E.g., sys_amp_mult_scale=2.0 makes multiplicative effects 2x stronger at fixed sys_amp.
    # For multiplicative: w_sys = 1 + sys_amp_mult_scale * sys_map, vs additive: n_inject ~ sys_amp * n_gal
    sys_amp_mult_scale: float = 1.0
    # Mean-conserving additive injection: in negative-lobe pixels, *remove*
    # existing galaxies (probabilistically) in addition to adding sources in
    # positive lobes. Preserves total n_gal and the input angular C_l (no
    # clipped-Gaussian distortion). When False, falls back to one-sided
    # positive-lobe-only Poisson injection (legacy behavior; biases n_gal by
    # +sys_amp and distorts the angular spectrum).
    mean_conserving_additive: bool = True
    # Debug: also apply the multiplicative w_sys to randoms. If True, an
    # unbiased estimator must yield ratio P_contam/P_clean ≈ 1 at all (k, mu).
    # Used to disentangle "real" injected systematic power from estimator-level
    # window-leakage / radial-projection effects.
    apply_sys_to_randoms: bool = False
    # Galactic latitude cut (degrees). When > 0, galaxies with |b| < gal_lat_cut_deg
    # are removed from both data and randoms, and the contamination map is zeroed
    # in the same region. Applied uniformly to clean and contaminated catalogs so
    # the ratio P_contam/P_clean is not affected by the Galactic-plane mask.
    gal_lat_cut_deg: float = 0.0


@dataclass
class PreparedCatalog:
    positions_rdd: np.ndarray
    weights: np.ndarray
    base_redshifts: Optional[np.ndarray]
    base_r: np.ndarray
    mock_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    kcen: np.ndarray
    all_pkmu: np.ndarray
    all_plk: np.ndarray
    mu_wedges: np.ndarray
    shot_noise: float
    output_path: Optional[str] = None
    label: Optional[str] = None
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    all_plk_null_lowest_mu: Optional[np.ndarray] = None  # P_ℓ(k) with lowest μ bin nulled


def _slug_number(value: Union[float, int]) -> str:
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


def _spec_to_jsonable(spec: ExperimentSpec) -> Dict[str, Any]:
    data = asdict(spec)
    if data['mu_wedges'] is not None:
        data['mu_wedges'] = list(data['mu_wedges'])
    return data


def build_kedges(spec: ExperimentSpec) -> np.ndarray:
    return np.arange(spec.k_min, spec.k_max + spec.delta_k, spec.delta_k)


def build_mu_wedges(spec: ExperimentSpec) -> np.ndarray:
    """Build mu bin edges for power spectrum.
    
    Supports: 'uniform', 'nonuniform', 'delta_function', or manual mu_wedges.
    """
    if spec.mu_wedges is not None:
        return np.asarray(spec.mu_wedges, dtype=float)

    strategy = spec.mu_binning_strategy.lower()
    
    if strategy == 'delta_function':
        # Delta-function systematic binning: use analytical formula
        from nonunif_binning import compute_mu_edges_for_delta_function
        
        # Extract required parameters
        if not hasattr(spec, 'sys_spec_type') or spec.sys_spec_type != 'delta':
            raise ValueError(
                'delta_function strategy requires sys_spec_type="delta"'
            )
        if not hasattr(spec, 'sys_ell_delta'):
            raise ValueError('delta_function strategy requires sys_ell_delta attribute')
        if not hasattr(spec, 'zmin') or not hasattr(spec, 'zmax'):
            raise ValueError('delta_function strategy requires zmin and zmax')
        
        z_eff = (spec.zmin + spec.zmax) / 2.0
        ell_contam = spec.sys_ell_delta
        R_window = getattr(spec, 'sys_window_r', 780.0)
        ell_max = max(spec.ells)
        ell_kernel_max = getattr(spec, 'sys_ell_kernel_max', 128)
        n_clean_bins = spec.n_clean_bins
        A = getattr(spec, 'sys_kernel_amp', 1e-5)
        mu1_min = getattr(spec, 'sys_mu1_min', 0.02)
        mu1_max = getattr(spec, 'sys_mu1_max', 0.3)
        verbose = getattr(spec, 'verbose', False)
        
        return compute_mu_edges_for_delta_function(
            ell_contam=ell_contam,
            z_eff=z_eff,
            ell_max=ell_max,
            ell_kernel_max=ell_kernel_max,
            R_window=R_window,
            A=A,
            n_clean_bins=n_clean_bins,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            verbose=verbose,
        )
    
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
        if spec.sys_amp_mult_scale != 1.0:
            parts.append(f'multscale{_slug_number(spec.sys_amp_mult_scale)}')
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


def recommended_base_kwargs(mock_type: str = 'halfdome') -> Dict[str, Any]:
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
        'zmin': 0.1,
        'zmax': 0.4,
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
        'ells': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        'nmesh': 512,
        'seed': 42,
        'quijote_geometry': 'full_cube' if mock_type_norm == 'quijote' else 'replicated',
        # Global plane-parallel LOS ('z') for both mock types.
        # Wedges are recovered analytically from multipoles via _poles_to_wedges.
        'los': 'z',
    }


def append_run_ledger(save_dir: str, entry: Dict[str, Any]) -> str:
    ledger_path = os.path.join(save_dir, 'run_ledger.jsonl')
    os.makedirs(save_dir, exist_ok=True)
    with open(ledger_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, sort_keys=True) + '\n')
    return ledger_path


def _stage_seeds(spec: ExperimentSpec, mock_idx: int) -> Dict[str, int]:
    """Return per-stage deterministic seeds derived from spec.seed and mock_idx.

    Each pipeline stage draws from its own independent RNG stream so that
    changing one stage (e.g. the dust seed) does not perturb the randoms or
    the stellar injection.  Seeds are offset by 10_000 * mock_idx so that
    different mock realizations never share a sub-stream.

    Keys
    ----
    quijote      : galaxy position loading / Halfdome subsampling
    dust         : delta-E(B-V) GRF synthesis + transverse map synthesis
    stellar_map  : Poisson star-count draw from stellar density map
    stellar_radec: angular position sampling within pixels (and injection add step)
    stellar_redshift: radial distance resampling for injected sources
    randoms      : uniform random catalog generation
    """
    base = int(spec.seed) + mock_idx * 10_000
    return {
        'quijote': base + 1,
        'dust': base + 2,
        'stellar_map': base + 3,
        'stellar_radec': base + 4,
        'stellar_redshift': base + 5,
        'randoms': base + 6,
    }


def _load_or_compute_gaia_healpix_mask(mock_idx: int) -> np.ndarray:
    """
    Load or compute the Gaia extragalactic coverage mask as a HEALPix array.
    
    The mask is True (1) where stellar_counts > 0 (survey has coverage),
    and False (0) where stellar_counts == 0 (no coverage).
    
    This mask is cached by mock_idx to avoid recomputing for each contamination mode.
    
    Returns
    -------
    mask : np.ndarray (HEALPix boolean array)
        True where coverage exists, False where no coverage.
    """
    if mock_idx in _gaia_healpix_mask_cache:
        return _gaia_healpix_mask_cache[mock_idx]
    
    # Load the stellar density map (same for all mocks)
    stellar_map = load_gaia_stellar_density(plot=False)
    
    # Create binary mask: True where stellar counts > 0
    mask = (stellar_map > 0).astype(bool)
    
    # Cache and return
    _gaia_healpix_mask_cache[mock_idx] = mask
    return mask


def _prepare_quijote_catalog(spec: ExperimentSpec, mock_idx: int, dm: desi_mock) -> PreparedCatalog:
    """Load and pre-process a Quijote periodic-box mock into a PreparedCatalog.

    Steps
    -----
    1. Load halo positions from the Quijote snapshot at mock_idx (optionally
       with RSD displacement along the z-axis).
    2. Down-sample by spec.ds_fac (default 5), or replicate the box by
       spec.rep_fac if spec.replicate is True.
    3. Convert Cartesian (x, y, z) ∈ [0, L]^3 to (RA, Dec, r) with the
       observer placed at the box centre (L/2, L/2, L/2).
    4. If spec.redshift_sel is True, compute a comoving-distance → redshift
       mapping via a Planck18 interpolator and store in base_redshifts so
       the redshift window [zmin, zmax] can be applied upstream in
       run_single_experiment.  Without replication the Quijote box at
       ds_fac=5 typically yields ~10^5 galaxies after the z-window cut.

    The periodic geometry means no angular footprint mask is needed; the
    full survey volume is the box itself.
    """
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
    
    # For periodic box, keep positions in Cartesian coordinates (x, y, z)
    # This ensures los='z' in CatalogFFTPower correctly refers to the box z-axis
    positions_rdd = galpos.T
    weights = np.ones(positions_rdd.shape[1], dtype=float)
    # weights = np.ones_like(positions_rdd, dtype=float)
    base_r = galpos[2]  # z-coordinate as the radial/depth coordinate

    # For periodic box, no redshift_sel is applied (the z-coordinate IS the radial coordinate)
    base_redshifts = None

    print("prepared catalog positions has shape", positions_rdd.shape)
    print('weights have shape', weights.shape)

    return PreparedCatalog(
        positions_rdd=positions_rdd,
        weights=weights,
        base_redshifts=base_redshifts,
        base_r=base_r,
        mock_idx=mock_idx,
        metadata={'boxsize_use': boxsize_use, 'quijote_basedir': quijote_basedir},
    )


def _gal_lat_b(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Return galactic latitude b (degrees) for equatorial RA/Dec arrays."""
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    return c.galactic.b.deg


def _apply_gal_lat_cut(catalog: PreparedCatalog, cut_deg: float) -> PreparedCatalog:
    """Remove catalog entries with |b| < cut_deg. Applied to data before contamination
    injection so both clean and contaminated catalogs share the same angular footprint."""
    if cut_deg <= 0.0:
        return catalog
    ra  = catalog.positions_rdd[0]
    dec = catalog.positions_rdd[1]
    b   = _gal_lat_b(ra, dec)
    keep = np.abs(b) >= cut_deg
    catalog.positions_rdd    = catalog.positions_rdd[:, keep]
    catalog.weights          = catalog.weights[keep]
    catalog.base_r           = catalog.base_r[keep]
    if catalog.base_redshifts is not None:
        catalog.base_redshifts = catalog.base_redshifts[keep]
    return catalog


def _apply_healpix_mask_to_catalog(catalog: PreparedCatalog, mask: np.ndarray, nside: Optional[int] = None) -> PreparedCatalog:
    """Apply a HEALPix binary mask to the catalog. Removes catalog entries in masked (False) pixels.
    
    Parameters
    ----------
    catalog : PreparedCatalog
        Input catalog to mask.
    mask : np.ndarray
        HEALPix binary mask (True = keep, False = masked out).
    nside : int, optional
        HEALPix nside. If None, inferred from mask length.
    
    Returns
    -------
    catalog : PreparedCatalog
        Masked catalog with entries in masked regions removed.
    """
    if mask is None or np.all(mask):
        return catalog  # No masking needed
    
    if nside is None:
        nside = hp.npix2nside(len(mask))
    
    ra  = catalog.positions_rdd[0]
    dec = catalog.positions_rdd[1]
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    
    # Keep entries where mask is True
    keep = mask[pix]
    n_before = catalog.positions_rdd.shape[1]
    
    catalog.positions_rdd = catalog.positions_rdd[:, keep]
    catalog.weights = catalog.weights[keep]
    catalog.base_r = catalog.base_r[keep]
    if catalog.base_redshifts is not None:
        catalog.base_redshifts = catalog.base_redshifts[keep]
    
    n_after = catalog.positions_rdd.shape[1]
    print(f'  Gaia coverage mask applied: {n_before:,} → {n_after:,} galaxies')
    
    return catalog


def _prepare_halfdome_catalog(spec: ExperimentSpec, mock_idx: int, dm: desi_mock) -> PreparedCatalog:
    if not spec.halfdome_basedir.endswith('/'):
        halfdome_basedir = spec.halfdome_basedir + '/'
    else:
        halfdome_basedir = spec.halfdome_basedir

    dm.halfdome_mock_basedir = halfdome_basedir
    
    # Use seeded sampling for reproducible galaxy subsampling
    seeds = _stage_seeds(spec, mock_idx)
    galpos, redshift = dm.load_halfdome_mock(mock_idx, n_sample=spec.n_sample, seed=seeds['quijote'])

    ra, dec, r = convert_to_ra_dec_distance(galpos, spec.boxsize, center_offset_mpc=0.0)
    r_values = np.asarray(r.value if hasattr(r, 'value') else r)
    positions_rdd = np.vstack([ra, dec, r_values])
    weights = np.ones_like(ra, dtype=float)

    print('halfdome catalog has shape', positions_rdd.shape)

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


def _get_or_compute_contamination_map(
    spec: ExperimentSpec, 
    mock_idx: int, 
    catalog: PreparedCatalog,
) -> np.ndarray:
    """
    Get or compute the transverse contamination map, using a cache to avoid
    recomputation when the same map is needed for multiple contamination modes.
    
    The map is cached by (spec_label, mock_idx) so it is computed once per
    mock_idx and reused for both additive and multiplicative modes.
    
    Reproducibility: The map is always computed with the same seed for a given
    (spec, mock_idx) pair, so the same field is generated every time. All RNG
    operations throughout the pipeline use deterministic seeds derived from
    spec.seed and mock_idx via _stage_seeds(), ensuring full reproducibility.
    """
    label = build_run_label(spec)
    cache_key = (label, mock_idx)
    
    # Return cached map if available
    if cache_key in _contamination_map_cache:
        return _contamination_map_cache[cache_key]
    
    # Otherwise, generate and cache it
    seeds = _stage_seeds(spec, mock_idx)
    periodic = (spec.mock_type == 'quijote')
    boxsize_use = catalog.metadata.get('boxsize_use', spec.boxsize)

    if spec.sys_spec_type == 'gaia_stellar':
        # Build a δn/n template from the Gaia stellar density map, normalized
        # to have RMS = spec.sys_amp or mean = spec.sys_amp depending on sys_amp_mode.
        # This lets us apply the SAME anisotropic angular template as both additive
        # and multiplicative contamination so we can directly compare their μ-dependence.
        #
        # Note: The Gaia extragalactic coverage mask has already been applied to the
        # data catalog early in run_single_experiment(). Here we just normalize the
        # stellar map (preserving its inherent Gaia zeros) to the target amplitude.
        stellar_map = load_gaia_stellar_density(plot=False)
        m = stellar_map.astype(float)
        
        # Identify regions with Gaia coverage (counts > 0) for normalization
        gaia_coverage = (stellar_map > 0)
        unmasked = m[gaia_coverage]
        
        # Apply galactic latitude masking if requested (for the map itself, not the data)
        mask_pixels = ~gaia_coverage.copy()
        if spec.gal_lat_cut_deg > 0.0:
            nside_m = hp.npix2nside(len(m))
            pix_idx = np.arange(len(m))
            pix_theta, pix_phi = hp.pix2ang(nside_m, pix_idx)
            pix_ra  = np.degrees(pix_phi)
            pix_dec = 90.0 - np.degrees(pix_theta)
            pix_b   = _gal_lat_b(pix_ra, pix_dec)
            plane_mask = np.abs(pix_b) < spec.gal_lat_cut_deg
            mask_pixels |= plane_mask  # Add galactic plane to masked region
            unmasked = m[~mask_pixels]  # Recompute unmasked after plane cut
        
        # Normalize the map to target amplitude (using only unmasked regions for statistics)
        if spec.sys_amp_mode == 'mean':
            current_mean = unmasked.mean()
            if abs(current_mean) > 1e-10:
                sys_map = (spec.sys_amp / current_mean) * m
            else:
                sys_map = m
        else:  # 'rms' mode (default)
            m_centered = m - unmasked.mean()
            std = m_centered[~mask_pixels].std()
            if std > 0:
                sys_map = (spec.sys_amp / std) * m_centered
            else:
                sys_map = m_centered
        
        # Zero out masked regions (Gaia no-coverage + galactic plane)
        sys_map[mask_pixels] = 0.0
    else:
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

    _contamination_map_cache[cache_key] = sys_map
    return sys_map


def _apply_transverse_additive_sys(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    seeds = _stage_seeds(spec, mock_idx)
    periodic = (spec.mock_type == 'quijote')
    boxsize_use = spec.boxsize * spec.rep_fac if spec.replicate else spec.boxsize
    
    # Use cached contamination map (computed once per mock_idx, reused across modes)
    sys_map = _get_or_compute_contamination_map(spec, mock_idx, catalog)

    n_gal = catalog.positions_rdd.shape[1]
    rng = np.random.default_rng(seeds['stellar_map'])

    if not periodic:
        # ──────────────────────────────────────────────────────────────────────
        # HALFDOME LIGHTCONE PATH: Handle positive lobe injection + (optionally)
        # negative lobe removal for mean-conserving additive systematics
        # ──────────────────────────────────────────────────────────────────────
        nside_map = hp.npix2nside(len(sys_map))
        
        # ── Phase 1: Inject galaxies in positive lobe ───────────────────────
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

        n_inject = len(ra_inject)
        rand_indices = rng.choice(len(catalog.base_r), size=n_inject, replace=True)
        r_star = catalog.base_r[rand_indices]
        galpos_inj = np.vstack([ra_inject, dec_inject, r_star])

        print('gal pos inject from healpix map has shape', galpos_inj.shape)

        print('catalog rmin/rmax:', catalog.base_r.min(), catalog.base_r.max())
        print('injected rmin/rmax:', galpos_inj[2].min(), galpos_inj[2].max())
        print('catalog ra/dec min/max:', catalog.positions_rdd[0].min(), catalog.positions_rdd[0].max(), catalog.positions_rdd[1].min(), catalog.positions_rdd[1].max())
        print('injected ra/dec min/max:', galpos_inj[0].min(), galpos_inj[0].max(), galpos_inj[1].min(), galpos_inj[1].max())

        # ── Phase 2: Remove galaxies in negative lobe (if mean_conserving) ──
        removal_mask = np.ones(n_gal, dtype=bool)
        n_removed = 0
        
        if spec.mean_conserving_additive:
            negative_map = np.clip(-sys_map, 0.0, None)
            max_removal_prob = negative_map.max()
            
            if max_removal_prob > 0:
                # Convert galaxy positions to HEALPix pixel indices
                ra_gal = catalog.positions_rdd[0]
                dec_gal = catalog.positions_rdd[1]
                theta_gal = np.radians(90.0 - dec_gal)  # Dec to theta
                phi_gal = np.radians(ra_gal)
                pix_gal = hp.ang2pix(nside_map, theta_gal, phi_gal, nest=False)
                
                # Compute removal probability for each galaxy based on local map value
                removal_probs = negative_map[pix_gal] / max_removal_prob
                
                # Probabilistically mark galaxies for removal
                u_remove = rng.uniform(0.0, 1.0, size=n_gal)
                removal_mask = u_remove >= removal_probs  # Keep if u >= prob
                n_removed = np.sum(~removal_mask)
                
                print(f'Mean-conserving removal: {n_removed} galaxies marked for removal '
                      f'({100.0 * n_removed / n_gal:.2f}% of catalog)')

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
            zs_inj = np.random.uniform(0.0, boxsize_use, size=n_inject)  # Randomize z to avoid artificial clustering in the middle plane
            galpos_inj = np.stack([xs_inj, ys_inj, zs_inj], axis=1).T


    inject_weights = np.ones(n_inject, dtype=float)

    print('positions rdd has shape', catalog.positions_rdd.shape)
    print('inject positions has shape', galpos_inj.shape)

    # Apply removal mask to existing catalog (halfdome only)
    if not periodic and spec.mean_conserving_additive and n_removed > 0:
        catalog.positions_rdd = catalog.positions_rdd[:, removal_mask]
        catalog.weights = catalog.weights[removal_mask]
        print(f'After removal: catalog has {catalog.positions_rdd.shape[1]} galaxies')

    catalog.positions_rdd = np.concatenate([catalog.positions_rdd, galpos_inj], axis=1)

    print("catalog.positions_rdd now has shape ", catalog.positions_rdd.shape)
    print('catalog.weights has shape', catalog.weights.shape)

    catalog.weights = np.concatenate([catalog.weights, inject_weights])
    print('catalog.weights after injection has shape', catalog.weights.shape)

    return catalog


def _apply_transverse_multiplicative_sys(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    seeds = _stage_seeds(spec, mock_idx)
    periodic = (spec.mock_type == 'quijote')
    boxsize_use = spec.boxsize * spec.rep_fac if spec.replicate else spec.boxsize

    sys_map = _get_or_compute_contamination_map(spec, mock_idx, catalog)
    
    # Scale the systematic map by sys_amp_mult_scale to control relative strength of multiplicative effect
    sys_map_scaled = sys_map * spec.sys_amp_mult_scale

    if not periodic:
        ra = catalog.positions_rdd[0]
        dec = catalog.positions_rdd[1]
        nside_map = hp.npix2nside(len(sys_map_scaled))
        weights, sys_weights = modify_fkp_weights(ra, dec, catalog.weights, sys_map_scaled, nside=nside_map)
        catalog.metadata['transverse_multiplicative_nside'] = nside_map
    else:
        # positions_rdd stores raw Cartesian (x, y, z) for periodic boxes —
        # no RA/Dec inversion needed
        x_box = catalog.positions_rdd[0]
        y_box = catalog.positions_rdd[1]
        ngrid = sys_map_scaled.shape[0]
        ix = np.floor(x_box / boxsize_use * ngrid).astype(int) % ngrid
        iy = np.floor(y_box / boxsize_use * ngrid).astype(int) % ngrid
        dn_over_n = sys_map_scaled[ix, iy]
        w_sys = np.clip(1.0 + dn_over_n, 0.01, 10.0)
        weights = catalog.weights * w_sys
        sys_weights = w_sys

    catalog.weights = weights
    catalog.metadata['transverse_multiplicative_sys_weights'] = sys_weights
    catalog.metadata['transverse_multiplicative_sys_map'] = sys_map_scaled
    catalog.metadata['transverse_multiplicative_sys_amp_mult_scale'] = spec.sys_amp_mult_scale
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


def _generate_uniform_randoms_in_healpix_mask(
    n_randoms: int, 
    gaia_mask: np.ndarray, 
    chi_interp,
    z_source: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random RA/Dec positions uniformly within a HEALPix mask region.
    
    This ensures the data/randoms ratio is maintained even when a mask removes
    regions of the sky.
    
    Parameters
    ----------
    n_randoms : int
        Number of random points to generate.
    gaia_mask : np.ndarray
        HEALPix mask (True = covered/keep, False = masked/skip).
    chi_interp : callable
        Comoving distance interpolator (z -> chi).
    z_source : np.ndarray
        Source redshift distribution to sample from.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    ra_rand, dec_rand, r_rand : np.ndarray
        Random RA (degrees), Dec (degrees), r (Mpc/h) positions.
    """
    rng = np.random.default_rng(seed)
    nside = hp.npix2nside(len(gaia_mask))
    
    # Get all pixels where mask is True (covered region)
    covered_pixels = np.where(gaia_mask)[0]
    if len(covered_pixels) == 0:
        raise ValueError("Gaia mask has no covered pixels.")
    
    # Sample from the source z-distribution
    z_indices = rng.choice(len(z_source), size=n_randoms, replace=True)
    z_rand = z_source[z_indices]
    
    # Sample pixel indices from covered pixels for each random
    pixel_indices = rng.choice(covered_pixels, size=n_randoms, replace=True)
    
    # For each pixel, generate RA/Dec uniformly within pixel bounds
    # Get pixel centers and approximate sizes
    lon_pix, lat_pix = hp.pix2ang(nside, pixel_indices, lonlat=True)  # (lon, lat) in degrees
    
    # Pixel angular size in degrees ~ sqrt(4π / N_pix) / sqrt(12)
    pixel_area_sr = hp.nside2pixarea(nside)  # steradians
    pixel_size_deg = np.degrees(np.sqrt(pixel_area_sr))  # approximate half-size
    
    # Generate RA/Dec uniformly around pixel centers
    # Sample from uniform distribution in pixel's approximate bounding box
    half_size = pixel_size_deg / 2.0
    ra_offset = rng.uniform(-half_size, half_size, size=n_randoms)
    dec_offset = rng.uniform(-half_size, half_size, size=n_randoms)
    
    ra_rand = lon_pix + ra_offset
    dec_rand = lat_pix + dec_offset
    
    # Wrap RA to [0, 360)
    ra_rand = ra_rand % 360.0
    
    # Clip Dec to [-90, 90]
    dec_rand = np.clip(dec_rand, -90.0, 90.0)
    
    # Convert z to comoving distance
    r_rand_mpc = chi_interp(z_rand)
    r_rand = r_rand_mpc * cosmo.h  # Convert to Mpc/h
    
    return ra_rand, dec_rand, r_rand


def _build_random_catalog_periodic(spec: ExperimentSpec, catalog: PreparedCatalog) -> Tuple[np.ndarray, np.ndarray]:
    seeds = _stage_seeds(spec, catalog.mock_idx)
    n_randoms = int(spec.n_random_factor * catalog.positions_rdd.shape[1])
    boxsize_use = catalog.metadata['boxsize_use']

    print('boxsize use:', boxsize_use)

    rng = np.random.default_rng(seeds['randoms'])
    rand_positions = rng.uniform(0.0, boxsize_use, size=(3, n_randoms))

    rand_weights = np.ones_like(rand_positions[0], dtype=float)

    return rand_positions, rand_weights

def _build_random_catalog(spec: ExperimentSpec, catalog: PreparedCatalog) -> Tuple[np.ndarray, np.ndarray]:
    seeds = _stage_seeds(spec, catalog.mock_idx)
    n_randoms = int(spec.n_random_factor * catalog.positions_rdd.shape[1])
    boxsize_use = catalog.metadata['boxsize_use']

    if spec.redshift_sel:
        chi_interp = grab_chi_interp()
        
        # Use original (pre-mask) z-distribution for randoms to maintain constant comoving density
        redshift_source = catalog.metadata.get('original_base_redshifts')
        if redshift_source is None:
            # Fallback for non-gaia_stellar: use current (possibly masked) base_redshifts
            redshift_source = catalog.base_redshifts
        
        if redshift_source is None:
            raise ValueError('Redshift selection requested but no base redshifts were prepared.')
        if len(redshift_source) == 0:
            print('Warning: redshift-selected data is empty; falling back to uniform random z sampling in [zmin, zmax].')
            redshift_source = None
        
        # For Gaia stellar, generate randoms uniformly WITHIN the masked region
        # This maintains the proper data/randoms ratio and avoids low-k power artifacts
        if spec.sys_spec_type == 'gaia_stellar':
            gaia_mask = catalog.metadata.get('gaia_coverage_mask')
            if gaia_mask is not None and redshift_source is not None:
                ra_rand, dec_rand, r_rand = _generate_uniform_randoms_in_healpix_mask(
                    n_randoms,
                    gaia_mask,
                    chi_interp,
                    redshift_source,
                    seed=seeds['randoms'],
                )
                rand_positions = np.array([ra_rand, dec_rand, r_rand], dtype=float)
                rand_weights = np.ones_like(ra_rand, dtype=float)
                print(f'  Randoms generated uniformly within Gaia mask: {n_randoms:,} points')
            else:
                # Fallback: generate uniformly over full sky, then filter
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
        else:
            # Standard generation for non-Gaia modes
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
        
        # Apply galactic latitude cut (standard cut, independent of Gaia mask)
        if spec.gal_lat_cut_deg > 0.0:
            b_rand = _gal_lat_b(rand_positions[0], rand_positions[1])
            keep = np.abs(b_rand) >= spec.gal_lat_cut_deg
            n_before = rand_positions.shape[1]
            rand_positions = rand_positions[:, keep]
            rand_weights = rand_weights[keep]
            print(f'  gal_lat_cut |b|>{spec.gal_lat_cut_deg:.1f}° applied to randoms: '
                  f'{n_before:,} → {rand_positions.shape[1]:,}')
        
        rand_positions, rand_weights = _maybe_apply_sys_to_randoms(
            spec, catalog, rand_positions, rand_weights
        )
        return rand_positions, rand_weights

    rng = np.random.default_rng(seeds['randoms'])
    randoms_positions = rng.uniform(0.0, boxsize_use, size=(n_randoms, 3))
    ra_rand, dec_rand, r_rand = convert_to_ra_dec_distance(randoms_positions, boxsize_use)
    r_values = np.asarray(r_rand.value if hasattr(r_rand, 'value') else r_rand)
    rand_positions = np.array([ra_rand, dec_rand, r_values], dtype=float)
    rand_weights = np.ones_like(ra_rand, dtype=float)
    if spec.gal_lat_cut_deg > 0.0:
        b_rand = _gal_lat_b(rand_positions[0], rand_positions[1])
        keep = np.abs(b_rand) >= spec.gal_lat_cut_deg
        n_before = rand_positions.shape[1]
        rand_positions = rand_positions[:, keep]
        rand_weights = rand_weights[keep]
        print(f'  gal_lat_cut |b|>{spec.gal_lat_cut_deg:.1f}° applied to randoms: '
              f'{n_before:,} → {rand_positions.shape[1]:,}')
    
    # Apply Gaia coverage mask to randoms for consistency with masked data
    if spec.sys_spec_type == 'gaia_stellar':
        gaia_mask = catalog.metadata.get('gaia_coverage_mask')
        if gaia_mask is not None:
            nside = hp.npix2nside(len(gaia_mask))
            # gaia_mask is True where coverage exists (stellar counts > 0)
            # We want to keep randoms in these regions
            theta = np.radians(90.0 - rand_positions[1])
            phi = np.radians(rand_positions[0])
            pix = hp.ang2pix(nside, theta, phi, nest=False)
            keep = gaia_mask[pix]
            n_before = rand_positions.shape[1]
            rand_positions = rand_positions[:, keep]
            rand_weights = rand_weights[keep]
            n_after = rand_positions.shape[1]
            print(f'  Gaia coverage mask applied to randoms: {n_before:,} → {n_after:,}')
    
    rand_positions, rand_weights = _maybe_apply_sys_to_randoms(
        spec, catalog, rand_positions, rand_weights
    )
    return rand_positions, rand_weights


def _maybe_apply_sys_to_randoms(
    spec: ExperimentSpec,
    catalog: PreparedCatalog,
    rand_positions: np.ndarray,
    rand_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phase C9: optionally apply the multiplicative w_sys to randoms.

    If the all-mu leakage in P(k,mu) is an estimator-level effect (e.g.
    α-normalization bias), applying the same w_sys to randoms will
    cancel it and the ratio P_contam/P_clean should equal 1 at all
    (k, mu). If the leakage persists, it reflects real injected power
    convolved with the radial-window response of the lightcone.
    """
    if not spec.apply_sys_to_randoms:
        return rand_positions, rand_weights
    if spec.contamination_mode != 'transverse_multiplicative':
        return rand_positions, rand_weights
    sys_map = catalog.metadata.get('transverse_multiplicative_sys_map')
    if sys_map is None:
        return rand_positions, rand_weights
    nside_map = catalog.metadata.get('transverse_multiplicative_nside')
    if nside_map is None:
        return rand_positions, rand_weights
    ra_r = rand_positions[0]
    dec_r = rand_positions[1]
    new_weights, _ = modify_fkp_weights(ra_r, dec_r, rand_weights, sys_map, nside=nside_map)
    return rand_positions, new_weights


def _poles_to_wedges(ells, plk_arr: np.ndarray, mu_wedges: np.ndarray) -> np.ndarray:
    """
    Convert power spectrum multipoles to μ-wedges analytically.

    P(k, μ₁<μ<μ₂) = Σ_ℓ P_ℓ(k) × W_ℓ(μ₁, μ₂)

    where the bin-averaged Legendre window is:
      W_0 = 1
      W_ℓ = [P_{ℓ+1}(μ₂) - P_{ℓ-1}(μ₂) - P_{ℓ+1}(μ₁) + P_{ℓ-1}(μ₁)] / (μ₂ - μ₁)
            using the recurrence ∫ L_ℓ dμ = [P_{ℓ+1}(μ) - P_{ℓ-1}(μ)] / (2ℓ+1)
            combined with the (2ℓ+1) prefactor from the multipole expansion.

    Parameters
    ----------
    ells : sequence of int, shape (nell,)
    plk_arr : ndarray, shape (nk, nell)  — multipoles indexed as [k, ell]
    mu_wedges : ndarray, shape (nmu+1,) — μ bin edges

    Returns
    -------
    pkmu : ndarray, shape (nk, nmu)
    """
    from scipy.special import legendre as Leg

    nmu = len(mu_wedges) - 1
    nk = plk_arr.shape[0]
    pkmu = np.zeros((nk, nmu), dtype=complex)

    for mu_idx in range(nmu):
        mu1, mu2 = mu_wedges[mu_idx], mu_wedges[mu_idx + 1]
        dmu = mu2 - mu1
        for ell_idx, ell in enumerate(ells):
            if ell == 0:
                window = 1.0
            else:
                # ∫_{μ1}^{μ2} L_ℓ(μ) dμ = [P_{ℓ+1}(μ) - P_{ℓ-1}(μ)]/(2ℓ+1) |_{μ1}^{μ2}
                Lp = Leg(ell + 1)
                Lm = Leg(ell - 1)
                integral = (Lp(mu2) - Lm(mu2) - Lp(mu1) + Lm(mu1)) / (2 * ell + 1)
                window = (2 * ell + 1) * integral / dmu
            pkmu[:, mu_idx] += window * plk_arr[:, ell_idx]

    return pkmu


def _wedges_to_poles(ells, pkmu: np.ndarray, mu_wedges: np.ndarray) -> np.ndarray:
    """
    Convert power spectrum μ-wedges back to multipoles via least-squares inversion.

    Inverts the relationship: P(k, μ₁<μ<μ₂) = Σ_ℓ P_ℓ(k) × W_ℓ(μ₁, μ₂)

    For each k independently, solves: W · P_poles = P_wedges
    to recover P_poles via W^{-1}.

    Parameters
    ----------
    ells : sequence of int, shape (nell,)
    pkmu : ndarray, shape (nk, nmu) — power in wedges indexed as [k, mu_wedge]
    mu_wedges : ndarray, shape (nmu+1,) — μ bin edges

    Returns
    -------
    plk_arr : ndarray, shape (nk, nell) — multipoles indexed as [k, ell]
    """
    from scipy.special import legendre as Leg

    nmu = len(mu_wedges) - 1
    nell = len(ells)
    nk = pkmu.shape[0]
    plk_arr = np.zeros((nk, nell), dtype=complex)

    # Build window matrix W for this mu_wedges configuration
    W = np.zeros((nmu, nell))
    for mu_idx in range(nmu):
        mu1, mu2 = mu_wedges[mu_idx], mu_wedges[mu_idx + 1]
        dmu = mu2 - mu1
        for ell_idx, ell in enumerate(ells):
            if ell == 0:
                window = 1.0
            else:
                Lp = Leg(ell + 1)
                Lm = Leg(ell - 1)
                integral = (Lp(mu2) - Lm(mu2) - Lp(mu1) + Lm(mu1)) / (2 * ell + 1)
                window = (2 * ell + 1) * integral / dmu
            W[mu_idx, ell_idx] = window

    # Compute W^{-1} (or use least-squares if not square)
    if nmu == nell:
        W_inv = np.linalg.inv(W)
    else:
        # Over/under-determined: use least-squares solution
        W_inv = np.linalg.pinv(W)

    # For each k, recover multipoles: P_poles = W^{-1} · P_wedges
    for k_idx in range(nk):
        plk_arr[k_idx, :] = W_inv @ pkmu[k_idx, :]

    return plk_arr


def _compute_raw_multipoles(
    spec: ExperimentSpec,
    catalog: PreparedCatalog,
    rand_positions: np.ndarray,
    rand_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute power spectrum multipoles P_ell(k) without computing wedges.

    This is the "raw" FFT computation that is independent of mu binning.
    The result can be cached and reused to reconstruct different mu binning schemes.

    Parameters
    ----------
    spec : ExperimentSpec
        Experiment specification
    catalog : PreparedCatalog
        Prepared galaxy catalog (positions, weights, etc.)
    rand_positions : np.ndarray
        Random catalog positions
    rand_weights : np.ndarray
        Random catalog weights

    Returns
    -------
    kcen : np.ndarray
        k bin centers, shape (nk,)
    all_plk : np.ndarray
        Power spectrum multipoles, shape (nk, nell)
    shot_noise : float
        Estimated shot noise

    Raises
    ------
    ValueError
        If FFT computation fails
    """
    from pypower import CatalogFFTPower

    try:
        kedges = build_kedges(spec)
        if not np.all(np.diff(kedges) > 0):
            raise ValueError("k edges must be strictly increasing")

        # Use uniform dummy mu binning for multipole computation
        # (poles don't depend on mu binning; we just need edges for the FFT)
        dummy_mu_wedges = np.linspace(0.0, 1.0, 2)

        position_type = 'xyz' if spec.mock_type == 'quijote' else 'rdd'
        los = 'firstpoint' if spec.mock_type == 'halfdome' else 'z'

        # For halfdome, compute poles only; for quijote, poles are automatic
        if spec.mock_type == 'halfdome':
            result = CatalogFFTPower(
                data_positions1=catalog.positions_rdd,
                data_weights1=catalog.weights,
                randoms_positions1=rand_positions,
                randoms_weights1=rand_weights,
                nmesh=spec.nmesh,
                los=los,
                position_type=position_type,
                resampler='tsc',
                dtype='f8',
                ells=spec.ells,
                edges=kedges,
                interlacing=3,
                shotnoise=None,
                mpiroot=0,
            )
            plk = result.poles.get_power()
            kcen = result.poles.k
            shot_noise = result.poles.shotnoise
        else:
            result = CatalogFFTPower(
                data_positions1=catalog.positions_rdd,
                data_weights1=catalog.weights,
                randoms_positions1=rand_positions,
                randoms_weights1=rand_weights,
                nmesh=spec.nmesh,
                los=los,
                position_type=position_type,
                resampler='tsc',
                dtype='f8',
                ells=spec.ells,
                edges=(kedges, dummy_mu_wedges),
                interlacing=3,
                shotnoise=None,
                mpiroot=0,
            )
            plk = result.poles.power
            kcen = result.wedges.k[:, 0]
            shot_noise = result.poles.shotnoise

        # plk has natural shape from pypower: (nell, nk)
        if plk.shape[0] != len(spec.ells):
            raise ValueError(
                f"Pole array mismatch: expected ({len(spec.ells)}, nk), got {plk.shape}"
            )

        return kcen, plk, shot_noise

    except Exception as e:
        raise ValueError(f"Failed to compute raw multipoles: {e}")


def _reconstruct_pkmu_from_poles(
    all_plk: np.ndarray,
    ells: tuple,
    mu_wedges: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct P(k, mu) wedges from power spectrum multipoles P_ell(k).

    Fast afterburner to convert cached P_ell(k) to P(k, mu) for any mu binning.
    Uses inverse Legendre polynomial transformation.

    Parameters
    ----------
    all_plk : np.ndarray
        Power spectrum multipoles, shape (nell, nk)
    ells : tuple
        Multipole orders, length nell
    mu_wedges : np.ndarray
        Mu bin edges for wedge reconstruction

    Returns
    -------
    pkmu : np.ndarray
        P(k, mu) in wedges, shape (nk, nmu)

    Raises
    ------
    ValueError
        If reconstruction fails
    """
    try:
        if all_plk.ndim != 2:
            raise ValueError(f"all_plk must be 2D (nell, nk), got shape {all_plk.shape}")
        nell, nk = all_plk.shape
        if nell != len(ells):
            raise ValueError(
                f"all_plk has {nell} ell rows but expected {len(ells)} (ells={ells})"
            )
        if len(mu_wedges) < 2:
            raise ValueError(f"mu_wedges must have at least 2 edges, got {len(mu_wedges)}")

        # Transpose (nell, nk) -> (nk, nell) for _poles_to_wedges
        plk_kell = all_plk.T
        pkmu = _poles_to_wedges(ells, plk_kell, mu_wedges)
        if pkmu.shape != (nk, len(mu_wedges) - 1):
            raise ValueError(
                f"pkmu reconstruction failed: expected shape (nk={nk}, nmu={len(mu_wedges)-1}), got {pkmu.shape}"
            )
        return pkmu

    except Exception as e:
        raise ValueError(f"Failed to reconstruct P(k,mu) from poles: {e}")


def _compute_power_spectra(
    spec: ExperimentSpec,
    catalog: PreparedCatalog,
    rand_positions: np.ndarray,
    rand_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Orchestrate power spectrum computation with caching.

    Workflow:
    1. Compute P_ell(k) multipoles (independent of mu binning)
    2. Build mu wedges for target binning strategy
    3. Reconstruct P(k, mu) from cached P_ell(k)

    Returns 4 values: (kcen, pkmu, all_plk, shot_noise)

    Parameters
    ----------
    spec : ExperimentSpec
        Experiment specification
    catalog : PreparedCatalog
        Prepared galaxy catalog
    rand_positions : np.ndarray
        Random catalog positions
    rand_weights : np.ndarray
        Random catalog weights

    Returns
    -------
    kcen : np.ndarray
        k bin centers
    pkmu : np.ndarray
        P(k, mu) in wedges
    all_plk : np.ndarray
        P_ell(k) multipoles (cached for reuse)
    shot_noise : float
        Estimated shot noise
    """
    try:
        # 1) Compute raw multipoles P_ell(k)
        kcen, all_plk, shot_noise = _compute_raw_multipoles(
            spec, catalog, rand_positions, rand_weights
        )

        if kcen.shape[0] != all_plk.shape[1]:
            raise ValueError(
                f"kcen and all_plk mismatch: kcen.shape={kcen.shape}, all_plk.shape={all_plk.shape}"
            )

        # 2) Compute mu wedges for target binning strategy
        mu_wedges = build_mu_wedges(spec)
        if len(mu_wedges) < 2:
            raise ValueError(f"mu_wedges must have at least 2 edges, got {len(mu_wedges)}")

        # 3) Reconstruct P(k, mu) from cached P_ell(k)
        pkmu = _reconstruct_pkmu_from_poles(all_plk, spec.ells, mu_wedges)

        if pkmu.shape[0] != kcen.shape[0]:
            raise ValueError(
                f"pkmu and kcen mismatch: {pkmu.shape[0]} vs {kcen.shape[0]}"
            )

        return kcen, pkmu, all_plk, shot_noise

    except Exception as e:
        raise ValueError(f"Power spectrum computation failed: {e}")


def _downsample_to_nbar(spec: ExperimentSpec, catalog: PreparedCatalog, mock_idx: int) -> PreparedCatalog:
    """
    Randomly subsample the catalog so its effective comoving number density
    matches spec.target_nbar [(h/Mpc)^3].

    The comoving volume is computed from the observed radial range of the
    catalog (after redshift selection) using a full-sky spherical shell:
        V = (4π/3) * (r_max^3 - r_min^3)
    which is correct for a full-sky lightcone like Halfdome.
    If the mock covers a known fraction of the sky it will over-count V and
    thus under-downsample slightly — but for full-sky mocks this is exact.
    """
    n_current = catalog.positions_rdd.shape[1]
    r = catalog.base_r  # comoving distance in Mpc/h

    r_min = float(r.min())
    r_max = float(r.max())
    vol_full_sky = (4.0 * np.pi / 3.0) * (r_max ** 3 - r_min ** 3)  # (Mpc/h)^3

    n_target = int(spec.target_nbar * vol_full_sky)
    print(
        f'  nbar downsample: V={vol_full_sky:.3e} (Mpc/h)^3, '
        f'target N={n_target:,} from nbar={spec.target_nbar:.2e}, '
        f'current N={n_current:,}'
    )

    if n_target >= n_current:
        print(f'  -> target N >= current N, no downsampling applied.')
        return catalog

    seeds = _stage_seeds(spec, mock_idx)
    rng = np.random.default_rng(seeds['quijote'])  # reuse a deterministic seed slot
    keep = rng.choice(n_current, size=n_target, replace=False)
    keep.sort()

    catalog.positions_rdd = catalog.positions_rdd[:, keep]
    catalog.weights = catalog.weights[keep]
    catalog.base_r = catalog.base_r[keep]
    if catalog.base_redshifts is not None:
        catalog.base_redshifts = catalog.base_redshifts[keep]
    catalog.metadata['nbar_downsampled_to'] = n_target
    return catalog


def run_single_experiment(spec: ExperimentSpec, mock_idx: int, dm: Optional[desi_mock] = None) -> ExperimentResult:
    if dm is None:
        dm = desi_mock()

    if spec.mock_type == 'quijote':
        print("Preparing Quijote catalog...")
        catalog = _prepare_quijote_catalog(spec, mock_idx, dm)
    elif spec.mock_type == 'halfdome':
        print("Preparing Halfdome catalog...")
        catalog = _prepare_halfdome_catalog(spec, mock_idx, dm)
    else:
        raise NotImplementedError(
            f"Unsupported mock_type='{spec.mock_type}'. Supported options are 'quijote' and 'halfdome'."
        )

    if spec.redshift_sel:
        print("Applying redshift selection...")
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

    if spec.target_nbar is not None:
        print("Applying nbar downsampling...")
        catalog = _downsample_to_nbar(spec, catalog, mock_idx)

    if spec.gal_lat_cut_deg > 0.0:
        print("Applying galactic latitude cut...")
        catalog = _apply_gal_lat_cut(catalog, spec.gal_lat_cut_deg)

    # For Gaia stellar systematics, apply the extragalactic coverage mask EARLY.
    # This ensures all three contamination modes (none, additive, multiplicative) use
    # the same masked data for fair comparison. The mask is applied before contamination
    # to avoid dimensional mismatches.
    if spec.sys_spec_type == 'gaia_stellar':
        print("Applying Gaia stellar systematics mask...")
        gaia_mask = _load_or_compute_gaia_healpix_mask(mock_idx)
        nside = hp.npix2nside(len(gaia_mask))
        catalog = _apply_healpix_mask_to_catalog(catalog, gaia_mask, nside=nside)
        # Store the mask in metadata so randoms generation can apply it too
        catalog.metadata['gaia_coverage_mask'] = gaia_mask

    # Cache the original (pre-mask) redshift distribution so randoms maintain
    # constant comoving density regardless of which angular regions are masked.
    if catalog.base_redshifts is not None:
        catalog.metadata['original_base_redshifts'] = catalog.base_redshifts.copy()

    print('Applying contamination...')
    catalog = _apply_contamination(spec, catalog, mock_idx)

    if spec.mock_type == 'quijote':
        rand_positions, rand_weights = _build_random_catalog_periodic(spec, catalog)  # Pre-build randoms for periodic box to ensure consistency across modes
        print('Random positions (x,y,z) range:')
        print('  x: [{:.2f}, {:.2f}]'.format(rand_positions[0].min(), rand_positions[0].max()))
        print('  y: [{:.2f}, {:.2f}]'.format(rand_positions[1].min(), rand_positions[1].max()))
        print('  z: [{:.2f}, {:.2f}]'.format(rand_positions[2].min(), rand_positions[2].max()))

        print('rand weights are all ones:', np.all(rand_weights == 1.0))
    else:
        rand_positions, rand_weights = _build_random_catalog(spec, catalog)

    if spec.mock_type=='quijote':
        print('Catalog positions (x,y,z) range after contamination:')
        print('  x: [{:.2f}, {:.2f}]'.format(catalog.positions_rdd[0].min(), catalog.positions_rdd[0].max()))
        print('  y: [{:.2f}, {:.2f}]'.format(catalog.positions_rdd[1].min(), catalog.positions_rdd[1].max()))
        print('  z: [{:.2f}, {:.2f}]'.format(catalog.positions_rdd[2].min(), catalog.positions_rdd[2].max()))
        print('Catalog weights range after contamination: [{:.3e}, {:.3e}]'.format(catalog.weights.min(), catalog.weights.max()))
        kcen, pkmu, plk, shot_noise = _compute_power_spectra(spec, catalog, rand_positions, rand_weights)

    else:
        kcen, pkmu, plk, shot_noise = _compute_power_spectra(spec, catalog, rand_positions, rand_weights)

    # Compute P_ℓ(k) with lowest μ bin nulled for diagnostics
    mu_wedges = build_mu_wedges(spec)
    pkmu_null = pkmu.copy()
    pkmu_null[:, 0] = 0.0  # Null the lowest μ bin
    plk_null = _wedges_to_poles(spec.ells, pkmu_null, mu_wedges)

    return ExperimentResult(
        spec=spec,
        kcen=kcen,
        all_pkmu=pkmu[None, ...],
        all_plk=plk[None, ...],
        mu_wedges=mu_wedges,
        shot_noise=shot_noise,
        all_plk_null_lowest_mu=plk_null.T[None, ...],
        run_metadata={
            'mock_idx': mock_idx,
            'n_data': int(catalog.positions_rdd.shape[1]),
            'n_randoms': int(rand_positions.shape[1]),
            'contamination_mode': spec.contamination_mode,
        },
    )


def run_experiment_grid(spec: ExperimentSpec, nmock: int, dm: Optional[desi_mock] = None) -> ExperimentResult:
    if dm is None:
        dm = desi_mock()

    mu_wedges = build_mu_wedges(spec)
    kedges = build_kedges(spec)
    nkbin = len(kedges) - 1
    nwedge = len(mu_wedges) - 1
    all_pkmu = np.zeros((nmock, nkbin, nwedge), dtype=complex)
    all_plk = np.zeros((nmock, len(spec.ells), nkbin), dtype=complex)
    all_plk_null_lowest_mu = np.zeros((nmock, len(spec.ells), nkbin), dtype=complex)
    kcen = None
    run_records: List[Dict[str, Any]] = []

    for mock_idx in range(nmock):
        print(f'  [{mock_idx + 1}/{nmock}] Computing...')
        result = run_single_experiment(spec, mock_idx=mock_idx, dm=dm)
        all_pkmu[mock_idx] = result.all_pkmu[0]
        all_plk[mock_idx] = result.all_plk[0]
        if result.all_plk_null_lowest_mu is not None:
            all_plk_null_lowest_mu[mock_idx] = result.all_plk_null_lowest_mu[0]
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
        all_plk_null_lowest_mu=all_plk_null_lowest_mu,
        shot_noise=result.shot_noise,
        run_metadata={
            'records': run_records,
            'nmock': nmock,
            'kedges': kedges.tolist(),
            'label': build_run_label(spec),
        },
    )


def run_variant_collection(
    specs: List[ExperimentSpec],
    nmock: int,
    dm: Optional[desi_mock] = None,
) -> Dict[str, ExperimentResult]:
    results: Dict[str, ExperimentResult] = {}
    for spec in specs:
        result = run_experiment_grid(spec, nmock=nmock, dm=dm)
        results[build_run_label(spec)] = result
    return results


def _normalize_options(values: Any, *, name: str) -> Tuple[Any, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        return (values,)
    if np.isscalar(values):
        return (values,)
    return tuple(values)


def _coerce_bool_options(values: Any, *, name: str) -> Tuple[bool, ...]:
    normalized = _normalize_options(values, name=name)
    coerced: List[bool] = []
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
    with_rsd_options: Tuple[bool, ...] = (False, True),
    contamination_modes: Tuple[str, ...] = ('none', 'stellar', 'dust', 'both'),
    stellar_fracs: Tuple[float, ...] = (0.01,),
    sfd_stds: Tuple[float, ...] = (0.01,),
    quijote_geometry: str = 'full_cube',
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> List[ExperimentSpec]:
    base_kwargs = dict(base_kwargs or {})
    specs: List[ExperimentSpec] = []
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
    contamination_modes: Tuple[str, ...] = ('none', 'stellar', 'dust', 'both'),
    stellar_fracs: Tuple[float, ...] = (0.01,),
    sfd_stds: Tuple[float, ...] = (0.01,),
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> List[ExperimentSpec]:
    base_kwargs = dict(base_kwargs or {})
    specs: List[ExperimentSpec] = []
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
    contamination_modes: Tuple[str, ...] = ('none', 'stellar', 'dust', 'both'),
    stellar_fracs: Tuple[float, ...] = (0.01,),
    sfd_stds: Tuple[float, ...] = (0.01,),
    with_rsd_options: Tuple[bool, ...] = (False, True),
    quijote_geometry: str = 'full_cube',
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> List[ExperimentSpec]:
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
    save_dict = {
        'kcen': result.kcen,
        'all_pkmu': result.all_pkmu,
        'all_plk': result.all_plk,
        'ells': np.asarray(result.spec.ells),
        'mu_wedges': result.mu_wedges,
        'shot_noise': result.shot_noise,
        'run_config_json': json.dumps(_spec_to_jsonable(result.spec), sort_keys=True),
        'run_metadata_json': json.dumps(result.run_metadata, default=str, sort_keys=True),
        'run_label': label,
    }
    # Include nulled multipoles if available
    if result.all_plk_null_lowest_mu is not None:
        save_dict['all_plk_null_lowest_mu'] = result.all_plk_null_lowest_mu
    np.savez(output_path, **save_dict)
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
