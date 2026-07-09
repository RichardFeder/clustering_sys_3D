#!/usr/bin/env python3
"""Compute power spectrum multipoles of contaminant signal alone.

This script:
1. Generates pure contaminant realizations (no clean data mixed in)
2. Computes power spectra for each contaminant realization independently
3. Stores results for comparison with contaminated = clean + contaminant

Per-realization caching:
  - Individual realizations saved to: cache/config_{label}/contaminant_only_{sys_type}/contaminant_only_{idx}.npz
  - Aggregated results saved to: cache/config_{label}/contaminant_only_{sys_type}/contaminant_only_agg.npz
  - Metadata: cache/config_{label}/contaminant_only_{sys_type}/{config,rerun}.yaml/.sh
  - Seed cache: cache/config_{label}/contaminant_only_{sys_type}/seeds.json (for reproducibility)

Mock selection:
  - --run-mode all:        Recompute all realizations (default)
  - --run-mode remaining:  Only compute missing realizations
  - --run-mode {0,1,2}:    Compute specific realization indices

Usage:
  # Power-law contaminant with 10 realizations
  "$PYBIN" compute_contaminant_only_multipoles.py \\
    --base-config z0.1_0.4_nbar3e-04_nmesh128 \\
    --sys-type power_law \\
    --sys-amp 0.01 \\
    --n-realizations 10

  # Delta-function contaminant with 5 realizations
  "$PYBIN" compute_contaminant_only_multipoles.py \\
    --base-config z0.1_0.4_nbar3e-04_nmesh128 \\
    --sys-type delta_function \\
    --sys-ell-contam 4 \\
    --sys-amp 0.01 \\
    --n-realizations 5
"""

import argparse
import datetime as dt
import glob
import json
import logging
import os
import shlex
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    from pypower import setup_logging, mpi
except ImportError:
    setup_logging = None
    mpi = None

# Add parent dir to path for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    ExperimentSpec,
    run_single_experiment,
)
from desi_mocks import desi_mock

logger = logging.getLogger(__name__)


def _get_config_label(zmin: float, zmax: float, target_nbar: Optional[float], nmesh: int) -> str:
    """Generate unique label for this configuration."""
    nbar_str = f"nbar{target_nbar:.0e}".replace("+", "") if target_nbar else "nbarcat"
    return f"z{zmin:.1f}_{zmax:.1f}_{nbar_str}_nmesh{nmesh}"


def _get_config_cache_dir(cache_root: str, config_label: str, subtype: str = "contamination") -> str:
    """Get/create subdirectory for this configuration."""
    config_dir = os.path.join(cache_root, f"config_{config_label}", subtype)
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def _parse_mock_indices(text: str, n_real: int) -> List[int]:
    """Parse realization selection string: 'all', 'remaining', or comma-sep indices."""
    text = text.strip().lower()
    if text == "all":
        return list(range(n_real))
    if text == "remaining":
        return []  # Will be filled by checking existing files
    # Assume comma-separated or space-separated indices
    try:
        indices = [int(x.strip()) for x in text.replace(",", " ").split()]
        return indices
    except ValueError:
        raise ValueError(f"Invalid realization indices: {text}")


def _find_existing_realization_files(cache_dir: str, n_real: int) -> List[int]:
    """Find which realizations have already been computed."""
    computed = []
    for i in range(n_real):
        path = os.path.join(cache_dir, f"contaminant_only_{i}.npz")
        if os.path.exists(path):
            computed.append(i)
    return computed


def _select_realizations_to_run(run_mode: str, n_real: int, cache_dir: Optional[str]) -> List[int]:
    """Determine which realization indices to run."""
    if run_mode == "all":
        return list(range(n_real))
    elif run_mode == "remaining":
        if cache_dir is None or not os.path.exists(cache_dir):
            return list(range(n_real))
        computed = _find_existing_realization_files(cache_dir, n_real)
        remaining = [i for i in range(n_real) if i not in computed]
        logger.info(f"Found {len(computed)} existing realizations, will compute {len(remaining)} remaining")
        return remaining
    else:
        return _parse_mock_indices(run_mode, n_real)


def _generate_or_load_seeds(cache_dir: str, n_real: int, base_seed: int = 42) -> List[int]:
    """Generate or load seeds for contaminant realizations."""
    seeds_file = os.path.join(cache_dir, "seeds.json")
    
    if os.path.exists(seeds_file):
        with open(seeds_file, "r") as f:
            seeds_data = json.load(f)
        seeds = seeds_data.get("seeds", [])
        if len(seeds) >= n_real:
            logger.info(f"Loaded cached seeds from {seeds_file}")
            return seeds[:n_real]
    
    # Generate new seeds from base seed using RNG
    rng = np.random.RandomState(base_seed)
    seeds = [int(rng.randint(1, 2**31 - 1)) for _ in range(n_real)]
    
    # Save for reproducibility
    os.makedirs(cache_dir, exist_ok=True)
    seeds_data = {
        "n_realizations": n_real,
        "base_seed": base_seed,
        "generated_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "seeds": seeds,
    }
    with open(seeds_file, "w") as f:
        json.dump(seeds_data, f, indent=2)
    logger.info(f"Generated and saved {n_real} seeds to {seeds_file}")
    
    return seeds


def _make_label_for_systematic(sys_cfg: Dict[str, Any]) -> str:
    """Generate cache label based on systematic type."""
    sys_type = sys_cfg.get("type", "power_law")
    
    if sys_type == "power_law":
        alpha = sys_cfg.get("alpha", -2.0)
        amplitude = sys_cfg.get("amplitude", 0.01)
        return f"powerlaw_alpha{alpha:.1f}_amp{amplitude:.4f}"
    
    elif sys_type == "delta_function":
        ell_contam = sys_cfg.get("ell_contam")
        amplitude = sys_cfg.get("amplitude", 0.01)
        if isinstance(ell_contam, (list, tuple)):
            ell_str = "_".join(str(e) for e in ell_contam)
        else:
            ell_str = str(ell_contam)
        return f"deltafn_ellcontam{ell_str}_amp{amplitude:.4f}"
    
    else:
        raise ValueError(f"Unknown systematic type: {sys_type}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _yaml_block(obj: Any, indent: int = 0) -> Iterable[str]:
    prefix = " " * indent
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, (dict, list, tuple)):
                yield f"{prefix}{key}:"
                yield from _yaml_block(val, indent + 2)
            else:
                yield f"{prefix}{key}: {json.dumps(val)}"
        return
    if isinstance(obj, (list, tuple)):
        for val in obj:
            if isinstance(val, (dict, list, tuple)):
                yield f"{prefix}-"
                yield from _yaml_block(val, indent + 2)
            else:
                yield f"{prefix}- {json.dumps(val)}"
        return
    yield f"{prefix}{json.dumps(obj)}"


def _write_yaml(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in _yaml_block(_to_jsonable(payload)):
            f.write(line)
            f.write("\n")


def _write_rerun_script(path: str, command: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(command)
        f.write("\n")
    os.chmod(path, 0o755)


def _build_rerun_command(script_path: str, argv: Tuple[str, ...]) -> str:
    quoted = " ".join(shlex.quote(a) for a in argv)
    return f'source ~/.desi_bashrc && "$PYBIN" {shlex.quote(script_path)} {quoted}'.rstrip()


def plot_mock_density_healpix(
    mock_idx: int,
    config_cache_dir: str,
    zmin: float,
    zmax: float,
) -> None:
    """Generate a healpix density visualization for one mock realization.
    
    Parameters
    ----------
    mock_idx : int
        Mock index to visualize
    config_cache_dir : str
        Output directory for the config
    zmin, zmax : float
        Redshift range for mock
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        import healpy as hp
    except ImportError:
        logger.warning("healpy or matplotlib not available; skipping density visualization")
        return
    
    try:
        from utils import convert_to_ra_dec_distance
        
        # Load one halfdome mock realization
        dm = desi_mock()
        from pipeline import DEFAULT_HALFDOME_BASEDIR
        dm.halfdome_mock_basedir = DEFAULT_HALFDOME_BASEDIR
        galpos, redshift = dm.load_halfdome_mock(mock_idx, n_sample=500_000)
        
        # Convert to RA/Dec
        boxsize = 1000.0
        ra, dec, r = convert_to_ra_dec_distance(galpos, boxsize, center_offset_mpc=0.0)
        
        # Create healpix map
        nside = 128  # healpix resolution
        npix = hp.nside2npix(nside)
        
        # Convert RA/Dec to HEALPix pixel indices
        theta = np.radians(90.0 - dec)  # colatitude from Dec
        phi = np.radians(ra)             # azimuth from RA
        pix = hp.ang2pix(nside, theta, phi)
        
        # Count galaxies per pixel
        gal_density = np.zeros(npix)
        np.add.at(gal_density, pix, 1)
        
        # Normalize by pixel area to get surface density
        pixel_area = hp.nside2pixarea(nside, degrees=True)
        gal_density /= pixel_area
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        hp.mollview(gal_density, fig=fig, title=f"Mock #{mock_idx}: Galaxy Density (z={zmin}-{zmax})",
                   cmap="viridis", format="%.2e")
        
        # Save figure
        fig_dir = os.path.join(config_cache_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"galaxy_density_healpix_mock{mock_idx}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Saved density visualization: {fig_path}")
    except Exception as e:
        logger.warning(f"Failed to generate density visualization: {e}")


def _load_base_config(cache_root: str, config_label: str) -> Dict[str, Any]:
    """Load effective configuration from clean baseline metadata."""
    config_dir = os.path.join(cache_root, f"config_{config_label}", "clean_baseline")
    config_file = os.path.join(config_dir, "config.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Base config not found: {config_file}\n"
            f"Run compute_clean_baseline.py first with config matching: {config_label}"
        )
    
    if yaml is None:
        raise ImportError(f"PyYAML not available; cannot load {config_file}")
    
    with open(config_file, "r") as f:
        metadata = yaml.safe_load(f) or {}
    
    return metadata.get("effective_config", {})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute power spectrum multipoles of pure contaminant signal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config specification: either base-config OR explicit config file
    p.add_argument("--base-config", type=str, default=None, 
                   help="Base config label (e.g., 'z0.1_0.4_nbar3e-04_nmesh128') to auto-load parameters from clean baseline")
    p.add_argument("--config", type=str, default=None, 
                   help="Path to YAML config file (alternative to --base-config)")
    
    # Redshift and mesh (only if not using --base-config)
    p.add_argument("--zmin", type=float, default=None, help="Redshift lower bound (use --base-config to auto-load)")
    p.add_argument("--zmax", type=float, default=None, help="Redshift upper bound (use --base-config to auto-load)")
    p.add_argument("--nmesh", type=int, default=None, help="Mesh resolution (use --base-config to auto-load)")
    p.add_argument("--target-nbar", type=float, default=None, help="Target number density (use --base-config to auto-load)")
    p.add_argument("--k-min", type=float, default=None, help="Minimum k (use --base-config to auto-load)")
    p.add_argument("--k-max", type=float, default=None, help="Maximum k (use --base-config to auto-load)")
    p.add_argument("--delta-k", type=float, default=None, help="k-bin width (use --base-config to auto-load)")
    
    # Number of contaminant realizations
    p.add_argument("--n-realizations", type=int, default=5, 
                   help="Number of independent contaminant realizations to generate")
    p.add_argument("--base-seed", type=int, default=42, 
                   help="Base seed for generating realization seeds (cached for reproducibility)")
    
    # Systematic type and parameters
    p.add_argument("--sys-type", choices=["power_law", "delta_function"], default=None, 
                   help="Systematic type (overrides config)")
    p.add_argument("--sys-amp", type=float, default=None, 
                   help="Systematic amplitude (overrides config)")
    p.add_argument("--sys-alpha", type=float, default=None, 
                   help="Power-law slope (overrides config)")
    p.add_argument("--sys-ell-contam", type=int, nargs="+", default=None, 
                   help="Delta-function multipole(s) (overrides config)")
    p.add_argument("--sys-ell-min", type=int, default=None, 
                   help="Minimum ell for power-law (overrides config)")
    p.add_argument("--sys-ell-max", type=int, default=None, 
                   help="Maximum ell for power-law (overrides config)")
    p.add_argument("--sys-ell-delta", type=int, default=None, 
                   help="Delta ell for power-law (overrides config)")
    
    p.add_argument("--run-mode", type=str, default="all",
                   help="Realization selection: 'all' (default), 'remaining', or comma-sep indices e.g. '0,2,3'")
    
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (default: cache/)")
    p.add_argument("--force", action="store_true", help="Recompute even if file exists")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def main() -> None:
    # Setup logging
    if setup_logging is not None:
        setup_logging()
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Handle MPI: only rank 0 computes, all ranks wait
    mpicomm = None
    if mpi is not None:
        mpicomm = mpi.COMM_WORLD
        if mpicomm.rank != 0:
            logger.setLevel(logging.WARNING)  # Suppress logging on non-rank-0
    
    # Determine cache root
    cache_root = args.output_dir or os.path.join(os.path.dirname(__file__), "cache")
    
    # Load configuration
    if args.base_config:
        # Load from clean baseline metadata
        logger.info(f"Loading base config from: {args.base_config}")
        base_cfg = _load_base_config(cache_root, args.base_config)
        
        # Extract parameters from base config
        redshift_cfg = base_cfg.get("redshift", {})
        mesh_cfg = base_cfg.get("mesh", {})
        mocks_cfg = base_cfg.get("mocks", {})
        k_cfg = base_cfg.get("k_range", {})
        
        zmin = redshift_cfg.get("zmin", 0.1)
        zmax = redshift_cfg.get("zmax", 0.4)
        nmesh = mesh_cfg.get("nmesh", 512)
        target_nbar = mocks_cfg.get("target_nbar")
        k_min = k_cfg.get("k_min", 0.006)
        k_max = k_cfg.get("k_max", 0.2)
        delta_k = k_cfg.get("delta_k", 0.01)
        
        # Allow CLI override for redshift/mesh only if explicitly provided
        if args.zmin is not None:
            zmin = args.zmin
        if args.zmax is not None:
            zmax = args.zmax
        if args.nmesh is not None:
            nmesh = args.nmesh
        if args.target_nbar is not None:
            target_nbar = args.target_nbar
        if args.k_min is not None:
            k_min = args.k_min
        if args.k_max is not None:
            k_max = args.k_max
        if args.delta_k is not None:
            delta_k = args.delta_k
        
        config = base_cfg
    else:
        # Load from YAML config file or use CLI parameters
        config_path = args.config
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for candidate in ["config_contaminant_powerlaw.yaml", "config_contaminant_deltafn.yaml"]:
                candidate_path = os.path.join(script_dir, candidate)
                if os.path.exists(candidate_path):
                    config_path = candidate_path
                    break
        
        if config_path and os.path.exists(config_path):
            if yaml is None:
                raise ImportError(f"PyYAML not available; cannot load {config_path}")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        redshift_cfg = config.get("redshift", {})
        k_cfg = config.get("k_range", {})
        
        # Extract values from config with defaults
        zmin = args.zmin if args.zmin is not None else redshift_cfg.get("zmin", 0.1)
        zmax = args.zmax if args.zmax is not None else redshift_cfg.get("zmax", 0.4)
        nmesh = args.nmesh if args.nmesh is not None else 512
        target_nbar = args.target_nbar if args.target_nbar is not None else redshift_cfg.get("target_nbar")
        k_min = args.k_min if args.k_min is not None else k_cfg.get("k_min", 0.006)
        k_max = args.k_max if args.k_max is not None else k_cfg.get("k_max", 0.2)
        delta_k = args.delta_k if args.delta_k is not None else k_cfg.get("delta_k", 0.01)
    
    # Extract systematic config from loaded config
    sys_cfg = config.get("systematic", {})
    
    # Build effective systematic config
    sys_type = args.sys_type if args.sys_type is not None else sys_cfg.get("type", "power_law")
    sys_amp = args.sys_amp if args.sys_amp is not None else sys_cfg.get("amplitude", 0.01)
    sys_mode = "transverse_additive"  # For contaminant-only, always use additive
    mean_conserving = sys_cfg.get("mean_conserving", True)
    
    # Systematic-type-specific parameters
    if sys_type == "power_law":
        sys_alpha = args.sys_alpha if args.sys_alpha is not None else sys_cfg.get("alpha", -2.0)
        # For Halfdome, don't apply Quijote defaults; only use if explicitly specified
        sys_ell_min = args.sys_ell_min if args.sys_ell_min is not None else sys_cfg.get("ell_min")
        sys_ell_max = args.sys_ell_max if args.sys_ell_max is not None else sys_cfg.get("ell_max")
        sys_ell_delta = args.sys_ell_delta if args.sys_ell_delta is not None else sys_cfg.get("ell_delta")
        ell_contam = None
    elif sys_type == "delta_function":
        # For delta function, ell_min/max are not used
        sys_alpha = None
        sys_ell_min = None
        sys_ell_max = None
        sys_ell_delta = None
        ell_contam = args.sys_ell_contam if args.sys_ell_contam else sys_cfg.get("ell_contam")
        if not ell_contam:
            raise ValueError("Must specify --sys-ell-contam for delta_function systematic")
        if isinstance(ell_contam, int):
            ell_contam = [ell_contam]
    else:
        raise ValueError(f"Unknown systematic type: {sys_type}")
    
    logger.info(f"Configuration: z=[{zmin}, {zmax}], nmesh={nmesh}, n_realizations={args.n_realizations}")
    logger.info(f"k-range: k_min={k_min}, k_max={k_max}, delta_k={delta_k}")
    logger.info(f"Systematic type: {sys_type}")
    if sys_type == "power_law":
        ell_range_str = f"[{sys_ell_min}, {sys_ell_max}]" if sys_ell_min is not None and sys_ell_max is not None else "(not specified, using defaults)"
        logger.info(f"  alpha={sys_alpha}, amplitude={sys_amp}, ell_range={ell_range_str}")
    else:
        logger.info(f"  ell_contam={ell_contam}, amplitude={sys_amp}")
    if target_nbar is not None:
        logger.info(f"  target_nbar={target_nbar}")
    
    # Generate or load cached seeds for reproducibility
    config_label = args.base_config if args.base_config else _get_config_label(zmin, zmax, target_nbar, nmesh)
    sys_label = _make_label_for_systematic({
        "type": sys_type,
        "alpha": sys_alpha,
        "amplitude": sys_amp,
        "ell_contam": ell_contam,
    })
    
    config_cache_dir = _get_config_cache_dir(cache_root, config_label, f"contaminant_only_{sys_label}")
    seeds = _generate_or_load_seeds(config_cache_dir, args.n_realizations, args.base_seed)
    
    logger.info(f"Config label: {config_label}")
    logger.info(f"Systematic label: {sys_label}")
    logger.info(f"Cache directory: {config_cache_dir}")
    logger.info(f"Using seeds: {seeds}")
    
    # Determine which realizations to run
    realizations_to_run = _select_realizations_to_run(args.run_mode, args.n_realizations, 
                                                      config_cache_dir if args.run_mode == "remaining" else None)
    logger.info(f"Will compute {len(realizations_to_run)} realizations: {realizations_to_run}")
    
    # Define ells once from spec (will be same for all realizations)
    ells = (0, 2, 4, 6, 8, 10, 12, 14, 16)  # Standard DESI multipoles
    
    # MPI: only rank 0 computes
    if mpicomm is None or mpicomm.rank == 0:
        all_plk_list = []
        kcen = None
        
        for real_idx in realizations_to_run:
            real_npz_path = os.path.join(config_cache_dir, f"contaminant_only_{real_idx}.npz")
            
            if os.path.exists(real_npz_path) and not args.force:
                logger.info(f"[{real_idx}] Using cached: {real_npz_path}")
                with np.load(real_npz_path, allow_pickle=False) as dat:
                    plk = np.asarray(dat["all_plk"])
                    if ells is None:
                        ells = tuple(int(v) for v in np.atleast_1d(dat.get("ells", [])).tolist())
                    if kcen is None:
                        kcen = np.asarray(dat.get("kcen"))
                    all_plk_list.append(plk)
                continue
            
            logger.info(f"[{real_idx}] Computing contaminant realization {real_idx}/{args.n_realizations-1}")
            try:
                seed_for_real = seeds[real_idx]
                
                # Build spec for contaminant-only: uniform Poisson base + applied contamination
                # The generate_uniform_catalog flag ensures we start with constant nbar,
                # no large-scale structure, just Poisson shot noise
                spec = ExperimentSpec(
                    mock_type="halfdome",
                    with_rsd=False,
                    contamination_mode=sys_mode,
                    generate_uniform_catalog=True,  # Use uniform Poisson catalog
                    redshift_sel=True,
                    zmin=zmin,
                    zmax=zmax,
                    k_min=k_min,
                    k_max=k_max,
                    delta_k=delta_k,
                    target_nbar=1e-4,
                    nmesh=nmesh,
                    ells=(0, 2, 4, 6, 8, 10, 12, 14, 16),
                    n_clean_bins=8,
                    mu_binning_strategy="nonuniform",
                    sys_amp=sys_amp,
                    sys_spec_type="delta" if sys_type == "delta_function" else "power_law",
                    sys_ell_min=sys_ell_min,
                    sys_ell_max=sys_ell_max,
                    sys_ell_delta=sys_ell_delta if sys_type == "power_law" else ell_contam[0] if ell_contam else None,
                    sys_amp_mode="rms",
                    sys_amp_mult_scale=1.0,
                    seed=seed_for_real,
                    save_dir=config_cache_dir,
                    output_name=f"contaminant_only",
                    mean_conserving_additive=mean_conserving,
                )
                
                # Run experiment - generates uniform catalog + applies contamination
                result = run_single_experiment(spec, real_idx)
                plk = result.all_plk[0]  # Shape: (nell, nk)
                
                if kcen is None:
                    kcen = np.asarray(result.kcen)
                
                # Save per-realization result
                np.savez(real_npz_path, all_plk=plk, ells=np.asarray(ells), kcen=np.asarray(kcen))
                all_plk_list.append(plk)
                logger.info(f"[{real_idx}] Saved: {real_npz_path}")
            
            except Exception as e:
                logger.error(f"[{real_idx}] Failed: {e}")
                raise
        
        # Aggregate results
        if all_plk_list and ells is not None:
            all_plk_agg = np.stack(all_plk_list, axis=0)  # Shape: (n_real, nell, nk)
            agg_path = os.path.join(config_cache_dir, "contaminant_only_agg.npz")
            
            np.savez(agg_path, all_plk=all_plk_agg, ells=np.asarray(ells), kcen=kcen)
            logger.info(f"Aggregated {len(all_plk_list)} realizations -> {agg_path}")
            logger.info(f"Aggregated shape: {all_plk_agg.shape}")
        
        # Generate density visualization for first computed realization
        if realizations_to_run:
            plot_mock_density_healpix(realizations_to_run[0], config_cache_dir, zmin, zmax)
        
        # Save metadata
        yaml_path = os.path.join(config_cache_dir, "config.yaml")
        rerun_path = os.path.join(config_cache_dir, "rerun.sh")
        
        rerun_cmd = _build_rerun_command(os.path.abspath(__file__), tuple(sys.argv[1:]))
        
        metadata = {
            "kind": f"contaminant_only_multipoles_{sys_type}",
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "config_label": config_label,
            "systematic_label": sys_label,
            "config_cache_dir": config_cache_dir,
            "seeds_file": os.path.join(config_cache_dir, "seeds.json"),
            "rerun_script": rerun_path,
            "rerun_command": rerun_cmd,
            "config_file": args.config or os.path.join(os.path.dirname(__file__), f"config_contaminant_{sys_type}.yaml"),
            "loaded_config": _to_jsonable(config),
            "contamination": {
                "mode": sys_mode,
                "systematic_type": sys_type,
                "amplitude": sys_amp,
                "mean_conserving": mean_conserving,
                "base_signal": "uniform (contaminant-only, no clean mock)",
            },
            "systematic_parameters": (
                {
                    "type": "power_law",
                    "alpha": sys_alpha,
                    "ell_min": sys_ell_min,
                    "ell_max": sys_ell_max,
                    "ell_delta": sys_ell_delta,
                } if sys_type == "power_law" else {
                    "type": "delta_function",
                    "ell_contam": ell_contam,
                }
            ),
            "redshift_range": {
                "zmin": zmin,
                "zmax": zmax,
            },
            "k_range": {
                "k_min": k_min,
                "k_max": k_max,
                "delta_k": delta_k,
            },
            "ells": list(ells) if ells else [],
            "n_realizations": args.n_realizations,
            "realizations_computed": realizations_to_run,
            "base_seed": args.base_seed,
            "realization_seeds": seeds,
            "runtime": {
                "python_executable": sys.executable,
                "cwd": os.getcwd(),
                "environment_setup": "source ~/.desi_bashrc",
                "python_var": "$PYBIN",
            },
            "notes": "Pure contaminant signal (no clean mock mixed in). Compare with clean_baseline and contaminated multipoles to study systematic effects.",
        }
        
        _write_yaml(yaml_path, metadata)
        _write_rerun_script(rerun_path, rerun_cmd)
        
        logger.info(f"Saved metadata: {yaml_path}")
        logger.info(f"Saved rerun script: {rerun_path}")
    
    # MPI: barrier to synchronize all ranks before exit
    if mpicomm is not None:
        mpicomm.Barrier()


if __name__ == "__main__":
    main()
