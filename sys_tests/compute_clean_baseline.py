#!/usr/bin/env python3
"""Compute clean (uncontaminated) baseline multipoles for systematic testing.

This script computes P_ell(k) for a standard configuration once, caching
the expensive FFT/multipole calculation for reuse by contamination variants.

Loads configuration from YAML file (config_clean_baseline.yaml by default).

Per-realization caching:
  - Individual mocks saved to: cache/config_{label}/clean_baseline/multipoles_mock_{idx}.npz
  - Aggregated results saved to: cache/config_{label}/clean_baseline/multipoles_agg.npz
  - Metadata: cache/config_{label}/clean_baseline/{config,rerun}.yaml/.sh

Mock selection:
  - --run-mode all:        Recompute all mocks (default)
  - --run-mode remaining:  Only compute missing mocks
  - --run-mode {0,1,2}:    Compute specific mock indices
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

# Environment setup
import multiprocessing

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
    build_run_label,
    run_experiment_grid,
    run_single_experiment,
    save_experiment_result,
)
from desi_mocks import desi_mock
from utils import convert_to_ra_dec_distance

logger = logging.getLogger(__name__)


def _get_config_label(zmin: float, zmax: float, target_nbar: Optional[float], nmesh: int) -> str:
    """Generate unique label for this configuration."""
    nbar_str = f"nbar{target_nbar:.0e}".replace("+", "") if target_nbar else "nbarcat"
    return f"z{zmin:.1f}_{zmax:.1f}_{nbar_str}_nmesh{nmesh}"


def _get_config_cache_dir(cache_root: str, config_label: str, subtype: str = "clean_baseline") -> str:
    """Get/create subdirectory for this configuration."""
    config_dir = os.path.join(cache_root, f"config_{config_label}", subtype)
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def _parse_mock_indices(text: str, nmock: int) -> List[int]:
    """Parse mock selection string: 'all', 'remaining', or comma-sep indices."""
    text = text.strip().lower()
    if text == "all":
        return list(range(nmock))
    if text == "remaining":
        return []  # Will be filled by checking existing files
    # Assume comma-separated or space-separated indices
    try:
        indices = [int(x.strip()) for x in text.replace(",", " ").split()]
        return indices
    except ValueError:
        raise ValueError(f"Invalid mock indices: {text}")


def _find_existing_mock_files(cache_dir: str, nmock: int) -> List[int]:
    """Find which mocks have already been computed."""
    computed = []
    for i in range(nmock):
        path = os.path.join(cache_dir, f"multipoles_mock_{i}.npz")
        if os.path.exists(path):
            computed.append(i)
    return computed


def _select_mocks_to_run(run_mode: str, nmock: int, cache_dir: Optional[str]) -> List[int]:
    """Determine which mock indices to run."""
    if run_mode == "all":
        return list(range(nmock))
    elif run_mode == "remaining":
        if cache_dir is None or not os.path.exists(cache_dir):
            return list(range(nmock))
        computed = _find_existing_mock_files(cache_dir, nmock)
        remaining = [i for i in range(nmock) if i not in computed]
        logger.info(f"Found {len(computed)} existing mocks, will compute {len(remaining)} remaining")
        return remaining
    else:
        return _parse_mock_indices(run_mode, nmock)



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


def _load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config_clean_baseline.yaml")
    
    if not os.path.exists(config_path):
        return {}
    
    if yaml is None:
        raise ImportError(f"PyYAML not available; cannot load {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute clean baseline multipoles (loads from config_clean_baseline.yaml by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--zmin", type=float, default=None, help="Redshift lower bound (overrides config)")
    p.add_argument("--zmax", type=float, default=None, help="Redshift upper bound (overrides config)")
    p.add_argument("--nmock", type=int, default=None, help="Number of mocks (overrides config)")
    p.add_argument("--nmesh", type=int, default=None, help="Mesh resolution (overrides config)")
    p.add_argument("--target-nbar", type=float, default=None, help="Target number density (overrides config)")
    
    p.add_argument("--run-mode", type=str, default="all", 
                   help="Mock selection: 'all' (default), 'remaining', or comma-sep indices e.g. '0,2,3'")
    
    p.add_argument("--omp-num-threads", type=int, default=None, 
                   help="Number of OpenMP threads (sets OMP_NUM_THREADS env var; defaults to cpu count)")
    
    p.add_argument("--force", action="store_true", help="Recompute even if cached")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def main() -> None:
    # Setup MPI communicator early
    if mpi is not None:
        mpicomm = mpi.COMM_WORLD
    else:
        mpicomm = None
    
    # Setup logging
    if setup_logging is not None:
        setup_logging()
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Only print MPI info on rank 0
    if mpicomm is not None and mpicomm.rank == 0:
        logger.info(f"MPI initialized: rank 0/{mpicomm.size-1}")
    elif mpicomm is not None:
        # Other ranks stay silent on non-critical messages
        if mpicomm.rank != 0:
            logger.setLevel(logging.WARNING)
    
    args = parse_args()
    
    # Setup OpenMP threads
    if args.omp_num_threads is not None:
        omp_threads = args.omp_num_threads
    else:
        omp_threads = multiprocessing.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    if mpicomm is None or mpicomm.rank == 0:
        logger.info(f"OMP_NUM_THREADS={omp_threads}")
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load YAML config
    config = _load_yaml_config(args.config)
    
    # Extract values from config with defaults
    redshift_cfg = config.get("redshift", {})
    mesh_cfg = config.get("mesh", {})
    multipoles_cfg = config.get("multipoles", {})
    k_cfg = config.get("k_range", {})
    binning_cfg = config.get("binning", {})
    mocks_cfg = config.get("mocks", {})
    output_cfg = config.get("output", {})
    
    # Apply CLI overrides
    zmin = args.zmin if args.zmin is not None else redshift_cfg.get("zmin", 0.1)
    zmax = args.zmax if args.zmax is not None else redshift_cfg.get("zmax", 0.4)
    nmock = args.nmock if args.nmock is not None else mocks_cfg.get("nmock", 5)
    nmesh = args.nmesh if args.nmesh is not None else mesh_cfg.get("nmesh", 512)
    target_nbar = args.target_nbar if args.target_nbar is not None else mocks_cfg.get("target_nbar")
    
    ells = tuple(multipoles_cfg.get("ells", [0, 2, 4, 6, 8, 10, 12, 14, 16]))
    k_min = k_cfg.get("k_min", 0.006)
    k_max = k_cfg.get("k_max", 0.2)
    delta_k = k_cfg.get("delta_k", 0.01)
    n_clean_bins = binning_cfg.get("n_clean_bins", 8)
    mu_binning_strategy = binning_cfg.get("mu_binning_strategy", "nonuniform")
    n_sample = mocks_cfg.get("n_sample", 20_000_000)
    ds_fac = mocks_cfg.get("ds_fac", 5)
    seed = mocks_cfg.get("seed", 42)
    with_rsd = mocks_cfg.get("with_rsd", False)
    
    cache_root = output_cfg.get("save_dir", "cache")
    cache_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_root)
    verbose = args.verbose or output_cfg.get("verbose", False)
    
    # Generate config label
    config_label = _get_config_label(zmin, zmax, target_nbar, nmesh)
    config_cache_dir = _get_config_cache_dir(cache_root, config_label, "clean_baseline")
    
    logger.info(f"Configuration: z=[{zmin}, {zmax}], nmesh={nmesh}, nmock={nmock}")
    logger.info(f"Config label: {config_label}")
    logger.info(f"Cache directory: {config_cache_dir}")
    if target_nbar is not None:
        logger.info(f"Target nbar: {target_nbar}")
    
    # Determine which mocks to run
    mocks_to_run = _select_mocks_to_run(args.run_mode, nmock, config_cache_dir if args.run_mode == "remaining" else None)
    if mpicomm is None or mpicomm.rank == 0:
        logger.info(f"Will compute {len(mocks_to_run)} mocks: {mocks_to_run}")
    
    # Build spec for mock runs
    spec = ExperimentSpec(
        mock_type="halfdome",
        with_rsd=with_rsd,
        contamination_mode="none",  # ALWAYS clean
        redshift_sel=True,
        zmin=zmin,
        zmax=zmax,
        k_min=k_min,
        k_max=k_max,
        delta_k=delta_k,
        n_sample=n_sample,
        target_nbar=target_nbar,
        ds_fac=ds_fac,
        nmesh=nmesh,
        ells=ells,
        n_clean_bins=n_clean_bins,
        mu_binning_strategy=mu_binning_strategy,
        sys_amp=0.0,  # No systematic
        sys_spec_type="power_law",
        sys_ell_min=6,
        sys_ell_max=64,
        sys_ell_delta=None,
        sys_amp_mode="rms",
        sys_amp_mult_scale=1.0,
        seed=seed,
        save_dir=config_cache_dir,
        output_name=f"baseline",
    )
    
    # Compute per-mock results (sequential loop on rank 0, all ranks participate in computations)
    all_plk_list = []
    
    # Only rank 0 executes the loop, but all ranks participate in pypower calculations
    if mpicomm is None or mpicomm.rank == 0:
        for mock_idx in mocks_to_run:
            mock_npz_path = os.path.join(config_cache_dir, f"multipoles_mock_{mock_idx}.npz")
            
            if os.path.exists(mock_npz_path) and not args.force:
                logger.info(f"[{mock_idx}] Using cached: {mock_npz_path}")
                with np.load(mock_npz_path, allow_pickle=False) as dat:
                    plk = np.asarray(dat["all_plk"])
                    all_plk_list.append(plk)
                continue
            
            logger.info(f"[{mock_idx}] Computing mock {mock_idx}/{nmock-1}")
            try:
                result = run_single_experiment(spec, mock_idx)
                plk = result.all_plk[0]  # Shape: (nell, nk)
                
                # Save per-mock result
                np.savez(mock_npz_path, all_plk=plk, ells=np.asarray(ells), kcen=np.asarray(result.kcen))
                all_plk_list.append(plk)
                logger.info(f"[{mock_idx}] Saved: {mock_npz_path}")
            except Exception as e:
                logger.error(f"[{mock_idx}] Failed: {e}")
                raise
    
    # Synchronize all ranks before aggregation (rank 0 waits for others if needed)
    if mpicomm is not None and hasattr(mpicomm, 'Barrier'):
        mpicomm.Barrier()
    
    # Aggregate results on rank 0 only
    if mpicomm is None or mpicomm.rank == 0:
        logger.info("Aggregating results...")
        all_plk_agg_list = []
        for mock_idx in mocks_to_run:
            mock_npz_path = os.path.join(config_cache_dir, f"multipoles_mock_{mock_idx}.npz")
            if os.path.exists(mock_npz_path):
                with np.load(mock_npz_path, allow_pickle=False) as dat:
                    plk = np.asarray(dat["all_plk"])
                    all_plk_agg_list.append(plk)
            else:
                logger.warning(f"Mock {mock_idx} file not found: {mock_npz_path}")
        
        if all_plk_agg_list:
            all_plk_agg = np.stack(all_plk_agg_list, axis=0)  # Shape: (nmock_computed, nell, nk)
            agg_path = os.path.join(config_cache_dir, "multipoles_agg.npz")
            
            # Get kcen from any computed mock
            kcen = None
            first_mock_path = os.path.join(config_cache_dir, f"multipoles_mock_{mocks_to_run[0]}.npz")
            if os.path.exists(first_mock_path):
                with np.load(first_mock_path, allow_pickle=False) as dat:
                    kcen = np.asarray(dat.get("kcen"))
            
            np.savez(agg_path, all_plk=all_plk_agg, ells=np.asarray(ells), kcen=kcen)
            logger.info(f"Aggregated {len(all_plk_agg_list)} mocks -> {agg_path}")
            logger.info(f"Aggregated shape: {all_plk_agg.shape}")
        
        # Generate density visualization for first computed mock
        if mocks_to_run:
            plot_mock_density_healpix(mocks_to_run[0], config_cache_dir, zmin, zmax)
        
        # Save metadata
        yaml_path = os.path.join(config_cache_dir, "config.yaml")
        rerun_path = os.path.join(config_cache_dir, "rerun.sh")
        
        rerun_cmd = _build_rerun_command(os.path.abspath(__file__), tuple(sys.argv[1:]))
        
        metadata = {
            "kind": "clean_baseline_multipoles",
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "config_label": config_label,
            "config_cache_dir": config_cache_dir,
            "rerun_script": rerun_path,
            "rerun_command": rerun_cmd,
            "config_file": args.config or os.path.join(os.path.dirname(__file__), "config_clean_baseline.yaml"),
            "loaded_config": _to_jsonable(config),
            "effective_config": {
                "redshift": {"zmin": zmin, "zmax": zmax},
                "mesh": {"nmesh": nmesh},
                "multipoles": {"ells": list(ells)},
                "k_range": {"k_min": k_min, "k_max": k_max, "delta_k": delta_k},
                "binning": {"n_clean_bins": n_clean_bins},
                "mocks": {"nmock": nmock, "seed": seed, "with_rsd": with_rsd, "target_nbar": target_nbar},
            },
            "mocks_computed": mocks_to_run,
            "mpi_info": {
                "mpi_enabled": mpicomm is not None,
                "mpi_size": mpicomm.size if mpicomm is not None else 1,
            },
            "runtime": {
                "python_executable": sys.executable,
                "cwd": os.getcwd(),
                "environment_setup": "source ~/.desi_bashrc",
                "python_var": "$PYBIN",
            },
        }
        
        _write_yaml(yaml_path, metadata)
        _write_rerun_script(rerun_path, rerun_cmd)
        
        logger.info(f"Saved metadata: {yaml_path}")
        logger.info(f"Saved rerun script: {rerun_path}")


if __name__ == "__main__":
    main()
