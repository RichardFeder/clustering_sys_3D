#!/usr/bin/env python3
"""Compute contaminated multipoles using cached clean baseline.

This script:
1. Loads the clean baseline multipoles (from compute_clean_baseline.py)
2. Runs the experiment with contamination (power-law or delta-function)
3. Saves contaminated multipoles alongside the clean baseline for comparison

Per-realization caching:
  - Individual mocks saved to: cache/config_{label}/contamination_{sys_type}/multipoles_mock_{idx}.npz
  - Aggregated results saved to: cache/config_{label}/contamination_{sys_type}/multipoles_agg.npz
  - Metadata: cache/config_{label}/contamination_{sys_type}/{config,rerun}.yaml/.sh

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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    from pypower import setup_logging
except ImportError:
    setup_logging = None

# Add parent dir to path for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    ExperimentSpec,
    build_run_label,
    run_experiment_grid,
    run_single_experiment,
    save_experiment_result,
)

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


def _find_clean_baseline(zmin: float, zmax: float) -> Optional[str]:
    """Find clean baseline file in cache by redshift range."""
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    pattern = f"clean_baseline_z{zmin:.1f}_{zmax:.1f}.npz"
    matches = glob.glob(os.path.join(cache_dir, pattern))
    if matches:
        return matches[0]
    return None


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
    
    # Extract the effective config that was used for the clean baseline
    return metadata.get("effective_config", {})


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute contaminated multipoles matching a base configuration",
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
    p.add_argument("--nmock", type=int, default=None, help="Number of mocks (use --base-config to auto-load)")
    p.add_argument("--target-nbar", type=float, default=None, help="Target number density (use --base-config to auto-load)")
    
    # Clean baseline file
    p.add_argument("--clean-baseline-file", type=str, default=None, 
                   help="Path to clean baseline NPZ (auto-finds if not specified)")
    
    # Systematic type and parameters (these are contaminant-specific)
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
                   help="Mock selection: 'all' (default), 'remaining', or comma-sep indices e.g. '0,2,3'")
    
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
        
        zmin = redshift_cfg.get("zmin", 0.1)
        zmax = redshift_cfg.get("zmax", 0.4)
        nmesh = mesh_cfg.get("nmesh", 512)
        nmock = mocks_cfg.get("nmock", 5)
        target_nbar = mocks_cfg.get("target_nbar")
        
        # Allow CLI override for redshift/mesh only if explicitly provided
        if args.zmin is not None:
            zmin = args.zmin
        if args.zmax is not None:
            zmax = args.zmax
        if args.nmesh is not None:
            nmesh = args.nmesh
        if args.nmock is not None:
            nmock = args.nmock
        if args.target_nbar is not None:
            target_nbar = args.target_nbar
        
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
        mocks_cfg = config.get("mocks", {})
        
        # Extract values from config with defaults
        zmin = args.zmin if args.zmin is not None else redshift_cfg.get("zmin", 0.1)
        zmax = args.zmax if args.zmax is not None else redshift_cfg.get("zmax", 0.4)
        nmock = args.nmock if args.nmock is not None else mocks_cfg.get("nmock", 5)
        nmesh = args.nmesh if args.nmesh is not None else 512
        target_nbar = args.target_nbar if args.target_nbar is not None else mocks_cfg.get("target_nbar")
    
    # Extract systematic config from loaded config
    sys_cfg = config.get("systematic", {})
    
    # Build effective systematic config
    sys_type = args.sys_type if args.sys_type is not None else sys_cfg.get("type", "power_law")
    sys_amp = args.sys_amp if args.sys_amp is not None else sys_cfg.get("amplitude", 0.01)
    sys_mode = sys_cfg.get("mode", "transverse_additive")
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
    
    # Find or verify clean baseline
    if args.clean_baseline_file:
        clean_baseline_file = os.path.abspath(args.clean_baseline_file)
        if not os.path.exists(clean_baseline_file):
            raise FileNotFoundError(f"Clean baseline not found: {clean_baseline_file}")
    elif args.base_config:
        # Auto-find from base config location
        config_label = args.base_config
        clean_baseline_file = os.path.join(cache_root, f"config_{config_label}", "clean_baseline", "multipoles_agg.npz")
        if not os.path.exists(clean_baseline_file):
            raise FileNotFoundError(
                f"Clean baseline not found: {clean_baseline_file}\n"
                f"Run compute_clean_baseline.py first with --config matching this setup."
            )
    else:
        clean_baseline_file = _find_clean_baseline(zmin, zmax)
        if not clean_baseline_file:
            raise FileNotFoundError(
                f"Could not find clean baseline for z=[{zmin}, {zmax}]. "
                f"Run compute_clean_baseline.py first or use --base-config."
            )

    logger.info(f"Configuration: z=[{zmin}, {zmax}], nmesh={nmesh}, nmock={nmock}")
    logger.info(f"Systematic type: {sys_type}")
    if sys_type == "power_law":
        ell_range_str = f"[{sys_ell_min}, {sys_ell_max}]" if sys_ell_min is not None and sys_ell_max is not None else "(not specified, using defaults)"
        logger.info(f"  alpha={sys_alpha}, amplitude={sys_amp}, ell_range={ell_range_str}")
    else:
        logger.info(f"  ell_contam={ell_contam}, amplitude={sys_amp}")
    if target_nbar is not None:
        logger.info(f"  target_nbar={target_nbar}")

    # Verify clean baseline and extract metadata
    with np.load(clean_baseline_file, allow_pickle=False) as dat:
        if "ells" not in dat or "all_plk" not in dat:
            raise ValueError(f"Invalid clean baseline file: missing required arrays")
        clean_ells = tuple(int(v) for v in np.atleast_1d(dat["ells"]).tolist())
        clean_all_plk = np.asarray(dat["all_plk"])
        clean_kcen = np.asarray(dat.get("kcen", np.arange(clean_all_plk.shape[-1])))

    logger.info(f"Clean baseline: shape={clean_all_plk.shape}, ells={clean_ells}")

    # Generate or use config label
    if args.base_config:
        config_label = args.base_config
    else:
        config_label = _get_config_label(zmin, zmax, target_nbar, nmesh)
    
    # Generate systematic label
    sys_label = _make_label_for_systematic({
        "type": sys_type,
        "alpha": sys_alpha,
        "amplitude": sys_amp,
        "ell_contam": ell_contam,
    })
    
    config_cache_dir = _get_config_cache_dir(cache_root, config_label, f"contamination_{sys_label}")
    
    logger.info(f"Config label: {config_label}")
    logger.info(f"Systematic label: {sys_label}")
    logger.info(f"Cache directory: {config_cache_dir}")
    
    # Determine which mocks to run
    mocks_to_run = _select_mocks_to_run(args.run_mode, nmock, config_cache_dir if args.run_mode == "remaining" else None)
    logger.info(f"Will compute {len(mocks_to_run)} mocks: {mocks_to_run}")
    
    # Build spec with contamination
    spec = ExperimentSpec(
        mock_type="halfdome",
        with_rsd=mocks_cfg.get("with_rsd", False),
        contamination_mode=sys_mode,
        redshift_sel=True,
        zmin=zmin,
        zmax=zmax,
        k_min=0.006,
        k_max=0.2,
        delta_k=0.01,
        n_sample=mocks_cfg.get("n_sample", 20_000_000),
        target_nbar=target_nbar,
        ds_fac=mocks_cfg.get("ds_fac", 5),
        nmesh=nmesh,
        ells=clean_ells,
        n_clean_bins=8,
        mu_binning_strategy="nonuniform",
        sys_amp=sys_amp,
        sys_spec_type="delta" if sys_type == "delta_function" else "power_law",
        sys_ell_min=sys_ell_min,
        sys_ell_max=sys_ell_max,
        sys_ell_delta=sys_ell_delta if sys_type == "power_law" else ell_contam[0] if ell_contam else None,
        sys_amp_mode="rms",
        sys_amp_mult_scale=1.0,
        seed=mocks_cfg.get("seed", 42),
        save_dir=config_cache_dir,
        output_name=f"contamination",
        mean_conserving_additive=mean_conserving,
    )
    
    # Compute per-mock results
    all_plk_list = []
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
            np.savez(mock_npz_path, all_plk=plk, ells=np.asarray(clean_ells), kcen=np.asarray(result.kcen))
            all_plk_list.append(plk)
            logger.info(f"[{mock_idx}] Saved: {mock_npz_path}")
        except Exception as e:
            logger.error(f"[{mock_idx}] Failed: {e}")
            raise
    
    # Aggregate results
    if all_plk_list:
        all_plk_agg = np.stack(all_plk_list, axis=0)  # Shape: (nmock_computed, nell, nk)
        agg_path = os.path.join(config_cache_dir, "multipoles_agg.npz")
        
        np.savez(agg_path, all_plk=all_plk_agg, ells=np.asarray(clean_ells), kcen=clean_kcen)
        logger.info(f"Aggregated {len(all_plk_list)} mocks -> {agg_path}")
        logger.info(f"Aggregated shape: {all_plk_agg.shape}")
    
    # Save metadata
    yaml_path = os.path.join(config_cache_dir, "config.yaml")
    rerun_path = os.path.join(config_cache_dir, "rerun.sh")
    
    rerun_cmd = _build_rerun_command(os.path.abspath(__file__), tuple(sys.argv[1:]))
    
    metadata = {
        "kind": f"contaminated_multipoles_additive_{sys_type}",
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "config_label": config_label,
        "systematic_label": sys_label,
        "config_cache_dir": config_cache_dir,
        "clean_baseline_file": clean_baseline_file,
        "rerun_script": rerun_path,
        "rerun_command": rerun_cmd,
        "config_file": args.config or os.path.join(os.path.dirname(__file__), f"config_contaminant_{sys_type}.yaml"),
        "loaded_config": _to_jsonable(config),
        "contamination": {
            "mode": sys_mode,
            "systematic_type": sys_type,
            "amplitude": sys_amp,
            "mean_conserving": mean_conserving,
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
        "ells": list(clean_ells),
        "mocks_computed": mocks_to_run,
        "runtime": {
            "nmock": nmock,
            "python_executable": sys.executable,
            "cwd": os.getcwd(),
            "environment_setup": "source ~/.desi_bashrc",
            "python_var": "$PYBIN",
        },
        "notes": "Contaminant-only multipoles can be computed as: all_plk_contaminant = all_plk_full - all_plk_clean",
    }
    
    _write_yaml(yaml_path, metadata)
    _write_rerun_script(rerun_path, rerun_cmd)
    
    logger.info(f"Saved metadata: {yaml_path}")
    logger.info(f"Saved rerun script: {rerun_path}")


if __name__ == "__main__":
    main()
