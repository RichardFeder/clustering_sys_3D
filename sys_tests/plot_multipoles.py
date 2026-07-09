#!/usr/bin/env python3
"""Plot cached multipoles from a configuration.

Usage:
    python plot_multipoles.py --config-str z0.1_0.4_nbarcatmesh512 [--mode clean|contamination_powerlaw...]
    
Loads aggregated multipole file from cache and creates a summary figure.
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from pypower import setup_logging
except ImportError:
    setup_logging = None

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import _wedges_to_poles, build_mu_wedges

logger = logging.getLogger(__name__)


def load_multipoles_from_cache(cache_dir: str, config_label: str, mode_pattern: str = "clean_baseline") -> dict:
    """
    Load multipoles from cache directory.
    
    First tries to load individual mock files (0, 1, 2, ..., nmock-1),
    then falls back to aggregated file if individual mocks not found.
    
    Parameters
    ----------
    cache_dir : str
        Path to cache directory
    config_label : str
        Configuration label (e.g., 'z0.1_0.4_nbarcatmesh512')
    mode_pattern : str
        Pattern to match in subdirectory name
        - 'clean_baseline' for clean baseline
        - 'contamination_*' for contamination modes (use glob pattern)
    
    Returns
    -------
    dict with keys: all_plk, ells, kcen, mu_wedges, config_label, mode_label, nmock
    """
    config_cache_dir = os.path.join(cache_dir, f"config_{config_label}")
    
    if not os.path.exists(config_cache_dir):
        raise FileNotFoundError(f"Config cache directory not found: {config_cache_dir}")
    
    # Find matching subdirectory
    subdirs = glob.glob(os.path.join(config_cache_dir, f"{mode_pattern}*"))
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories matching '{mode_pattern}' in {config_cache_dir}")
    
    subdir = subdirs[0]
    mode_label = os.path.basename(subdir)
    
    # Try to load individual mock files first
    # Detect naming convention: check for multipoles_mock_0.npz or contaminant_only_0.npz
    all_plk_list = []
    mock_idx = 0
    mock_filename = None
    agg_filename = None
    
    # Try contaminant_only first
    test_path = os.path.join(subdir, "contaminant_only_0.npz")
    if os.path.exists(test_path):
        mock_filename = "contaminant_only_{}.npz"
        agg_filename = "contaminant_only_agg.npz"
    else:
        # Fall back to multipoles_mock
        test_path = os.path.join(subdir, "multipoles_mock_0.npz")
        if os.path.exists(test_path):
            mock_filename = "multipoles_mock_{}.npz"
            agg_filename = "multipoles_agg.npz"
    
    if mock_filename is not None:
        mock_idx = 0
        while True:
            mock_path = os.path.join(subdir, mock_filename.format(mock_idx))
            if not os.path.exists(mock_path):
                break
            with np.load(mock_path, allow_pickle=False) as dat:
                plk = np.asarray(dat["all_plk"])  # Shape: (nell, nk)
                all_plk_list.append(plk)
            mock_idx += 1
    
    if all_plk_list:
        logger.info(f"Loaded {len(all_plk_list)} individual mock files")
        all_plk = np.stack(all_plk_list, axis=0)  # Shape: (nmock, nell, nk)
        # Load ells and kcen from first mock
        first_mock_path = os.path.join(subdir, mock_filename.format(0))
        with np.load(first_mock_path, allow_pickle=False) as dat:
            ells = tuple(int(e) for e in np.atleast_1d(dat["ells"]))
            kcen = np.asarray(dat["kcen"])
    else:
        # Fall back to aggregated file (try both naming conventions)
        agg_path = os.path.join(subdir, "contaminant_only_agg.npz")
        if not os.path.exists(agg_path):
            agg_path = os.path.join(subdir, "multipoles_agg.npz")
        if not os.path.exists(agg_path):
            raise FileNotFoundError(f"No individual mocks or aggregated file found in {subdir}")
        
        logger.info(f"No individual mocks found, loading aggregated file")
        with np.load(agg_path, allow_pickle=False) as dat:
            all_plk = np.asarray(dat["all_plk"])  # Shape: (nmock, nell, nk)
            ells = tuple(int(e) for e in np.atleast_1d(dat["ells"]))
            kcen = np.asarray(dat["kcen"])
    
    # Load metadata for mu_wedges
    config_path = os.path.join(subdir, "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Reconstruct mu_wedges from binning config if available
    n_mu_bins = config.get("binning", {}).get("n_clean_bins", 8)
    mu_wedges = np.linspace(0.0, 1.0, n_mu_bins + 1)
    
    return {
        "all_plk": all_plk,
        "ells": ells,
        "kcen": kcen,
        "mu_wedges": mu_wedges,
        "config_label": config_label,
        "mode_label": mode_label,
        "subdir": subdir,
        "nmock": all_plk.shape[0],
    }


def plot_multipoles(data: dict, target_ells: list = None, figsize: tuple = (10, 6)) -> tuple:
    """
    Plot power spectrum multipoles with error bars.
    
    Parameters
    ----------
    data : dict
        Output from load_multipoles_from_cache()
    target_ells : list, optional
        Multipole orders to plot. Defaults to [0, 2, 4, 6, 8, 10, 12, 14, 16]
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if target_ells is None:
        target_ells = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    
    all_plk = data["all_plk"]  # (nmock, nell, nk)
    ells = data["ells"]
    kcen = data["kcen"].real
    mode_label = data["mode_label"]
    nmock = data["nmock"]
    
    # Filter out NaN k-bins (keep only valid bins)
    valid_mask = ~np.isnan(all_plk[0, 0, :])  # Check first mock, first ell
    kcen_valid = kcen[valid_mask]
    all_plk_valid = all_plk[:, :, valid_mask]
    
    logger.info(f"Plotting {nmock} mocks, ells={ells}")
    logger.info(f"Valid k-bins: {valid_mask.sum()} / {len(valid_mask)}")
    
    # Create color map
    ell_colors = {ell: f"C{i}" for i, ell in enumerate(ells)}
    ell_labels = {ell: f"$P_{{{ell}}}(k)$" for ell in ells}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each target multipole
    for ell in target_ells:
        if ell not in ells:
            continue
        
        ell_idx = list(ells).index(ell)
        # Shape: (nmock, nk_valid)
        plk_ell = all_plk_valid[:, ell_idx, :].real
        
        # Compute mean and stderr
        mean_ell = np.mean(plk_ell, axis=0)
        std_ell = np.std(plk_ell, axis=0)
        stderr_ell = std_ell / np.sqrt(nmock)
        
        logger.debug(f"ell={ell}: mean range [{mean_ell.min():.3e}, {mean_ell.max():.3e}], stderr range [{stderr_ell.min():.3e}, {stderr_ell.max():.3e}]")
        
        # Plot with prominent error bars
        ax.errorbar(
            kcen_valid, mean_ell, yerr=stderr_ell,
            label=ell_labels[ell],
            color=ell_colors[ell],
            linewidth=2.5,
            marker="o",
            markersize=6,
            capsize=4,
            elinewidth=1.5,
            alpha=0.85,
        )
    
    ax.set_xlabel("k [h/Mpc]", fontsize=12)
    ax.set_ylabel(r"$P_\ell(k)$ [(Mpc/h)$^3$]", fontsize=12)
    ax.set_title(f"Power Spectrum Multipoles ({nmock} mocks): {mode_label}", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    
    return fig, ax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot cached power spectrum multipoles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config-str",
        type=str,
        required=True,
        help="Configuration label (e.g., z0.1_0.4_nbarcatmesh512)",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="clean_baseline",
        help="Mode to plot: 'clean_baseline', or 'contamination_*' pattern",
    )
    p.add_argument(
        "--target-ells",
        type=int,
        nargs="+",
        default=None,
        help="Multipole orders to plot (default: 0 2 4 6 8 10 12 14 16)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figure (default: cache subdirectory)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
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
    
    # Determine cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    logger.info(f"Loading from cache: {cache_dir}")
    
    # Load multipoles
    try:
        data = load_multipoles_from_cache(cache_dir, args.config_str, args.mode)
        logger.info(f"Loaded multipoles for config '{args.config_str}' mode '{args.mode}'")
        logger.info(f"Shape all_plk: {data['all_plk'].shape}")
        logger.info(f"Ells: {data['ells']}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load multipoles: {e}")
        sys.exit(1)
    
    # Create figure
    target_ells = args.target_ells or [0, 2, 4, 6, 8, 10, 12, 14, 16]
    fig, ax = plot_multipoles(data, target_ells=target_ells)
    
    # Determine output path
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = data["subdir"]
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"multipoles_{data['mode_label']}.png")
    
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved figure: {output_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()
