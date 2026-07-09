#!/usr/bin/env python3
"""Compute and plot P(k,mu) under multiple mu-binning strategies.

This script consumes cached multipole outputs from sys_tests/cache and computes
P(k,mu) wedges for:
1) Uniform mu bins
2) Non-uniform (window-free) bins
3) Non-uniform + window correction (delta-function or power-law contaminants)

For delta-function contamination cases, it also compares the analytical
(delta-k) and numerical (flat-k integration) windowed edge calculations as a
sanity check for the window model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

# Add repo root to path (script is in sys_tests/)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from pipeline import _poles_to_wedges
from nonunif_binning import (
    compare_delta_k_approaches,
    compute_mu_edges_for_delta_function,
    compute_null_bins,
    compute_r_window_from_redshifts,
    compute_windowed_mu_edges_solve_mu1,
)


@dataclass
class Dataset:
    name: str
    kind: str  # "clean" or "delta"
    ell_contam: Optional[int]
    all_plk: np.ndarray  # (nmock, nell, nk)
    ells: np.ndarray
    kcen: np.ndarray
    clean_ref: Optional[np.ndarray] = None  # (nmock_ref, nell, nk)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and plot P(k,mu) binning comparisons")
    p.add_argument("--config-label", type=str, default="z0.1_0.4_nbar3e-04_nmesh512")
    p.add_argument("--cache-root", type=str, default="sys_tests/cache")
    p.add_argument("--output-root", type=str, default="sys_tests/results/pkmu_binning")
    p.add_argument("--n-clean-bins", type=int, default=8)
    p.add_argument("--ell-max", type=int, default=16)
    p.add_argument("--ell-kernel-max", type=int, default=128)
    p.add_argument("--zmin", type=float, default=0.1)
    p.add_argument("--zmax", type=float, default=0.4)
    p.add_argument("--window-r", type=float, default=None)
    p.add_argument("--powerlaw-ell-eff", type=int, default=20)
    p.add_argument("--window-compare-threshold", type=float, default=5e-3)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _effective_k_from_ell(ell_eff: int, z_eff: float) -> Tuple[float, float]:
    """Compute effective transverse scale via k_perp = ell / chi(z_eff)."""
    from astropy.cosmology import Planck18 as cosmo

    chi_mpc_h = float(cosmo.comoving_distance(z_eff).value * cosmo.h)
    if not (chi_mpc_h > 0):
        raise ValueError(f"Invalid chi(z_eff={z_eff}): {chi_mpc_h}")
    return float(ell_eff) / chi_mpc_h, chi_mpc_h

def plot_mu_edges_by_strategy(edge_summary: Dict[str, Dict[str, list]], fpath: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))

    strategy_order = [
        "uniform",
        "nonuniform_nowindow",
        "nonuniform_window_delta",
        "nonuniform_window_powerlaw",
    ]
    strategy_colors = {
        "uniform": "C0",
        "nonuniform_nowindow": "C1",
        "nonuniform_window_delta": "C2",
        "nonuniform_window_powerlaw": "C3",
    }

    shown = set()
    for _, edges_dict in edge_summary.items():
        for strat in strategy_order:
            if strat not in edges_dict:
                continue
            edges = np.asarray(edges_dict[strat], dtype=float)
            label = strat if strat not in shown else None
            ax.vlines(
                edges,
                ymin=0.0,
                ymax=1.0,
                color=strategy_colors.get(strat, "k"),
                lw=1.8,
                alpha=0.8,
                label=label,
            )
            shown.add(strat)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r'$\mu$', fontsize=16)
    ax.set_yticks([])
    ax.grid(alpha=0.25, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=4, fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as dat:
        all_plk = np.asarray(dat["all_plk"])
        ells = np.asarray(dat["ells"])
        kcen = np.asarray(dat["kcen"])
    if all_plk.ndim == 2:
        all_plk = all_plk[None, ...]
    return all_plk, ells, kcen


def _ensure_edges_01(edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=float).copy()
    e = np.unique(np.clip(e, 0.0, 1.0))
    if e.size == 0:
        raise ValueError("Empty edge array")
    if e[0] > 1e-10:
        e = np.concatenate(([0.0], e))
    else:
        e[0] = 0.0
    if e[-1] < 1.0 - 1e-10:
        e = np.concatenate((e, [1.0]))
    else:
        e[-1] = 1.0
    if np.any(np.diff(e) <= 0):
        raise ValueError(f"Invalid mu edges after normalization: {e}")
    return e


def _infer_delta_ell(dirname: str) -> Optional[int]:
    m = re.search(r"ellcontam(\d+)", dirname)
    return int(m.group(1)) if m else None


def load_datasets(args: argparse.Namespace) -> Dict[str, Dataset]:
    cfg_dir = os.path.join(args.cache_root, f"config_{args.config_label}")
    clean_path = os.path.join(cfg_dir, "clean_baseline", "multipoles_agg.npz")
    clean_plk, clean_ells, clean_kcen = _load_npz(clean_path)

    out: Dict[str, Dataset] = {
        "clean": Dataset(
            name="clean",
            kind="clean",
            ell_contam=None,
            all_plk=clean_plk,
            ells=clean_ells,
            kcen=clean_kcen,
            clean_ref=None,
        )
    }

    for entry in sorted(os.listdir(cfg_dir)):
        if not (entry.startswith("contaminant_only_deltafn_") or entry.startswith("contaminant_only_powerlaw_")):
            continue
        agg_path = os.path.join(cfg_dir, entry, "contaminant_only_agg.npz")
        if not os.path.exists(agg_path):
            continue
        contam_plk, ells, kcen = _load_npz(agg_path)
        nref = min(clean_plk.shape[0], contam_plk.shape[0])
        total_plk = clean_plk[:nref] + contam_plk[:nref]

        if entry.startswith("contaminant_only_deltafn_"):
            ell_contam = _infer_delta_ell(entry)
            label = f"delta_ell{ell_contam}" if ell_contam is not None else entry
            kind = "delta"
        else:
            ell_contam = None
            label = entry
            kind = "powerlaw"

        out[label] = Dataset(
            name=label,
            kind=kind,
            ell_contam=ell_contam,
            all_plk=total_plk,
            ells=ells,
            kcen=kcen,
            clean_ref=clean_plk[:nref],
        )

    if args.verbose:
        print("Loaded datasets:")
        for key, ds in out.items():
            print(f"  {key:>12s}: all_plk={ds.all_plk.shape}, ells={tuple(ds.ells.tolist())}")

    return out


def poles_to_pkmu(all_plk: np.ndarray, ells: np.ndarray, mu_edges: np.ndarray) -> np.ndarray:
    nmock = all_plk.shape[0]
    out = []
    for imock in range(nmock):
        plk_kell = np.asarray(all_plk[imock]).T  # (nk, nell)
        pkmu = _poles_to_wedges(ells, plk_kell, mu_edges)
        out.append(np.real(pkmu))
    return np.stack(out, axis=0)


def compute_mu_edge_sets_for_dataset(args: argparse.Namespace, ds: Dataset, r_window: float, z_eff: float) -> Dict[str, np.ndarray]:
    # Non-uniform no-window (from Hand-like null construction)
    mu_nonuniform_nowindow = _ensure_edges_01(
        compute_null_bins(ell_max=args.ell_max, n_clean_bins=args.n_clean_bins)
    )

    # Uniform: match number of bins to non-uniform for fair comparison
    n_bins = len(mu_nonuniform_nowindow) - 1
    mu_uniform = np.linspace(0.0, 1.0, n_bins + 1)

    edge_sets: Dict[str, np.ndarray] = {
        "uniform": mu_uniform,
        "nonuniform_nowindow": mu_nonuniform_nowindow,
    }

    if ds.kind == "delta":
        if ds.ell_contam is None:
            raise ValueError(f"Could not infer ell_contam for {ds.name}")

        # Analytical delta-function windowed solution used for production edges
        try:
            mu_window = compute_mu_edges_for_delta_function(
                ell_contam=ds.ell_contam,
                z_eff=z_eff,
                ell_max=args.ell_max,
                ell_kernel_max=args.ell_kernel_max,
                R_window=r_window,
                n_clean_bins=args.n_clean_bins,
                verbose=args.verbose,
            )
            edge_sets["nonuniform_window_delta"] = _ensure_edges_01(mu_window)
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to compute windowed edges for {ds.name}: {e}")

    elif ds.kind == "powerlaw":
        # Power-law contamination windowed solution
        try:
            mu_window, _ = compute_windowed_mu_edges_solve_mu1(
                ell_max=args.ell_max,
                n_clean_bins=args.n_clean_bins,
                k=ds.kcen,
                R_window=r_window,
                alpha=2.0,  # Power-law index (typical for dust/stars)
                ell_kernel_max=args.ell_kernel_max,
                verbose=args.verbose,
            )
            edge_sets["nonuniform_window_powerlaw"] = _ensure_edges_01(mu_window)
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to compute windowed edges for {ds.name}: {e}")

    return edge_sets


def compare_window_edge_methods(args: argparse.Namespace, ds: Dataset, r_window: float, z_eff: float) -> Dict[str, float]:
    if ds.kind != "delta" or ds.ell_contam is None:
        return {}

    cmp_out = compare_delta_k_approaches(
        ell_contam=ds.ell_contam,
        z_eff=z_eff,
        ell_max=args.ell_max,
        ell_kernel_max=args.ell_kernel_max,
        R_window=r_window,
        n_clean_bins=args.n_clean_bins,
        verbose=args.verbose,
    )

    status = "ok" if cmp_out["edges_diff"] <= args.window_compare_threshold else "warn"
    return {
        "ell_contam": float(ds.ell_contam),
        "k_c": float(cmp_out["k_c"]),
        "diff_linf": float(cmp_out["diff_linf"]),
        "diff_l2": float(cmp_out["diff_l2"]),
        "mu1_numerical": float(cmp_out["mu1_numerical"]),
        "mu1_analytical": float(cmp_out["mu1_analytical"]),
        "mu1_diff": float(cmp_out["mu1_diff"]),
        "edges_diff": float(cmp_out["edges_diff"]),
        "threshold": float(args.window_compare_threshold),
        "status": status,
    }


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.abs(den) > 0
    out[mask] = num[mask] / den[mask]
    return out


def plot_clean_dataset(kcen: np.ndarray, pkmu_by_strategy: Dict[str, np.ndarray], edges_by_strategy: Dict[str, np.ndarray], fpath: str) -> None:
    strategies = ["uniform", "nonuniform_nowindow"]
    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 4), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        pkmu = pkmu_by_strategy[strat]  # (nmock, nk, nmu)
        edges = edges_by_strategy[strat]
        nmu = pkmu.shape[-1]
        cmap = plt.get_cmap("viridis")

        for imu in range(nmu):
            lo, hi = edges[imu], edges[imu + 1]
            mean_pk = np.nanmean(pkmu[:, :, imu], axis=0)
            err_pk = np.nanstd(pkmu[:, :, imu], axis=0) / np.sqrt(pkmu.shape[0])
            lw = 2.8 if imu == 0 else 1.5
            color = "crimson" if imu == 0 else cmap((imu + 0.5) / max(nmu, 2))
            label = f"{lo:.3f}<mu<{hi:.3f}" + (" (junk)" if imu == 0 else "")
            ax.errorbar(kcen, kcen * mean_pk, yerr=kcen * err_pk, color=color, lw=lw, alpha=0.9, label=label)

        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_title(strat)

    axes[0].set_ylabel("k P(k,mu) [(Mpc/h)^2]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contamination_dataset(
    kcen: np.ndarray,
    pkmu_by_strategy: Dict[str, np.ndarray],
    clean_pkmu_by_strategy: Dict[str, np.ndarray],
    edges_by_strategy: Dict[str, np.ndarray],
    title: str,
    fpath: str,
) -> None:
    """Plot P(k,mu) for contaminated datasets with available strategies.
    
    Dynamically uses strategies that were successfully computed. For delta
    contamination, expects 'nonuniform_window_delta'; for power-law,
    expects 'nonuniform_window_powerlaw'.
    """
    # Determine available strategies
    all_strategies = ["uniform", "nonuniform_nowindow", "nonuniform_window_delta", "nonuniform_window_powerlaw"]
    strategies = [s for s in all_strategies if s in pkmu_by_strategy]
    
    fig, axes = plt.subplots(2, len(strategies), figsize=(10, 5), sharex=True)
    # Ensure axes is always 2D array for consistent indexing
    if len(strategies) == 1:
        axes = axes.reshape(2, 1)
    elif len(strategies) == 2:
        axes = axes.T.reshape(2, 2)

    for icol, strat in enumerate(strategies):
        pkmu = pkmu_by_strategy[strat]
        pkmu_clean = clean_pkmu_by_strategy[strat]
        edges = edges_by_strategy[strat]
        nmu = pkmu.shape[-1]
        cmap = plt.get_cmap("viridis")

        ax_top = axes[0, icol]
        ax_bot = axes[1, icol]

        for imu in range(nmu):
            lo, hi = edges[imu], edges[imu + 1]
            label = f"{lo:.3f}<mu<{hi:.3f}" + (" (junk)" if imu == 0 else "")
            color = "crimson" if imu == 0 else cmap((imu + 0.5) / max(nmu, 2))
            lw = 2.8 if imu == 0 else 1.4

            mean_sys = np.nanmean(pkmu[:, :, imu], axis=0)
            err_sys = np.nanstd(pkmu[:, :, imu], axis=0) / np.sqrt(pkmu.shape[0])
            mean_clean = np.nanmean(pkmu_clean[:, :, imu], axis=0)

            ratio = _safe_ratio(mean_sys, mean_clean)

            ax_top.errorbar(kcen, mean_sys, yerr=err_sys, color=color, lw=lw, alpha=0.9, label=label)
            ax_bot.plot(kcen, ratio, color=color, lw=lw, alpha=0.9, label=label)

            if imu==0:
                ax_top.set_ylim(1e2, 1.2*np.nanmax(np.abs(mean_sys)))


        ax_top.set_xscale("log")
        ax_top.grid(alpha=0.25)
        ax_top.set_title(strat)
        ax_bot.axhline(1.0, color="k", lw=1.2, ls="--")
        ax_bot.set_xscale("log")
        ax_bot.grid(alpha=0.25)
        ax_bot.set_xlabel("k [h/Mpc]")
        ax_bot.set_ylim(1e-2, 1e3)
        ax_bot.set_yscale('log')
        ax_top.set_yscale('log')

    axes[0, 0].set_ylabel("$P_{\\rm sys}(k,\\mu)$ [(Mpc/h)$^3$]")
    axes[1, 0].set_ylabel("|$P_{\\rm sys} / P_{\\rm clean}$|")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=5, fontsize=8, frameon=True)
    fig.suptitle(title, y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contamination_mu_response(
    ds_name: str,
    ell_eff: int,
    k_eff: float,
    k_selected: float,
    k_idx: int,
    chi_mpc_h: float,
    pkmu_by_strategy: Dict[str, np.ndarray],
    clean_pkmu_by_strategy: Dict[str, np.ndarray],
    edges_by_strategy: Dict[str, np.ndarray],
    fpath: str,
) -> None:
    """Plot contamination power vs mu for each available binning strategy."""
    strategy_order = ["uniform", "nonuniform_nowindow", "nonuniform_window_delta", "nonuniform_window_powerlaw"]
    strategy_labels = {
        "uniform": "uniform",
        "nonuniform_nowindow": "nonuniform_nowindow",
        "nonuniform_window_delta": "nonuniform_window_delta",
        "nonuniform_window_powerlaw": "nonuniform_window_powerlaw",
    }
    available = [s for s in strategy_order if s in pkmu_by_strategy and s in clean_pkmu_by_strategy]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")

    for i, strat in enumerate(available):
        pkmu = pkmu_by_strategy[strat]  # (nmock, nk, nmu)
        print('pkmu has shape', pkmu.shape, 'for strategy', strat)
        # pkmu_clean = clean_pkmu_by_strategy[strat]  # (nmock, nk, nmu)
        mu_edges = edges_by_strategy[strat]
        mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

        # each ratio normalized by the zeroth mu bin (junk bin) for that strategy
        # ratio_mu = np.nanmean(pkmu[:,:,k_idx] / pkmu[:, 0, k_idx][:, None,:], axis=0)
        # k_idx = 0
        ratio_mu = np.nanmean(pkmu[:,k_idx, :] / pkmu[:, k_idx, 0][:,None], axis=0)

        print('ratio mu has shape', ratio_mu.shape, 'for strategy', strat)
        # delta_mu = np.nanmean(pkmu - pkmu_clean, axis=0)[k_idx, :]
        ax.plot(
            mu_centers,
            np.abs(ratio_mu),
            marker="o",
            ms=4,
            lw=1.8,
            color=cmap(i),
            label=strategy_labels[strat],
        )

    ax.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.8)
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\mu$", fontsize=14)
    ax.set_ylabel(r"$P(k,\mu) / P(k,\mu=0)$", fontsize=14)
    ax.set_ylim(1e-6, 1e1)
    ax.set_yscale('log')
    ax.set_title(
        f"{ds_name}: mu-response at k~ell/chi\n"
        f"ell_eff={ell_eff}, chi(z_eff)={chi_mpc_h:.2f} Mpc/h, "
        f"k_eff={k_eff:.4f}, k_bin={k_selected:.4f} (idx={k_idx})"
    )
    ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    fig_dir = os.path.join(args.output_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    z_eff = 0.5 * (args.zmin + args.zmax)
    r_window = float(args.window_r) if args.window_r is not None else float(compute_r_window_from_redshifts(args.zmin, args.zmax))

    datasets = load_datasets(args)

    summary: Dict[str, object] = {
        "config_label": args.config_label,
        "zmin": args.zmin,
        "zmax": args.zmax,
        "z_eff": z_eff,
        "R_window": r_window,
        "n_clean_bins": args.n_clean_bins,
        "ell_max": args.ell_max,
        "ell_kernel_max": args.ell_kernel_max,
        "datasets": {},
    }

    edge_summary: Dict[str, Dict[str, list]] = {}

    for name, ds in datasets.items():
        edges_by_strategy = compute_mu_edge_sets_for_dataset(args, ds, r_window, z_eff)
        edge_summary[name] = {k: v.tolist() for k, v in edges_by_strategy.items()}
        window_compare = compare_window_edge_methods(args, ds, r_window, z_eff)

        pkmu_by_strategy: Dict[str, np.ndarray] = {}
        clean_pkmu_by_strategy: Dict[str, np.ndarray] = {}

        for strat, edges in edges_by_strategy.items():
            pkmu_by_strategy[strat] = poles_to_pkmu(ds.all_plk, ds.ells, edges)
            if ds.clean_ref is not None:
                clean_pkmu_by_strategy[strat] = poles_to_pkmu(ds.clean_ref, ds.ells, edges)

        npz_path = os.path.join(args.output_root, f"pkmu_bins_{name}.npz")
        np_payload = {
            "kcen": ds.kcen,
            "ells": ds.ells,
            "all_plk": ds.all_plk,
        }
        for strat, edges in edges_by_strategy.items():
            np_payload[f"mu_edges_{strat}"] = edges
            np_payload[f"pkmu_{strat}"] = pkmu_by_strategy[strat]
            if strat in clean_pkmu_by_strategy:
                np_payload[f"pkmu_clean_{strat}"] = clean_pkmu_by_strategy[strat]
        np.savez(npz_path, **np_payload)

        if ds.kind == "clean":
            fig_path = os.path.join(fig_dir, "pkmu_binning_clean.png")
            plot_clean_dataset(ds.kcen, pkmu_by_strategy, edges_by_strategy, fig_path)
            mu_resp_fig = None
            mu_resp_meta = None
        else:
            fig_path = os.path.join(fig_dir, f"pkmu_binning_{name}.png")
            title = f"{name}: binning strategy and junk-bin isolation"
            plot_contamination_dataset(
                ds.kcen,
                pkmu_by_strategy,
                clean_pkmu_by_strategy,
                edges_by_strategy,
                title,
                fig_path,
            )

            ell_eff = int(ds.ell_contam) if ds.ell_contam is not None else int(args.powerlaw_ell_eff)
            k_eff, chi_mpc_h = _effective_k_from_ell(ell_eff, z_eff)
            print('k_eff:', k_eff)
            print('ds.kcen:', ds.kcen)

            kcen_eval = ds.kcen.copy()

            kcen_eval[np.isnan(kcen_eval)] = np.inf  # Avoid NaN issues in argmin
            k_idx = int(np.argmin(np.abs(kcen_eval - k_eff)))
            k_selected = float(kcen_eval[k_idx])
            print('k_selected:', k_selected)

            mu_resp_fig = os.path.join(fig_dir, f"contamination_mu_response_{name}.png")
            plot_contamination_mu_response(
                ds_name=name,
                ell_eff=ell_eff,
                k_eff=k_eff,
                k_selected=k_selected,
                k_idx=k_idx,
                chi_mpc_h=chi_mpc_h,
                pkmu_by_strategy=pkmu_by_strategy,
                clean_pkmu_by_strategy=clean_pkmu_by_strategy,
                edges_by_strategy=edges_by_strategy,
                fpath=mu_resp_fig,
            )
            mu_resp_meta = {
                "figure": mu_resp_fig,
                "ell_eff": ell_eff,
                "chi_mpc_h": chi_mpc_h,
                "k_eff": float(k_eff),
                "k_selected": k_selected,
                "k_index": k_idx,
            }

        summary["datasets"][name] = {
            "kind": ds.kind,
            "ell_contam": ds.ell_contam,
            "nmock": int(ds.all_plk.shape[0]),
            "nmu_by_strategy": {k: int(len(v) - 1) for k, v in edges_by_strategy.items()},
            "npz": npz_path,
            "figure": fig_path,
            "mu_response": mu_resp_meta,
            "window_compare": window_compare,
        }

        if args.verbose:
            print(f"[{name}] saved {npz_path}")
            print(f"[{name}] saved {fig_path}")
            if mu_resp_fig is not None:
                print(f"[{name}] saved {mu_resp_fig}")
            if window_compare:
                print(f"[{name}] window compare: {window_compare}")


    # plt.figure(figsize=(10, 3))
    # for name, edges_dict in edge_summary.items():
    #     for strat, edges in edges_dict.items():
    #         plt.step(edges, np.arange(len(edges)), where="post", label=f"{name} ({strat})", color='C'+str(len(edges_dict)))
    # plt.xlabel('$\\mu$', fontsize=16)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(fig_dir, "mu_edges_by_strategy.png"), dpi=200)
    # plt.show()

    plot_mu_edges_by_strategy(
        edge_summary,
        os.path.join(fig_dir, "mu_edges_by_strategy.png"),
    )



    edge_path = os.path.join(args.output_root, "mu_edges_all_strategies.json")
    with open(edge_path, "w", encoding="utf-8") as f:
        json.dump(edge_summary, f, indent=2)

    meta_path = os.path.join(args.output_root, "summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results: {args.output_root}")
    print(f"Figures: {fig_dir}")
    print(f"Summary: {meta_path}")
    print(f"Mu edges: {edge_path}")

def new_func(edge_summary):
    plt.figure(figsize=(10, 3))

    for name, edges_dict in edge_summary.items():
        for strat, edges in edges_dict.items():
            plt.step(edges, np.arange(len(edges)), where="post", label=f"{name} ({strat})")
    plt.xlabel('$\\mu$', fontsize=16)


if __name__ == "__main__":
    main()
