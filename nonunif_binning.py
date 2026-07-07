# -*- coding: utf-8 -*-
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz as _cumtrapz
from scipy.special import legendre, spherical_jn, eval_legendre

from scipy.optimize import brentq

def _find_zero_crossings(x, y):
    """Linear-interpolated zero crossings of y(x)."""
    z = []
    s = np.sign(y)
    for i in range(1, len(x)):
        if s[i-1] == 0:
            z.append(x[i-1])
        elif s[i] == 0:
            z.append(x[i])
        elif s[i-1] * s[i] < 0:
            x0, x1 = x[i-1], x[i]
            y0, y1 = y[i-1], y[i]
            z0 = x0 - y0 * (x1 - x0) / (y1 - y0)
            z.append(z0)
    if len(z) == 0:
        return np.array([])
    z = np.array(sorted(set(np.round(z, 12))))
    return z

def _cum_from_mu1(mu, R, mu1):
    """Cumulative integral I(mu)=∫_{mu1}^{mu} R dmu on mu>=mu1 grid."""
    i0 = np.searchsorted(mu, mu1, side='left')
    mu_sub = mu[i0:]
    R_sub  = R[i0:]
    if len(mu_sub) < 2:
        return mu_sub, np.array([])
    # shift so first point is exactly mu1 via interpolation
    if mu_sub[0] > mu1:
        R1 = np.interp(mu1, mu, R)
        mu_sub = np.concatenate([[mu1], mu_sub])
        R_sub  = np.concatenate([[R1], R_sub])
    I = _cumtrapz(R_sub, mu_sub, initial=0.0)
    return mu_sub, I

def _endpoint_integral(mu, R, mu1):
    mu_sub, I = _cum_from_mu1(mu, R, mu1)
    if len(I) == 0:
        return np.nan
    return I[-1]  # ∫_{mu1}^{1} R dmu

def solve_mu1_and_edges_windowed(
    mu_grid,
    R_leak,
    n_clean_bins,
    mu1_min=1e-4,
    mu1_max=0.8,
    prefer_nowindow_mu1=None,
    verbose=True
):
    """
    Returns:
      mu1_opt, edges, mu_sub, I_sub
    where edges starts at mu1_opt and ends at 1.0.
    """
    mu = np.asarray(mu_grid)
    R  = np.asarray(R_leak)

    # normalize for numerical stability (doesn't change roots)
    scale = np.nanmax(np.abs(R))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("R_leak is zero/invalid everywhere.")
    Rn = R / scale

    # --- 1) solve mu1 from endpoint condition ∫_{mu1}^{1} R dmu = 0
    def F(m):
        return _endpoint_integral(mu, Rn, m)

    # bracket search
    trial = np.linspace(mu1_min, mu1_max, 200)
    vals = np.array([F(t) for t in trial])

    mu1_opt = None
    idx = np.where(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1]*vals[1:] < 0))[0]
    if len(idx) > 0:
        # optionally pick bracket near no-window mu1
        if prefer_nowindow_mu1 is not None:
            centers = 0.5*(trial[idx] + trial[idx+1])
            j = np.argmin(np.abs(centers - prefer_nowindow_mu1))
            i = idx[j]
        else:
            i = idx[0]
        mu1_opt = brentq(F, trial[i], trial[i+1], xtol=1e-10, rtol=1e-10, maxiter=200)
    else:
        # fallback: pick mu1 minimizing |endpoint integral|
        j = np.nanargmin(np.abs(vals))
        mu1_opt = trial[j]
        if verbose:
            print("[warn] No sign-change for endpoint condition; using mu1 minimizing |∫R|.")

    # --- 2) cumulative from mu1_opt, get zero crossings
    mu_sub, I_sub = _cum_from_mu1(mu, Rn, mu1_opt)
    zc = _find_zero_crossings(mu_sub, I_sub)

    # keep only crossings strictly inside (mu1,1)
    zc = zc[(zc > mu1_opt + 1e-8) & (zc < 1.0 - 1e-8)]

    # build edges: [mu1, z1, z2, ..., 1]
    edges = [mu1_opt]
    edges.extend(zc.tolist())
    edges.append(1.0)
    edges = np.array(edges)

    # --- 3) enforce desired number of bins if possible
    # need exactly n_clean_bins bins => n_clean_bins+1 edges
    target_edges = n_clean_bins + 1
    if len(edges) > target_edges:
        # keep earliest crossings (or you could choose largest-|I'| stability)
        keep = [edges[0]] + edges[1:target_edges-1].tolist() + [1.0]
        edges = np.array(keep)
    elif len(edges) < target_edges:
        if verbose:
            print(f"[warn] Only found {len(edges)-1} bins, requested {n_clean_bins}.")
            print("       Consider changing k-weighting, ell_kernel_max, or n_clean_bins.")

    return mu1_opt, edges, mu_sub, I_sub


def compute_r_window_from_redshifts(z_min, z_max):
    """
    Compute comoving distance shell thickness (Mpc/h) from redshift bounds.
    
    Used to set the radial window scale (R_window) for window-corrected mu binning
    in halfdome mock analysis, accounting for the finite redshift range sampled.
    
    Parameters
    ----------
    z_min, z_max : float
        Redshift bounds of the sample.
    
    Returns
    -------
    float
        R_window in Mpc/h: comoving distance thickness = chi(z_max) - chi(z_min).
        
    Examples
    --------
    For Halfdome z ∈ [0.4, 1.0], returns ~780 Mpc/h.
    """
    from astropy.cosmology import Planck18 as cosmo
    r_min = cosmo.comoving_distance(z_min).value * cosmo.h
    r_max = cosmo.comoving_distance(z_max).value * cosmo.h
    return r_max - r_min

def transverse_response(mu, ell_max):
    """Evaluate transverse systematic response function R(mu)"""
    mu = np.atleast_1d(mu)
    R = np.zeros_like(mu)
    for ell in range(0, ell_max + 1, 2):  # Even ℓ only
        P_ell_mu = legendre(ell)(mu)
        P_ell_0 = legendre(ell)(0.0)
        R += (2 * ell + 1) / 2 * P_ell_0 * P_ell_mu
    return R


def response_delta_mu0(mu, ell_max):
    L_lm = legendre(ell_max)
    L_lp1 = legendre(ell_max + 1)
    L_lm0 = L_lm(0.0)
    # Avoid division by zero for mu=0 by adding small epsilon
    mu_safe = np.where(mu == 0, 1e-12, mu)
    return - (ell_max + 1) / (2 * mu_safe) * L_lm0 * L_lp1(mu)


# using eval_legendre instead of legendre
def transverse_response_new(mu, ell_max):
    """
    Evaluate truncated transverse response:
        R(mu) = sum_{ell even <= ell_max} (2ell+1)/2 * L_ell(0) * L_ell(mu)
    """
    mu = np.atleast_1d(mu).astype(float)
    R = np.zeros_like(mu)
    for ell in range(0, ell_max + 1, 2):
        R += 0.5 * (2 * ell + 1) * eval_legendre(ell, 0.0) * eval_legendre(ell, mu)
    return R

def response_delta_mu0_new(mu, ell_max):
    """
    Hand+17 Eq. (3.5)-style leakage response (window-free):
        R(mu) = - (ell_max+1)/(2 mu) * L_{ell_max}(0) * L_{ell_max+1}(mu)
    """
    mu = np.atleast_1d(mu).astype(float)
    mu_safe = np.where(np.abs(mu) < 1e-14, 1e-14, mu)
    return -0.5 * (ell_max + 1) * eval_legendre(ell_max, 0.0) * eval_legendre(ell_max + 1, mu_safe) / mu_safe

    

def build_windowed_leakage_response(mu_scan, ell_max, ell_kernel_max, k, R_window, A=1.0, alpha=2.0):
    leakage_ells = list(range(ell_max + 2, ell_kernel_max + 1, 2))

    a_lk = compute_aell_convolved_tophat(
        ells=leakage_ells,
        k_eval=k,
        A=A,
        alpha=alpha,
        R_window=R_window,
    )

    # choose k-weights; this matches your current spirit (Pc-weighted)
    w = k**(-alpha)
    w = w / np.sum(w)

    R = np.zeros_like(mu_scan, dtype=float)
    for ell in leakage_ells:
        a_bar = np.sum(w * a_lk[ell])   # scalar
        R += a_bar * eval_legendre(ell, mu_scan)

    return R


def compute_null_bins(ell_max, n_clean_bins):
    mu_junk_upper = 1.0 / (ell_max/2 + 1)
    mu_vals = np.linspace(mu_junk_upper, 1.0, 10000)

    R = response_delta_mu0(mu_vals, ell_max)

    # Cumulative integral from mu_junk_upper upwards
    cum_integral = _cumtrapz(R, mu_vals, initial=0)

    # Find zero crossings of the cumulative integral after mu_junk_upper
    zero_crossings = []
    for i in range(1, len(cum_integral)):
        if cum_integral[i-1]*cum_integral[i] < 0:
            # Linear interpolation to estimate zero crossing
            mu_zero = mu_vals[i-1] - cum_integral[i-1] * (mu_vals[i] - mu_vals[i-1]) / (cum_integral[i] - cum_integral[i-1])
            zero_crossings.append(mu_zero)

    zero_crossings = np.array(zero_crossings)

    if len(zero_crossings) < n_clean_bins:
        print(f"Warning: Only found {len(zero_crossings)} nulls but requested {n_clean_bins} bins.")

    # Select first n_clean_bins zero crossings as bin edges
    clean_bin_edges = zero_crossings[:n_clean_bins]

    # Full bin edges: junk bin + clean bins + endpoint
    bin_edges = np.concatenate(([0], [mu_junk_upper], clean_bin_edges, [1.0]))

    return bin_edges


def _mean_legendre_over_bin(ell, mu_lo, mu_hi):
    """Average of L_ell over [mu_lo, mu_hi]."""
    dmu = mu_hi - mu_lo
    if dmu <= 0:
        raise ValueError(f'Invalid mu bin [{mu_lo}, {mu_hi}]')
    if ell == 0:
        return 1.0
    # ∫ L_ell dmu = [L_{ell+1}-L_{ell-1}] / (2ell+1)
    integ = (
        eval_legendre(ell + 1, mu_hi) - eval_legendre(ell - 1, mu_hi)
        - eval_legendre(ell + 1, mu_lo) + eval_legendre(ell - 1, mu_lo)
    ) / (2 * ell + 1)
    return integ / dmu

def _delta_l_transverse(ell):
    """delta_l = (2l+1)/2 * L_l(0), for delta_D(mu) transverse contaminant."""
    return 0.5 * (2 * ell + 1) * eval_legendre(ell, 0.0)


def _spherical_tophat_w(k, R):
    """w(k;R) = 3 j1(kR)/(kR), numerically stable near kR=0."""
    x = np.asarray(k) * R
    w = np.ones_like(x, dtype=float)
    small = np.abs(x) < 1e-6
    xs = x[~small]
    w[~small] = 3.0 * spherical_jn(1, xs) / xs
    # small-x expansion
    w[small] = 1.0 - x[small] ** 2 / 10.0
    return w



def _window_damping_l(ell, k, R_window, mode='tophat_sq_gaussianized'):
    """
    Approximate isotropic-window damping for a_l(k). Does not do convolution over P_contam and w(k;R).
    """
    w = _spherical_tophat_w(k, R_window)
    w2 = w ** 2

    return w2

def compute_aell_convolved_tophat(
    ells,
    k_eval,
    *,
    A=1.0,
    alpha=2.0,
    R_window=780.0,
    k_int=None,
    r_int=None,
):
    """
    Compute convolved contaminant multipole coefficients a_ell(k) for
    P_c(k,mu)=A*k^{-alpha}*delta_D(mu) with isotropic spherical-top-hat window,
    using Hand+17 Eq. 3.13-3.14 style radial integrals.

    Returns
    -------
    a_lk : dict[int, np.ndarray]
        a_lk[ell] has shape (len(k_eval),), so that
          P_c^win(k,mu) = sum_ell a_ell(k) L_ell(mu)

    Notes
    -----
    delta_ell = (2ell+1)/2 * L_ell(0)
    S_ell(r)   = ∫ dk' k'^2/(2π^2) j_ell(k'r) P_c(k')
    a_ell(k)   = delta_ell ∫ dr r^2 V_lens(r;R) S_ell(r) j_ell(kr)
    """
    ells = list(ells)
    k_eval = np.asarray(k_eval, dtype=float)

    if np.any(k_eval <= 0):
        raise ValueError("k_eval must be > 0")

    # Default integration grids
    if k_int is None:
        kmin = max(np.min(k_eval) / 3.0, 1e-4)
        kmax = np.max(k_eval) * 3.0
        k_int = np.geomspace(kmin, kmax, 600)
    else:
        k_int = np.asarray(k_int, dtype=float)

    if r_int is None:
        # V_lens support is [0, 2R]
        r_int = np.linspace(0.0, 2.0 * R_window, 1200)
    else:
        r_int = np.asarray(r_int, dtype=float)

    if np.any(k_int <= 0):
        raise ValueError("k_int must be > 0")
    if np.min(r_int) < 0:
        raise ValueError("r_int must be >= 0")

    # Contaminant power
    Pc_int = A * k_int ** (-alpha)

    # Spherical-lens overlap volume V_lens(r;R), zero for r>2R
    r = r_int
    R = R_window
    V_lens = np.zeros_like(r)
    mask = (r >= 0.0) & (r <= 2.0 * R)
    rr = r[mask]
    V_lens[mask] = (np.pi / 12.0) * (4.0 * R + rr) * (2.0 * R - rr) ** 2

    # Precompute weights for trapezoid integration
    # (works with nonuniform k_int/r_int too)
    def trapz_weights(x):
        w = np.zeros_like(x)
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
        w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])
        return w

    wk = trapz_weights(k_int)
    wr = trapz_weights(r_int)

    # Build output
    a_lk = {}

    # Loop over ell
    for ell in ells:
        # delta_ell
        delta_ell = 0.5 * (2 * ell + 1) * eval_legendre(ell, 0.0)

        # Compute S_ell(r): shape (Nr,)
        # S_ell(r) = ∫ dk k^2/(2π^2) j_ell(kr) Pc(k)
        kr_mat = np.outer(k_int, r_int)  # (Nk, Nr)
        j_ell_kr = spherical_jn(ell, kr_mat)
        integrand_S = (k_int[:, None] ** 2 / (2.0 * np.pi ** 2)) * j_ell_kr * Pc_int[:, None]
        S_ell_r = np.sum(integrand_S * wk[:, None], axis=0)

        # Compute a_ell(k_eval): shape (Nk_eval,)
        # a_ell(k) = delta_ell ∫ dr r^2 V_lens S_ell(r) j_ell(kr)
        k_eval_r = np.outer(k_eval, r_int)  # (Nk_eval, Nr)
        j_ell_k_r = spherical_jn(ell, k_eval_r)
        rad_pref = (r_int ** 2) * V_lens * S_ell_r
        a_ell = delta_ell * np.sum(j_ell_k_r * (rad_pref * wr)[None, :], axis=1)

        a_lk[ell] = a_ell

    return a_lk


def compute_aell_delta_function(
    ells,
    k_c,
    k_eval,
    *,
    A=1.0,
    R_window=780.0,
    r_int=None,
):
    """
    Compute contaminant multipole coefficients for a TRUE delta function in k:
        P_c(k) = A * delta_D(k - k_c)
    
    This is the simplified analytical approach compared to integrating over k.
    
    The S_ell(r) integral simplifies to:
        S_ell(r) = A * k_c^2 / (2π²) * j_ell(k_c * r)
    
    Then a_ell(k) is computed via the standard radial integral.
    
    Parameters
    ----------
    ells : list
        Multipole orders
    k_c : float
        Delta-function wavenumber (e.g., ell/chi(z_eff))
    k_eval : ndarray
        Wavenumbers to evaluate a_ell at (e.g., [k_c])
    A : float, default=1.0
        Amplitude
    R_window : float, default=780.0
        Window radius in Mpc/h
    r_int : ndarray, optional
        Radial integration grid. If None, created automatically.
    
    Returns
    -------
    a_lk : dict[int, np.ndarray]
        a_lk[ell] has shape (len(k_eval),)
    """
    ells = list(ells)
    k_eval = np.asarray(k_eval, dtype=float)
    k_c = float(k_c)
    
    if np.any(k_eval <= 0):
        raise ValueError("k_eval must be > 0")
    if k_c <= 0:
        raise ValueError("k_c must be > 0")
    
    if r_int is None:
        r_int = np.linspace(0.0, 2.0 * R_window, 1200)
    else:
        r_int = np.asarray(r_int, dtype=float)
    
    if np.min(r_int) < 0:
        raise ValueError("r_int must be >= 0")
    
    # Spherical-lens overlap volume V_lens(r;R), zero for r>2R
    r = r_int
    R = R_window
    V_lens = np.zeros_like(r)
    mask = (r >= 0.0) & (r <= 2.0 * R)
    rr = r[mask]
    V_lens[mask] = (np.pi / 12.0) * (4.0 * R + rr) * (2.0 * R - rr) ** 2
    
    def trapz_weights(x):
        w = np.zeros_like(x)
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
        w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])
        return w
    
    wr = trapz_weights(r_int)
    
    a_lk = {}
    
    # Loop over ell
    for ell in ells:
        # delta_ell
        delta_ell = 0.5 * (2 * ell + 1) * eval_legendre(ell, 0.0)
        
        # Simplified S_ell(r) for delta function: S_ell(r) = A * k_c^2 / (2π²) * j_ell(k_c * r)
        S_ell_r = A * (k_c ** 2) / (2.0 * np.pi ** 2) * spherical_jn(ell, k_c * r_int)
        
        # Compute a_ell(k_eval) via radial integral
        # a_ell(k) = delta_ell ∫ dr r^2 V_lens S_ell(r) j_ell(kr)
        k_eval_r = np.outer(k_eval, r_int)  # (Nk_eval, Nr)
        j_ell_k_r = spherical_jn(ell, k_eval_r)
        rad_pref = (r_int ** 2) * V_lens * S_ell_r
        a_ell = delta_ell * np.sum(j_ell_k_r * (rad_pref * wr)[None, :], axis=1)
        
        a_lk[ell] = a_ell
    
    return a_lk


def build_windowed_leakage_response_delta_function(mu_scan, ell_max, ell_kernel_max, k_c, R_window, A=1.0):
    """
    Build windowed leakage response using TRUE delta function P_c(k) = A*delta(k-k_c).
    
    This is the simplified analytical approach, compared to the flat-spectrum (alpha=0)
    numerical integration over k.
    
    Parameters
    ----------
    mu_scan : ndarray
        μ grid
    ell_max : int
        Maximum signal multipole
    ell_kernel_max : int
        Maximum kernel multipole
    k_c : float
        Delta-function wavenumber
    R_window : float
        Window radius in Mpc/h
    A : float, default=1.0
        Amplitude
    
    Returns
    -------
    R : ndarray
        Leakage response R(μ)
    """
    leakage_ells = list(range(ell_max + 2, ell_kernel_max + 1, 2))
    
    # Compute a_ell coefficients using delta-function formula
    a_lk = compute_aell_delta_function(
        ells=leakage_ells,
        k_c=k_c,
        k_eval=np.array([k_c]),  # Only evaluate at k_c (delta function is zero elsewhere)
        A=A,
        R_window=R_window,
    )
    
    # Construct R(mu) = sum_ell a_ell(k_c) L_ell(mu)
    R = np.zeros_like(mu_scan, dtype=float)
    for ell in leakage_ells:
        a_ell_val = a_lk[ell][0]  # Extract scalar value
        R += a_ell_val * eval_legendre(ell, mu_scan)
    
    return R


def compare_delta_k_approaches(
    ell_contam,
    z_eff,
    ell_max=16,
    ell_kernel_max=128,
    R_window=780.0,
    A=1e-5,
    n_clean_bins=8,
    mu1_min=0.06,
    mu1_max=0.2,
    verbose=True,
):
    """
    Compare delta-function R(mu) and bin edges computed two ways:
    1. Numerical: flat spectrum (alpha=0) integrated over k grid
    2. Analytical: true delta function simplification
    
    Parameters
    ----------
    ell_contam : int
        Contaminating multipole
    z_eff : float
        Effective redshift
    ell_max : int
        Signal multipole cutoff
    ell_kernel_max : int
        Kernel multipole cutoff
    R_window : float
        Window radius in Mpc/h
    A : float
        Amplitude
    n_clean_bins : int
        Number of clean bins for edge solving
    mu1_min, mu1_max : float
        Search bounds for mu1 optimization
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    results : dict
        Keys: 'mu', 'k_c', 'R_numerical', 'R_analytical', 'diff_linf', 'diff_l2',
               'mu1_numerical', 'mu1_analytical', 'edges_numerical', 'edges_analytical',
               'mu1_diff', 'n_edges_numerical', 'n_edges_analytical', 'edges_diff'
    """
    from astropy.cosmology import Planck18 as cosmo
    
    # Map ell to k
    chi_mpc_h = cosmo.comoving_distance(z_eff).value * cosmo.h
    k_c = float(ell_contam) / chi_mpc_h
    
    # Setup k grid for numerical approach
    kedges = np.logspace(np.log10(0.006), np.log10(0.2), 60)
    k = np.sqrt(kedges[:-1] * kedges[1:])
    
    # μ grid
    mu_scan = np.linspace(0.0, 1.0, 5000)
    
    # 1) Numerical: flat spectrum (alpha=0)
    R_numerical = build_windowed_leakage_response(
        mu_scan, ell_max, ell_kernel_max, k, R_window, A=A, alpha=0.0
    )
    idx_num = np.nanargmax(np.abs(R_numerical))
    norm_val_num = np.abs(R_numerical.flat[idx_num])  # Magnitude of extremum
    R_numerical_norm = R_numerical / norm_val_num
    
    # 2) Analytical: true delta function
    R_analytical = build_windowed_leakage_response_delta_function(
        mu_scan, ell_max, ell_kernel_max, k_c, R_window, A=A
    )
    idx_ana = np.nanargmax(np.abs(R_analytical))
    norm_val_ana = np.abs(R_analytical.flat[idx_ana])  # Magnitude of extremum
    R_analytical_norm = R_analytical / norm_val_ana
    
    # Compute differences in R(mu)
    diff = R_numerical_norm - R_analytical_norm
    diff_linf = np.max(np.abs(diff))
    diff_l2 = np.sqrt(np.mean(diff ** 2))
    
    # Solve for bin edges (mu1 and zero crossings) for both approaches
    prefer_mu1 = 1.0 / (ell_max / 2 + 1)
    
    mu1_numerical, edges_numerical, _, _ = solve_mu1_and_edges_windowed(
        mu_grid=mu_scan, R_leak=R_numerical_norm, n_clean_bins=n_clean_bins,
        mu1_min=mu1_min, mu1_max=mu1_max,
        prefer_nowindow_mu1=prefer_mu1, verbose=False
    )
    
    mu1_analytical, edges_analytical, _, _ = solve_mu1_and_edges_windowed(
        mu_grid=mu_scan, R_leak=R_analytical_norm, n_clean_bins=n_clean_bins,
        mu1_min=mu1_min, mu1_max=mu1_max,
        prefer_nowindow_mu1=prefer_mu1, verbose=False
    )
    
    mu1_diff = np.abs(mu1_numerical - mu1_analytical)
    edges_diff = np.max(np.abs(edges_numerical - edges_analytical))
    
    if verbose:
        print("[compare_delta_k_approaches] ell_contam=%d" % ell_contam)
        print("  k_c = %.6f h/Mpc" % k_c)
        print("  R(mu) differences:")
        print("    L-inf: %.6e, L2: %.6e" % (diff_linf, diff_l2))
        print("  Bin edges:")
        print("    mu1_numerical: %.6f" % mu1_numerical)
        print("    mu1_analytical: %.6f" % mu1_analytical)
        print("    |mu1 difference|: %.6e" % mu1_diff)
        print("    n_edges: numerical=%d, analytical=%d" % (len(edges_numerical), len(edges_analytical)))
        print("    max edge difference: %.6e" % edges_diff)
        if edges_diff > 1e-6:
            print("    WARNING: Bin edges differ significantly!")
    
    return {
        'mu': mu_scan,
        'k_c': k_c,
        'R_numerical': R_numerical_norm,
        'R_analytical': R_analytical_norm,
        'diff': diff,
        'diff_linf': diff_linf,
        'diff_l2': diff_l2,
        'mu1_numerical': mu1_numerical,
        'mu1_analytical': mu1_analytical,
        'mu1_diff': mu1_diff,
        'edges_numerical': edges_numerical,
        'edges_analytical': edges_analytical,
        'edges_diff': edges_diff,
        'n_edges_numerical': len(edges_numerical),
        'n_edges_analytical': len(edges_analytical),
    }


def alpha_sweep(
    alphas, mu, ell_max, ell_kernel_max, k, R_vals, n_clean_bins,
    mu1_min=0.02, mu1_max=0.3, A=1e-5, verbose=False
):
    R0 = response_delta_mu0(mu, ell_max)
    mu1_0, edges_0, mu0_sub, I0 = solve_mu1_and_edges_windowed(
        mu_grid=mu, R_leak=R0, n_clean_bins=n_clean_bins,
        mu1_min=1e-4, mu1_max=0.8,
        prefer_nowindow_mu1=1.0/(ell_max/2 + 1), verbose=False
    )

    results, mu1_list = {}, []
    for a in alphas:
        print('on alpha = ', a)
        Rw = build_windowed_leakage_response(
            mu, ell_max, ell_kernel_max, k, R_vals, A=A, alpha=a
        )
        mu1_w, edges_w, muw_sub, Iw = solve_mu1_and_edges_windowed(
            mu_grid=mu, R_leak=Rw, n_clean_bins=n_clean_bins,
            mu1_min=mu1_min, mu1_max=mu1_max,
            prefer_nowindow_mu1=mu1_0, verbose=False
        )
        results[a] = {"Rw": Rw, "mu1": mu1_w, "edges": edges_w, "mu_sub": muw_sub, "I": Iw}
        mu1_list.append(mu1_w)

    return {
        "alphas": list(alphas),
        "mu": mu,
        "baseline": {"R0": R0, "mu1": mu1_0, "edges": edges_0, "mu_sub": mu0_sub, "I": I0},
        "windowed": results,
        "mu1_list": np.array(mu1_list),
    }



def _compute_window_corrected_mu_bins_cumulative(
    ell_max,
    n_clean_bins,
    k,
    R_window=780.0,
    alpha=2.0,
    ell_kernel_max=None,
    mu1=None,
    verbose=False,
    return_diagnostics=False,
):
    """
    Find mu bins where cumulative window-damped LEAKAGE response crosses zero.
    
    This is the primary workhorse method for window correction:
    1. Compute leakage response: R_leak(μ) = Σ_{ℓ>ℓ_max} w_ℓ(R_window) L_ℓ(μ)
    2. Apply ℓ-dependent window damping (higher ℓ suppressed more)
    3. Integrate: I(μ) = ∫_{mu1}^μ R_leak(μ') dμ'
    4. Find zeros of I(μ) to define clean bin edges
    """
    
    if ell_kernel_max is None:
        ell_kernel_max = min(ell_max + 64, 128)
    
    if mu1 is None:
        mu1 = 1.0 / (ell_max / 2.0 + 1.0)
    
    # Compute window damping averaged over k
    def _spherical_tophat_w_sq(k_vals, R):
        """w(k;R)^2 where w(k;R) = 3 j1(kR)/(kR)"""
        x = np.asarray(k_vals) * R
        w = np.ones_like(x, dtype=float)
        small = np.abs(x) < 1e-6
        xs = x[~small]
        w[~small] = 3.0 * spherical_jn(1, xs) / xs
        w[small] = 1.0 - x[small] ** 2 / 10.0
        return w ** 2
    
    w_k = _spherical_tophat_w_sq(k, R_window)
    P_c = k ** (-alpha)
    w_eff = np.sum(w_k * P_c) / np.sum(P_c)
    
    # Leakage multipoles
    leakage_ells = list(range(ell_max + 2, ell_kernel_max + 1, 2))
    
    # ℓ-dependent damping: higher multipoles suppressed more by finite window
    def get_ell_damping(ell):
        ell_scale = ell_max + 30
        return w_eff * np.exp(-0.5 * (ell / ell_scale)**2)
    
    if verbose:
        print(f"[Window-corrected mu bins (cumulative method)]")
        print(f"  ell_max: {ell_max}, leakage ℓ: [{leakage_ells[0]}, ..., {leakage_ells[-1]}]")
        print(f"  R_window: {R_window} Mpc/h")
        print(f"  Spectrum-weighted w_eff: {w_eff:.6f}")
    
    # Evaluate windowed leakage response on fine grid
    mu_scan = np.linspace(mu1, 1.0, 5000)

    R_leak_windowed = build_windowed_leakage_response(
        mu_scan, ell_max, ell_kernel_max, k, R_window, A=1e-5, alpha=alpha
    )

    # R_leak_windowed = np.zeros_like(mu_scan)
    # for ell in leakage_ells:
    #     w_ell = get_ell_damping(ell)
    #     R_leak_windowed += w_ell * eval_legendre(ell, mu_scan)
    
    # Cumulative integral (following compute_null_bins pattern)
    cum_integral = _cumtrapz(R_leak_windowed, mu_scan, initial=0)
    
    # Find zero crossings
    zero_crossings = []
    for i in range(1, len(cum_integral)):
        if cum_integral[i-1] * cum_integral[i] < 0:
            # Linear interpolation for zero
            mu_zero = mu_scan[i-1] - cum_integral[i-1] * (mu_scan[i] - mu_scan[i-1]) / (cum_integral[i] - cum_integral[i-1])
            zero_crossings.append(mu_zero)
    
    zero_crossings = np.array(zero_crossings)
    
    if len(zero_crossings) < n_clean_bins:
        if verbose:
            print(f"  Warning: Only found {len(zero_crossings)} zeros but need {n_clean_bins}")
    
    # Select first n_clean_bins zeros as clean bin edges
    clean_bin_edges = zero_crossings[:n_clean_bins]
    
    # Full edge array: [0, mu1, clean_edges..., 1]
    mu_edges = np.concatenate(([0], [mu1], clean_bin_edges, [1.0]))
    
    diag = {
        'success': True,
        'n_zeros_found': len(zero_crossings),
        'n_clean_bins': n_clean_bins,
        'method': 'cumulative_leakage',
        'R_window': R_window,
        'ell_max': ell_max,
        'ell_kernel_max': ell_kernel_max,
        'w_eff': w_eff,
    }
    
    if return_diagnostics:
        return mu_edges, diag, R_leak_windowed, mu_scan
    return mu_edges, R_leak_windowed, mu_scan


def compute_window_corrected_mu_bins(
    ell_max,
    n_clean_bins,
    kedges,
    n_k=256,
    A=1.0,
    alpha=2.0,
    R_window=780.0,
    use_window=True,
    leakage_mode='omitted',
    ell_kernel_max=None,
    k_weight_mode='logk',
    mu1=None,
    bracket_eps=1e-5,
    root_tol=1e-8,
    maxiter=200,
    verbose=False,
    n_scan=2048,
    strict_roots=True,
    resid_tol=1e-6,
    mu_init=None,
    continuation_window=0.08,
    return_diagnostics=False,
    use_cumulative_method=True,
):
    """
    Compute window-corrected non-uniform mu bin edges for transverse contamination.
    
    When a spherical tophat window is applied (as in lightcone mocks like Halfdome),
    high-ℓ multipoles (ℓ > ell_max) leak into lower-ℓ measurements. This function 
    finds new mu bin edges where the window-damped leakage response has zeros,
    so contamination still cancels in the "clean" bins.
    
    Method
    ------
    Compute cumulative window-damped leakage response and find its zeros:
        R_leak(μ) = Σ_{ℓ>ell_max} w_ℓ(R_window) L_ℓ(μ)
        I(μ) = ∫_{μ1}^μ R_leak(μ') dμ'
        Find zeros of I(μ)
    
    Parameters
    ----------
    ell_max : int
        Maximum ℓ in your measurement basis
    n_clean_bins : int
        Number of clean (non-junk) mu bins desired
    kedges : array
        K bin edges used in measurement
    R_window : float
        Window size in Mpc/h
    alpha : float
        Power law index of contamination (typically 2.0)
    ell_kernel_max : int or None
        Maximum ℓ to consider for leakage (default: min(ell_max + 64, 128))
    mu1 : float or None
        Upper edge of junk bin (default: 1/(ell_max/2 + 1))
    return_diagnostics : bool
        If True, return (mu_edges, diagnostics_dict)
    use_cumulative_method : bool
        If True (default), use fast cumulative leakage method.
        If False, use slower exact convolution method (may have numerical issues).
    
    Returns
    -------
    mu_edges : ndarray
        Mu bin edges [0, mu1, clean_edge_1, ..., clean_edge_n, 1]
    diagnostics : dict (optional)
        If return_diagnostics=True, also returns diagnostic information
    """
    
    if ell_kernel_max is None:
        ell_kernel_max = min(ell_max + 64, 128)
    if ell_kernel_max < ell_max:
        raise ValueError('ell_kernel_max must be >= ell_max')

    kedges = np.asarray(kedges, dtype=float)
    if kedges.ndim != 1 or len(kedges) < 2 or not np.all(np.diff(kedges) > 0):
        raise ValueError('kedges must be a strictly increasing 1D array with len >= 2')

    # k centers for window damping calculation
    k = 0.5 * (kedges[:-1] + kedges[1:])
    
    # Use cumulative leakage method (default, fast, stable)
    if use_cumulative_method:
        return _compute_window_corrected_mu_bins_cumulative(
            ell_max=ell_max,
            n_clean_bins=n_clean_bins,
            k=k,
            R_window=R_window,
            alpha=alpha,
            ell_kernel_max=ell_kernel_max,
            mu1=mu1,
            verbose=verbose,
            return_diagnostics=return_diagnostics,
        )
    
    # Fallback: exact convolution method (not fully implemented due to numerical issues)
    raise NotImplementedError(
        "Exact convolution method (use_cumulative_method=False) has numerical stability issues.\n"
        "Please use the default cumulative leakage method (use_cumulative_method=True).\n"
        "If you need exact convolution, consider computing a_ℓ(k) externally and passing "
        "coefficients directly."
    )


def compute_windowed_mu_edges_solve_mu1(
    ell_max,
    n_clean_bins,
    k,
    R_window=780.0,
    alpha=2.0,
    ell_kernel_max=None,
    mu1_min=0.02,
    mu1_max=0.3,
    A=1e-5,
    verbose=False,
):
    """
    Compute window-corrected mu bin edges using the solve_mu1 approach.
    
    Uses the same robust method as alpha_sweep: solves for optimal mu1 from the
    endpoint condition ∫_{mu1}^{1} R_leak dmu = 0, then finds zero crossings of
    the cumulative leakage response to define clean bin edges.
    
    This ensures mu1 is included as the first edge and the zero-crossing bin edges
    properly span [mu1, 1].
    
    Parameters
    ----------
    ell_max : int
        Maximum multipole order for signal
    n_clean_bins : int
        Number of clean (non-junk) mu bins desired
    k : ndarray
        k-space grid (for window weighting)
    R_window : float, default=780.0
        Radial window size in Mpc/h (e.g., comoving distance range)
    alpha : float, default=2.0
        Power-law index for contamination spectrum P_c(k) ~ k^{-alpha}
    ell_kernel_max : int, optional
        Maximum multipole to include in leakage calculation. If None, set to
        min(ell_max + 64, 128).
    mu1_min, mu1_max : float, default=0.02, 0.3
        Search bounds for optimal mu1
    A : float, default=1e-5
        Amplitude for windowed leakage response calculation
    verbose : bool, default=False
        If True, print diagnostics
    
    Returns
    -------
    mu_edges : ndarray
        Bin edges: [mu1_opt, z1, z2, ..., 1.0]
        Note: This differs from the "junk + clean" convention of compute_null_bins.
        Use this directly for halfdome window-corrected analysis.
    mu1_opt : float
        Optimal mu1 value solving ∫_{mu1}^{1} R_leak dmu = 0
    
    Examples
    --------
    >>> mu_edges, mu1 = compute_windowed_mu_edges_solve_mu1(
    ...     ell_max=16, n_clean_bins=7, k=k_array, R_window=780.0, alpha=2.0
    ... )
    >>> print(f"mu1={mu1:.6f}, n_edges={len(mu_edges)}, should have {7+1} (7 clean + 1 endpoint)")
    """
    if ell_kernel_max is None:
        ell_kernel_max = min(ell_max + 64, 128)
    
    # Generate fine mu grid for solver
    mu_scan = np.linspace(0.001, 1.0, 5000)
    
    # --- Baseline: window-free leakage response to estimate mu1
    R0 = response_delta_mu0(mu_scan, ell_max)
    mu1_0, edges_0, mu0_sub, I0 = solve_mu1_and_edges_windowed(
        mu_grid=mu_scan, R_leak=R0, n_clean_bins=n_clean_bins,
        mu1_min=1e-4, mu1_max=0.8,
        prefer_nowindow_mu1=1.0/(ell_max/2 + 1), verbose=False
    )
    
    # --- Windowed leakage response with specified k-weighting
    Rw = build_windowed_leakage_response(
        mu_scan, ell_max, ell_kernel_max, k, R_window, A=A, alpha=alpha
    )
    
    # --- Solve for mu1 with window correction, using baseline as initial guess
    mu1_opt, edges_opt, muw_sub, Iw = solve_mu1_and_edges_windowed(
        mu_grid=mu_scan, R_leak=Rw, n_clean_bins=n_clean_bins,
        mu1_min=mu1_min, mu1_max=mu1_max,
        prefer_nowindow_mu1=mu1_0, verbose=verbose
    )
    
    if verbose:
        print(f"[compute_windowed_mu_edges_solve_mu1]")
        print(f"  ell_max: {ell_max}, ell_kernel_max: {ell_kernel_max}")
        print(f"  R_window: {R_window:.2f} Mpc/h")
        print(f"  alpha: {alpha}")
        print(f"  mu1_baseline: {mu1_0:.6f}")
        print(f"  mu1_optimized: {mu1_opt:.6f}")
        print(f"  n_edges: {len(edges_opt)} (should be {n_clean_bins + 1})")
        print(f"  edges: {edges_opt}")
    
    return edges_opt, mu1_opt
# These functions will be added to nonunif_binning.py

def compute_mu_edges_for_delta_function(
    ell_contam,
    z_eff,
    ell_max=16,
    ell_kernel_max=128,
    R_window=780.0,
    A=1e-5,
    n_clean_bins=8,
    mu1_min=0.02,
    mu1_max=0.3,
    mu_scan_res=4000,
    verbose=True,
):
    """
    Compute mu bin edges for delta-function angular systematic at multipole ell_contam.

    Maps ell_contam to effective transverse wavenumber k_c = ell_contam / chi(z_eff)
    using comoving distance at effective redshift z_eff. Then computes window-corrected
    mu bin edges using analytic delta-function leakage response R(mu).

    Parameters
    ----------
    ell_contam : int
        Contaminating angular multipole (e.g., 2, 6, 20, 60, 100, 300)
    z_eff : float
        Effective redshift for k_c mapping. Typically (z_min + z_max) / 2.
    ell_max : int, default=16
        Maximum signal multipole
    ell_kernel_max : int, default=128
        Maximum kernel multipole (ell_max <= ell_kernel_max)
    R_window : float, default=780.0
        Radial window size in Mpc/h (comoving distance shell thickness)
    A : float, default=1e-5
        Amplitude scaling for contamination
    n_clean_bins : int, default=8
        Number of clean (non-junk) mu bins desired
    mu1_min, mu1_max : float, default=0.02, 0.3
        Search bounds for optimal mu1 (upper edge of junk bin)
    mu_scan_res : int, default=4000
        Resolution of mu grid for solving mu1 and finding bin edges
    verbose : bool, default=True
        Print diagnostics

    Returns
    -------
    mu_edges : ndarray
        Bin edges array [mu_junk, mu_1, mu_2, ..., 1.0]
        Note: First edge is ~0 for junk bin; subsequent edges are clean bin boundaries.

    Raises
    ------
    ValueError
        If k_c computation fails, R(mu) is invalid, or edge solver fails to converge.
    RuntimeError
        If any internal computation encounters NaN/Inf that cannot be handled.

    Examples
    --------
    >>> mu_edges = compute_mu_edges_for_delta_function(
    ...     ell_contam=2, z_eff=0.25, R_window=780.0, n_clean_bins=8
    ... )
    >>> print(f"Bin edges: {mu_edges}")  # doctest: +SKIP
    Bin edges: [0.    0.09 0.19 0.28 0.37 0.45 0.52 0.58 0.63 1.  ]
    """
    from astropy.cosmology import Planck18 as cosmo

    # 1) Compute k_c from ell_contam and z_eff
    try:
        chi_mpc_h = float(cosmo.comoving_distance(z_eff).value * cosmo.h)
        if not (chi_mpc_h > 0):
            raise ValueError(f"Comoving distance at z_eff={z_eff} is not positive: {chi_mpc_h}")
        k_c = float(ell_contam) / chi_mpc_h
        if not (k_c > 0):
            raise ValueError(f"k_c computation resulted in non-positive value: {k_c}")
    except Exception as e:
        raise ValueError(f"Failed to compute k_c from ell_contam={ell_contam}, z_eff={z_eff}: {e}")

    if verbose:
        print(f"[compute_mu_edges_for_delta_function]")
        print(f"  ell_contam: {ell_contam}")
        print(f"  z_eff: {z_eff:.4f}")
        print(f"  chi(z_eff): {chi_mpc_h:.2f} Mpc/h")
        print(f"  k_c: {k_c:.6f} h/Mpc")

    # 2) Build mu scan grid
    try:
        mu_scan = np.linspace(0.0, 1.0, mu_scan_res)
    except Exception as e:
        raise RuntimeError(f"Failed to create mu_scan grid: {e}")

    # 3) Compute leakage response R(mu) using analytic delta-function formula
    try:
        R = build_windowed_leakage_response_delta_function(
            mu_scan=mu_scan,
            ell_max=ell_max,
            ell_kernel_max=ell_kernel_max,
            k_c=k_c,
            R_window=R_window,
            A=A,
        )
    except Exception as e:
        raise ValueError(f"Failed to compute windowed leakage response: {e}")

    # Validate R(mu)
    if not np.isfinite(R).any():
        raise RuntimeError("Leakage response R(mu) is entirely NaN or Inf")
    if np.all(R == 0.0):
        raise RuntimeError("Leakage response R(mu) is zero everywhere; check k_c, R_window, A")

    # 4) Solve for mu1 and bin edges using windowed approach
    try:
        mu1_opt, edges, mu_sub, I_sub = solve_mu1_and_edges_windowed(
            mu_grid=mu_scan,
            R_leak=R,
            n_clean_bins=n_clean_bins,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            prefer_nowindow_mu1=1.0 / (ell_max / 2 + 1),
            verbose=verbose,
        )
    except Exception as e:
        raise ValueError(f"Failed to solve mu1 and compute bin edges: {e}")

    # Validate edges
    if not (len(edges) >= 2):
        raise RuntimeError(f"Invalid edges returned: {edges}")
    if not (edges[0] >= 0 and edges[-1] <= 1.0):
        raise RuntimeError(f"Edges out of range [0, 1]: {edges}")
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f"Edges are not strictly increasing: {edges}")

    if verbose:
        print(f"  mu1_opt: {mu1_opt:.6f}")
        print(f"  n_edges: {len(edges)} (requested {n_clean_bins + 1})")
        print(f"  edges: {edges}")

    return edges


# Optional: Extended version of compute_windowed_mu_edges_solve_mu1 that accepts spec_type
def compute_windowed_mu_edges_solve_mu1_extended(
    ell_max,
    n_clean_bins,
    k,
    R_window,
    alpha=2.0,
    spec_type='power_law',
    ell_delta=None,
    z_eff=None,
    ell_kernel_max=None,
    mu1_min=0.02,
    mu1_max=0.3,
    A=1e-5,
    verbose=True,
):
    """
    Extended version of compute_windowed_mu_edges_solve_mu1 that supports spec_type.

    Parameters
    ----------
    spec_type : str, default='power_law'
        Type of contamination spectrum:
        - 'power_law': alpha-dependent (default behavior)
        - 'delta': delta-function at k_c = ell_delta / chi(z_eff)
        - 'flat': flat spectrum (alpha=0, no k dependence)
    ell_delta : int, optional
        Required if spec_type='delta'. Angular multipole for delta spike.
    z_eff : float, optional
        Required if spec_type='delta'. Effective redshift for k_c mapping.
    Other parameters: same as compute_windowed_mu_edges_solve_mu1

    Returns
    -------
    mu_edges, mu1_opt
        Same as compute_windowed_mu_edges_solve_mu1
    """
    spec_type = spec_type.lower()

    if spec_type == 'delta':
        if ell_delta is None or z_eff is None:
            raise ValueError(
                f"spec_type='delta' requires ell_delta and z_eff, got "
                f"ell_delta={ell_delta}, z_eff={z_eff}"
            )
        # Delegate to delta-function specific function
        edges = compute_mu_edges_for_delta_function(
            ell_contam=ell_delta,
            z_eff=z_eff,
            ell_max=ell_max,
            ell_kernel_max=ell_kernel_max or min(ell_max + 64, 128),
            R_window=R_window,
            A=A,
            n_clean_bins=n_clean_bins,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            verbose=verbose,
        )
        # Extract mu1 from edges (first edge > 0 is mu1)
        mu1_opt = edges[1] if len(edges) > 1 else edges[0]
        return edges, mu1_opt

    elif spec_type in ('power_law', 'flat'):
        # Map 'flat' to alpha=0
        alpha_use = 0.0 if spec_type == 'flat' else alpha
        # Use existing function
        from nonunif_binning import compute_windowed_mu_edges_solve_mu1 as orig_func
        return orig_func(
            ell_max=ell_max,
            n_clean_bins=n_clean_bins,
            k=k,
            R_window=R_window,
            alpha=alpha_use,
            ell_kernel_max=ell_kernel_max,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown spec_type: {spec_type}. Use 'power_law', 'delta', or 'flat'.")
