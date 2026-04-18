import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz as _cumtrapz
from scipy.special import legendre

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