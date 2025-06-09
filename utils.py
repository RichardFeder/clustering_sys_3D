import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import simps


def init_test_params(**kwargs):

    test_params = dict({'galtype':'ELG', 'zmin':0.8, 'zmax':1.6, 'P0':3000, 'mock_type':'EZmock', 'desi_year':1, 'nmesh':512})

    test_params.update(kwargs)

    return test_params

def grab_chi_interp(zmin=0.0, zmax=2.0, nz=10000):
        
    zgrid = np.linspace(zmin, zmax, nz)
    chigrid = cosmo.comoving_distance(zgrid).to(u.Mpc).value
    chi_interp = interp1d(zgrid, chigrid, kind='linear', bounds_error=False, fill_value='extrapolate')

    return chi_interp


def fast_convert_to_posarray(chi_interp, ra, dec, z):
    cat_r = chi_interp(z)

    pos_data = np.array([ra, dec, cat_r])

    return pos_data


def compute_shotnoise(weights, volume):
    w = np.asarray(weights)
    return (np.sum(w**2) / np.sum(w)**2) * volume

def compute_veff_from_galaxies(galaxy_z, weights, P0_k, nbins=100, sky_area_deg2=14000, cosmo=cosmo):
    """
    Compute V_eff(k) from per-galaxy redshifts and weights.

    Parameters
    ----------
    galaxy_z : array-like
        Redshifts of galaxies.
    weights : array-like
        Per-galaxy weights (e.g., systematics × FKP).
    P0_k : float
        Power spectrum value at given scale k [Mpc^3 / h^3].
    nbins : int
        Number of redshift bins.
    sky_area_deg2 : float
        Survey area in deg^2.
    cosmo : astropy.cosmology
        Cosmology instance.

    Returns
    -------
    Veff : float
        Effective volume in [Mpc/h]^3.
    """
    h = cosmo.h
    area_sr = sky_area_deg2 * (np.pi / 180.0)**2  # convert area to steradians

    # Bin the redshifts and sum weights in each bin
    z_bins = np.linspace(np.min(galaxy_z), np.max(galaxy_z), nbins + 1)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    dz = z_bins[1] - z_bins[0]

    sum_weights, _ = np.histogram(galaxy_z, bins=z_bins, weights=weights)

    # Compute comoving volume per bin
    dVc_dz = cosmo.differential_comoving_volume(z_centers).value  # [Mpc^3 / sr / dz]
    dVc = dVc_dz * area_sr * dz  # [Mpc^3]

    # Compute weighted n(z) [h^3 / Mpc^3]
    nbar = sum_weights / dVc  # [gal / Mpc^3]
    nbar *= h**3  # [h^3 / Mpc^3]

    # Effective volume integrand
    factor = (nbar * P0_k / (1 + nbar * P0_k))**2
    integrand = factor * dVc  # [Mpc^3]

    Veff = simps(integrand, z_centers)  # integrate over z
    return Veff