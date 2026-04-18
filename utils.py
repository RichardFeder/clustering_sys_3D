import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
import astropy.units as u


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


def gen_interp_fn_dcom_z(zmin=0, zmax=10):
    ''' Compute interpolated function for chi --> z. Assumes Planck18 '''
    h = cosmo.h

    _z_grid = np.linspace(zmin, zmax, 10000)
    _dcom_grid = cosmo.comoving_distance(_z_grid).value  # in Mpc
    
    # Create inverse interpolator: D_c -> z
    dcom_to_z_interp = interp1d(_dcom_grid, _z_grid, kind='cubic', bounds_error=False, fill_value='extrapolate')

    return dcom_to_z_interp

def comoving_distance_to_redshift(dc_mpc, dcom_to_z_interp):
    """
    Vectorized conversion of comoving distance (Mpc) to redshift using interpolation.
    
    Parameters
    ----------
    dc_mpc : float or array-like
        Comoving distance(s) in Mpc.

    dcom_to_z_interp : float or array-like
        Interpolating function to query redshift given distance

    Returns
    -------
    z : float or ndarray
        Redshift(s) corresponding to the input comoving distances.
    """
    dc_mpc = np.atleast_1d(dc_mpc)
    z = dcom_to_z_interp(dc_mpc)
    return z if z.ndim > 0 else z.item()

def convert_to_ra_dec_distance(galpos_mpc_per_h, L_box_mpc_per_h, center_offset_mpc=None):
    ''' Takes 3D positions and converts to RA/DEC/chi'''

    x_orig, y_orig, z_orig = galpos_mpc_per_h[:,0], galpos_mpc_per_h[:,1], galpos_mpc_per_h[:,2]
    if center_offset_mpc is None:
        center_offset_mpc = L_box_mpc_per_h / 2.0

    print('Center offset [Mpc/h]:', center_offset_mpc)
        
    x_relative = x_orig - center_offset_mpc
    y_relative = y_orig - center_offset_mpc
    z_relative = z_orig - center_offset_mpc

    r = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2) * u.Mpc
    
    ra_rad = np.arctan2(y_relative, x_relative)
    ra_deg = np.degrees(ra_rad)
    # Ensure RA is in [0, 360)
    ra_deg[ra_deg < 0] += 360

    dec_rad = np.arcsin(z_relative / r.value) # .value to remove units for arcsin
    dec_deg = np.degrees(dec_rad)

    return ra_deg, dec_deg, r
    