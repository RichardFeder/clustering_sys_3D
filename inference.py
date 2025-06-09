import numpy as np
import sys

sys.path.append('/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20240118-1.0.0/code/pyclass/main/lib/python3.10/site-packages/')
sys.path.append('/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20240118-1.0.0/code/desilike/main/lib/python3.10/site-packages/')

import cosmoprimo
import desilike
import pyclass

from cosmoprimo.fiducial import DESI

from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.parameter import ParameterCollection
from desilike import setup_logging

from pypower import PowerSpectrumStatistics

from desilike.profilers import MinuitProfiler

from astropy.cosmology import Planck18 as cosmo




def desi_mock_cov(kmin=0., kmax=0.3, nedges=61, z=1.3, fnl=0., b1=1.4, shotnoise=1e3, volume=6.4e9, \
                 ells=(0, 2)):


    cosmo = DESI()

    edges = np.linspace(kmin, kmax, nedges)

    k = (edges[:-1] + edges[1:]) / 2.
    nmodes = 4. * np.pi / 3. * (edges[1:]**3 - edges[:-1]**3)

    fo = cosmo.get_fourier()
    pk = fo.pk_interpolator(of='delta_cb')(k, z=z)
    pk_prim = cosmo.get_primordial(mode='scalar').pk_interpolator()(k)
    pphi_prim = 9 / 25 * 2 * np.pi**2 / k**3 * pk_prim / cosmo.h**3
    alpha = 1. / (pk / pphi_prim)**0.5

    # PNG response of dark matter halos in spherical collapse
    bphi = 2. * 1.686 * (b1 - 1.)
    b = b1 + bphi * fnl * alpha
    f = fo.sigma8_z(z, of='theta_cb') / fo.sigma8_z(z, of='delta_cb')

    if shotnoise is None:
        shotnoise = 1 / 1e-4
        
    poles = []
    poles.append((b**2 + 2. / 3. * f * b + 1. / 5. * f**2) * pk + shotnoise)
    poles.append((4. / 3. * f * b + 4. / 7. * f**2) * pk)
    poles = np.array(poles, dtype='f8')

    mean = PowerSpectrumStatistics(edges, k, poles, nmodes=nmodes, ells=ells, shotnoise_nonorm=shotnoise, statistic='multipole')
    cov = [2. * (2. * np.pi)**3 / (2 * ell + 1) / (volume * nmodes) * poles[0]**2 for ell in ells]
    cov = np.diag(np.concatenate(cov, axis=0))
    
    rng = np.random.RandomState(seed=42)
    mocks = []
    for i in range(1000):
        tmp = mean.deepcopy()
        tmp.power_nonorm.flat[...] = rng.multivariate_normal(mean.power_nonorm.ravel(), cov)
        mocks.append(tmp)
        
    data, mocks = mocks[0], mocks[1:]

    return data, mocks, k



def inference_profile_likelihood(likelihood, seed=42, niterations=5):
    
    # Seed used to decide on starting point
    profiler = MinuitProfiler(likelihood, seed=seed)
    # Find best fit, starting from 5 different starting points
    # NOTE: With MPI, these runs are performed in parallel
    profiles = profiler.maximize(niterations=niterations)

    return profiles


def pk_ratio_fnl(k, fnl, b1, z=1.0):
    """
    Return the ratio P_fnl(k) / P_gaussian(k) for local PNG.

    Parameters
    ----------
    k : array-like
        Wavenumbers [h/Mpc].
    f_nl : float
        Local PNG parameter.
    b1 : float
        Linear bias of the tracer.
    bphi : float
        PNG response coefficient.
    alpha_k : array-like or None
        Optional array of alpha(k) values; if None, use default k^{-2} scaling.

    Returns
    -------
    ratio : array-like
        Theoretical ratio of P(k)_fnl / P(k)_gaussian.
    """

    cosmo = DESI()

    # edges = np.linspace(kmin, kmax, nedges)

    # k = (edges[:-1] + edges[1:]) / 2.
    # nmodes = 4. * np.pi / 3. * (edges[1:]**3 - edges[:-1]**3)

    fo = cosmo.get_fourier()
    pk = fo.pk_interpolator(of='delta_cb')(k, z=z)
    pk_prim = cosmo.get_primordial(mode='scalar').pk_interpolator()(k)
    pphi_prim = 9 / 25 * 2 * np.pi**2 / k**3 * pk_prim / cosmo.h**3
    alpha = 1. / (pk / pphi_prim)**0.5

    # PNG response of dark matter halos in spherical collapse
    bphi = 2. * 1.686 * (b1 - 1.)
    b = b1 + bphi * fnl * alpha

    # if alpha_k is None:
    #     # Normalized scale-dependent shape
    #     alpha_k = (1e-2 / k)**2

    # delta_b = f_nl * bphi * alpha_k
    ratio = (b / b1)**2
    return ratio

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

    
