import os
import tempfile

import numpy as np
from matplotlib import pyplot as plt

import healpy as hp
from tqdm import tqdm
from scipy.stats import rv_histogram, norm, uniform

from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic, ICRS
from astropy import units as u
from astropy.coordinates import FK5, FK4 # Import these for potential transformations if needed, but ICRS is usually the target.

from star_sim import *


class contam():

    ''' Contamination module for generating contamination catalogs or emulating selection function errors '''
    
    def __init__(self, pos_fn=None, zerr_fn=None, zcontam=None, nside=256):
        self._pos_fn = pos_fn
        self._zerr_fn = zerr_fn
        self._zcontam_fn = zcontam
        self.nside = nside

        self.npix = hp.nside2npix(self.nside)
            
            
    def load_pos_fn(self, pos_fn):
        ''' Loads into function that can be called easily '''
        self.pos_fn = pos_fn

    def load_zerr_fn(self, zerr_fn):
        self.zerr_fn = zerr_fn

    def load_zcontam_kernel(self, zcontam):
        ''' Defines redshift distribution of contaminants '''
        self.zcontam = zcontam

    
    def apply_zerr(self, ztrue, ra=None, dec=None):
        if self.zerr_fn is None:
            return ztrue
    
        ztrue = np.asarray(ztrue)
        ra = np.asarray(ra) if ra is not None else None
        dec = np.asarray(dec) if dec is not None else None
    
        return ztrue + self.zerr_fn(ztrue, ra, dec)

    def generate_contam_catalog(self, ncontam):
        ''' 
        Additive contamination
        ncontam : number of contaminating sources
        
        '''
        contam_ra, contam_dec = self.pos_fn(ncontam)

        contam_z = self.zcontam(ncontam)

        return contam_ra, contam_dec, contam_z


    def stellar_contam_gen(self, N_gal, redshift_contam_kernel=None, zmin=None, zmax=None, frac=0.01, mask=None, plot=True):

        ''' 
        Wrapper function to generate catalog of contaminants

        Assumes redshift kernel is defined with scipy.stats, can be either analytical or empirical
        
        '''
        stellar_map = load_gaia_stellar_density()
        
        realization = poisson_star_map_from_fraction(stellar_map, int(N_gal), frac=frac, mask=mask)
        
        # Visualize
        if plot:
            hp.mollview(realization, title="Contaminated Poisson Realization", max=2)
        
        ra_star, dec_star = generate_poisson_radec_from_map(realization)

        # distribution of stellar redshifts? depends on the case, draw randomly from kernel

        if redshift_contam_kernel is None and zmin is None and zmax is None:
            print('need to either specify redshift contamination kernel or zmin/zmax, returning RA/DEC and setting redshift_star to None..')
            return ra_star, dec_star, None

        elif redshift_contam_kernel is not None:

            redshift_star = redshift_contam_kernel.rvs(size=len(ra_star))

        elif zmin is not None and zmax is not None:

            unif_dist = uniform(loc=zmin, scale=zmax-zmin)
            redshift_star = unif_dist.rvs(size=len(ra_star))
    
        return ra_star, dec_star, redshift_star
        
        
    def generate_dust_selection_err(self, mode='dust', debv_std=0.005, use_cl_sfd=True, alpha=-10):
        ''' 
        For mimicking error in selection function. 
        Returns dN/N in healpix format which can be used with modify_fkp_weights to modify selection function

        Modes: 'dust' (E(B-V) errors)
        alpha : scales dE(B-V) to dN/N (galaxies typically negatively correlated). ELGs very sensitive (-10), LRGs/QSOs less sensitive (~ -1 to -1.5)
        
        '''

        if use_cl_sfd:
            self.ebv_map, self.cl_sfd = gen_sfd_hp()
            self.delta_ebv = gen_delta_ebv_map_uncorr(self.cl_sfd, std=debv_std[d])

        elif use_sfd_rongpu_resid:
            self.delta_ebv = None # todo
            
        else:
            self.delta_ebv = np.random.normal(loc=0.0, scale=debv_std, size=self.npix)
            
        self.delta_n_over_n_uncorr = gen_dn_n_map(grf_map=self.delta_ebv, alpha=alpha)



def position_dependent_zerr(ztrue, ra, dec, sigz_nominal=0.01):
    '''Toy redshift errors that increase near RA=180'''
    sigz = sigz_nominal * (1 + 0.5 * np.cos(np.radians(ra)))
    return np.random.normal(0, sigz)


@staticmethod
def bimodal_zcontam(n, bim_frac=0.7, z0=0.3, z1=1.0, sz0=0.05, sz1=0.1):
    '''Toy redshift contamination: two Gaussian components'''
    n1 = int(bim_frac * n)
    n2 = n - n1
    z0draw = np.random.normal(loc=z0, scale=sz0, size=n1)
    z1draw = np.random.normal(loc=z1, scale=sz1, size=n2)
    return np.concatenate([z0draw, z1draw])

def redshift_err(sig_level):

    zerr = np.random.normal(0, sig_level)

    return zerr


def load_gaia_stellar_density(fpath=None, plot=True, vmax=500):

    if fpath is None:
        fpath = 'stars/stellar_density_map_12_lt_g_lt_17.npy'
    
    stellar_map = np.load(fpath)
    if plot:
        hp.mollview(stellar_map, title='Gaia stellar density, $12 < G < 17$', max=vmax)
        plt.show()

    return stellar_map


def poisson_star_map_from_fraction(stellar_map, N_gal, frac=0.05, mask=None):

    map_clean = np.copy(stellar_map)
    map_clean = np.nan_to_num(map_clean, nan=0.0, posinf=0.0, neginf=0.0)
    map_clean[map_clean < 0] = 0.0

    if mask is not None:
        map_clean[~mask] = 0.0

    # Normalize to make it a probability distribution
    total_weight = np.sum(map_clean)
    if total_weight <= 0:
        raise ValueError("Stellar map has zero or negative total weight after masking.")
    prob_map = map_clean / total_weight

    # Target total number of stars
    N_star = int(frac * N_gal)

    # Expected stars per pixel
    expected_stars = prob_map * N_star

    # Poisson draw
    star_counts = np.random.poisson(expected_stars)

    return star_counts


def generate_poisson_radec_from_map(poisson_mean_map, seed=None):
    """
    Given a HEALPix map of Poisson means (expected counts per pixel),
    return RA and Dec positions of a Poisson sampling.

    Parameters
    ----------
    poisson_mean_map : ndarray
        HEALPix map (1D array of shape [Npix]) with mean number of objects per pixel.
    seed : int or None
        Optional seed for reproducibility.

    Returns
    -------
    ra_deg : ndarray
        Right ascension values (degrees).
    dec_deg : ndarray
        Declination values (degrees).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    nside = hp.npix2nside(len(poisson_mean_map))
    mean_map = np.nan_to_num(poisson_mean_map, nan=0.0, posinf=0.0, neginf=0.0)
    mean_map[mean_map < 0] = 0.0

    # Draw Poisson number of samples in each pixel
    counts = rng.poisson(mean_map)

    # Repeat pixel indices according to counts
    ipix = np.repeat(np.arange(len(counts)), counts)
    if len(ipix) == 0:
        return np.array([]), np.array([])

    # Sample uniform positions within pixels
    theta, phi = hp.pix2ang(nside, ipix, nest=False)
    dtheta = rng.uniform(0, hp.nside2resol(nside), size=len(ipix))  # very small offsets
    dphi = rng.uniform(0, 2 * np.pi / hp.nside2npix(nside), size=len(ipix))
    theta = np.clip(theta + dtheta, 0, np.pi)
    phi = (phi + dphi) % (2 * np.pi)

    # Convert to RA/Dec
    ra_deg = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    return ra_deg, dec_deg


def gen_delta_ebv_map_uncorr(cl_sfd, nside=256, std=0.01):
    
    # Simulate GRF with same power spectrum
    grf_map = hp.synfast(cl_sfd, nside=nside, lmax=len(cl_sfd)-1, new=True)
    
    # Normalize to have std ~ 0.01
    grf_map -= np.mean(grf_map)
    grf_map *= std / np.std(grf_map)  # sets std to 0.01 (10 mmag)

    return grf_map


def gen_dn_n_map(cl_sfd=None, grf_map=None, alpha=-10, std=0.01):

    if grf_map is None:
        grf_map = gen_delta_ebv_map_uncorr(cl_sfd, std=std)

    dn_n = alpha * grf_map

    return dn_n


def modify_fkp_weights(ra, dec, w_fkp, delta_n_over_n_map, nside=256):
    """
    Modulate FKP weights using a contamination map (delta_n_over_n).

    Parameters:
    -----------
    ra, dec : array_like
        RA/DEC positions of galaxies in degrees.
    w_fkp : array_like
        Original FKP weights.
    delta_n_over_n_map : array_like
        HEALPix map of contamination δn/n.
    nside : int
        HEALPix resolution of the map.

    Returns:
    --------
    w_total : array_like
        Modified total weights including contamination.
    """

    # Convert RA/DEC to theta, phi
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    # Get pixel indices for each position
    pix = hp.ang2pix(nside, theta, phi)

    # Extract contamination δn/n values
    dn_over_n = delta_n_over_n_map[pix]

    # Systematic weight: 1 + δn/n
    w_sys = 1.0 + dn_over_n
    w_sys = np.clip(w_sys, 0.01, 10.0)  # optional: avoid extreme weights

    # Total weight: FKP × systematics
    w_total = w_fkp * w_sys
    return w_total, w_sys


    