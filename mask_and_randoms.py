import numpy as np
import healpy as hp
from astropy.cosmology import Planck18 as cosmo

def generate_uniform_randoms(chi_interp, n_randoms, zmin=0.4, zmax=1.0, data_z=None, seed=None):
    rng = np.random.default_rng(seed)

    # Uniform RA
    ra = rng.uniform(0, 360, n_randoms)

    # Uniform sin(Dec)
    sin_dec = rng.uniform(-1, 1, n_randoms)
    dec = np.degrees(np.arcsin(sin_dec))
    
    # Uniform z or weighted z

    if data_z is None:
        z = rng.uniform(zmin, zmax, n_randoms)  # or use a dN/dz sampling
    else:
        rand_indices = rng.choice(len(data_z), size=n_randoms, replace=True)
        z = data_z[rand_indices]
    
    r_gal_mpc = chi_interp(z)
    print('min/max rgal:', np.min(r_gal_mpc), np.max(r_gal_mpc))

    r = r_gal_mpc * cosmo.h  # if your mock is in Mpc/h

    return ra, dec, r, z

def gen_hp_mask_from_data(data_ra, data_dec, min_counts=10, nside=64):
    theta = np.radians(90. - data_dec)
    phi = np.radians(data_ra)
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    # Create a mask of valid pixels
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix, minlength=npix)
    valid_pix = np.where(counts >= min_counts)[0]
    mask = np.zeros(npix, dtype=bool)
    mask[valid_pix] = True

    return mask

def generate_random_catalog_from_data(data_ra, data_dec, data_z, data_nbar, n_randoms=1e6, P0=2600, nside=64, min_counts=10, seed=None):
    """
    Generate a random catalog matching the data's angular footprint and redshift distribution.

    Parameters
    ----------
    data_ra : array_like
        Right ascension of data catalog [deg].
    data_dec : array_like
        Declination of data catalog [deg].
    data_z : array_like
        Redshift of data catalog.
    data_nbar : array_like
        Corresponding nbar(z) values for the data catalog.
    n_randoms : int
        Number of random points to generate.
    P0 : float
        Power spectrum amplitude used in FKP weighting.
    nside : int
        HEALPix resolution to define the angular mask.
    min_counts : int
        Minimum number of data points in a pixel to be considered valid.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    rand_ra, rand_dec, rand_z, rand_nbar, rand_fkp : arrays
        Angular and radial coordinates and weights of the random catalog.
    """
    rng = np.random.default_rng(seed)

    # Convert data RA/Dec to HEALPix pixels
    mask = gen_hp_mask_from_data(data_ra, data_dec, nside=nside)

    # Generate uniform points on the sphere and keep those in the mask
    rand_ra, rand_dec = [], []
    while len(rand_ra) < n_randoms:
        ra_try = rng.uniform(0., 360., size=n_randoms)
        dec_try = np.degrees(np.arcsin(rng.uniform(-1., 1., size=n_randoms)))

        theta_try = np.radians(90. - dec_try)
        phi_try = np.radians(ra_try)
        pix_try = hp.ang2pix(nside, theta_try, phi_try, nest=False)

        keep = mask[pix_try]
        rand_ra.extend(ra_try[keep])
        rand_dec.extend(dec_try[keep])

        # Trim if oversampled
        if len(rand_ra) > n_randoms:
            rand_ra = rand_ra[:n_randoms]
            rand_dec = rand_dec[:n_randoms]

    rand_ra = np.array(rand_ra)
    rand_dec = np.array(rand_dec)

    # Sample redshifts and nbar(z) from the data catalog
    rand_indices = rng.choice(len(data_z), size=n_randoms, replace=True)
    rand_z = data_z[rand_indices]
    rand_nbar = data_nbar[rand_indices]

    # Compute FKP weights
    rand_fkp = 1.0 / (1.0 + rand_nbar * P0)

    return rand_ra, rand_dec, rand_z, rand_nbar, rand_fkp, mask
