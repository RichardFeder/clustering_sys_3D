import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import Galactic, ICRS
from astropy import units as u
from astropy.coordinates import FK5, FK4 # Import these for potential transformations if needed, but ICRS is usually the target.


def load_gaia_stellar_density(fpath=None, plot=True, vmax=500):

    if fpath is None:
        fpath = 'stars/stellar_density_map_12_lt_g_lt_17.npy'
    
    stellar_map = np.load(fpath)
    if plot:
        hp.mollview(stellar_map, title='Gaia stellar density, $12 < G < 17$', max=vmax)
        plt.show()

    return stellar_map


def simple_halo_thick_disk_stellar_density(N_gal, frac=0.05, nside=64,
                                           halo_power_law_exp=1.0, # Controls how fast density drops away from GC projection
                                           thick_disk_amplitude=0.5, # Relative strength of thick disk
                                           thick_disk_b_scale=15.0, # Scale height in degrees for thick disk
                                           seed=None):
    """
    Generates a simplified stellar density map focusing on the stellar halo and
    thick disk components, suitable for high Galactic latitude surveys.
    The model is based on Galactic coordinates (l, b).

    The density is modeled as a sum of:
    1. A 'halo-like' component: Higher density near the Galactic Center's projection
       and generally decreasing with angular distance.
    2. A 'thick-disk-like' component: Peaking around the Galactic equator (b=0)
       and decaying exponentially with increasing |b|, but with a larger scale
       than the thin disk.

    Parameters
    ----------
    N_gal : int
        The total number of galaxies in your galaxy sample.
    frac : float, optional
        The fraction of stars relative to the galaxy sample (N_star = frac * N_gal).
        Defaults to 0.05.
    nside : int, optional
        HEALPix nside parameter for the output map. Higher nside means finer resolution.
        Defaults to 64.
    halo_power_law_exp : float, optional
        Exponent for the angular distance from the Galactic Center (l=0, b=0) for
        the halo-like component. A higher value means density drops faster away
        from the GC projection. Defaults to 1.0.
    thick_disk_amplitude : float, optional
        Relative amplitude of the thick disk component compared to the halo.
        Defaults to 0.5.
    thick_disk_b_scale : float, optional
        Scale in degrees for the exponential decay of the thick disk component
        with increasing absolute Galactic latitude (|b|). Defaults to 15.0 degrees.
        This roughly mimics the vertical extent.
    seed : int or None, optional
        Optional seed for reproducibility. Defaults to None.

    Returns
    -------
    ra_deg : ndarray
        Right ascension values (degrees) for the generated stars.
    dec_deg : ndarray
        Declination values (degrees) for the generated stars.
    notional_stellar_map : ndarray
        The HEALPix map of the notional stellar density before Poisson sampling (in Galactic coords).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)

    # Get Galactic coordinates (l, b) for each HEALPix pixel center
    # healpy's default coordinate system for pix2ang is usually spherical (theta, phi)
    # where theta=0 is north pole and phi is azimuth.
    # To interpret this as Galactic (l, b), we need to set the coord system when calling mollview
    # and perform the coordinate transform when generating RA/Dec from l,b.
    # But for the map creation, let's treat theta, phi as co-latitude and longitude in Galactic frame.
    theta_gal, phi_gal = hp.pix2ang(nside, ipix, nest=False)
    l_deg = np.degrees(phi_gal)
    b_deg = 90.0 - np.degrees(theta_gal)

    # --- 1. Halo-like Component (simpler power law decay from Galactic Center projection) ---
    # We want density to decrease as we move away from the Galactic Center's projection
    # (l=0, b=0). We can use an angular distance.
    # This is a highly simplified proxy for a 3D halo projected onto 2D.
    # Angular distance from (l=0, b=0)
    # Convert to radians for spherical distance calculation if needed, but for a simplified model
    # we can use a proxy like 1 / (1 + angle_from_GC^exp) or similar.
    # Let's use a "distance" based on |l| and |b| for simplicity, adjusted for wrap-around.
    # A simple inverse power law on angular distance is common for halos.
    # Here, 'angular_distance_proxy' is just an approximation.
    l_rad = np.radians(l_deg)
    b_rad = np.radians(b_deg)

    # Calculate angular distance from (l=0, b=0) using Euclidean distance in l,b space for simplicity
    # This isn't strictly spherical distance but works for a notional model
    # And we need to handle l wrapping from 0 to 360 or -180 to 180.
    # For l, consider minimum distance (e.g., dist from 350 to 10 is 20, not 340)
    l_dist_from_0 = np.minimum(l_deg, 360 - l_deg) # Distance to 0 or 360
    # Add a small epsilon to avoid division by zero at the GC projection itself
    angular_distance_proxy = np.sqrt(l_dist_from_0**2 + b_deg**2) + 1e-6 # Add epsilon

    # The halo component is highest at the GC projection and decreases outwards
    halo_component = 1.0 / (angular_distance_proxy**halo_power_law_exp)
    halo_component = halo_component / np.max(halo_component) # Normalize its max to 1

    # --- 2. Thick Disk-like Component ---
    # Exponential decay from the Galactic plane (|b|=0) with a larger scale height
    # This component will be strong at low |b| and drop off at high |b|.
    # Add a small offset to avoid issues at b=0 or very small b
    thick_disk_component = thick_disk_amplitude * np.exp(-np.abs(b_deg) / thick_disk_b_scale)

    # --- Combine Components ---
    # The halo contribution should generally be present across the sky.
    # The thick disk is confined to lower |b|.
    notional_density = halo_component + thick_disk_component
    notional_density[notional_density < 0] = 0.0

    # Normalize the notional density to make it a probability distribution
    total_notional_density = np.sum(notional_density)
    if total_notional_density <= 0:
        raise ValueError("Notional stellar density has zero or negative total weight.")
    prob_map = notional_density / total_notional_density

    # Target total number of stars
    N_star = int(frac * N_gal)

    # Expected stars per pixel
    expected_stars = prob_map * N_star

    # Poisson draw
    star_counts = rng.poisson(expected_stars)

    # Generate RA/Dec positions from Poisson counts
    mean_map = np.nan_to_num(star_counts, nan=0.0, posinf=0.0, neginf=0.0)
    mean_map[mean_map < 0] = 0.0

    ipix_sampled = np.repeat(np.arange(len(mean_map)), mean_map.astype(int))
    if len(ipix_sampled) == 0:
        return np.array([]), np.array([]), notional_density

    # Get Galactic l,b for the sampled pixels (re-use initial l_deg, b_deg based on ipix_sampled)
    # This is slightly more precise as it maps back to the original pixel centers that were sampled
    l_sampled = l_deg[ipix_sampled] * u.deg
    b_sampled = b_deg[ipix_sampled] * u.deg

    # Convert Galactic (l, b) to ICRS (RA, Dec) using astropy
    galactic_coords = Galactic(l=l_sampled, b=b_sampled)
    icrs_coords = galactic_coords.transform_to(ICRS())

    ra_deg = icrs_coords.ra.deg
    dec_deg = icrs_coords.dec.deg

    # Add small random offsets within the pixel for smoother distribution
    # This ensures stars aren't only at pixel centers.
    # These offsets are applied directly to the final RA/Dec.
    pixel_resolution_deg = np.degrees(hp.nside2resol(nside))
    ra_offset = rng.uniform(-0.5 * pixel_resolution_deg, 0.5 * pixel_resolution_deg, size=len(ra_deg))
    dec_offset = rng.uniform(-0.5 * pixel_resolution_deg, 0.5 * pixel_resolution_deg, size=len(dec_deg))

    ra_deg = (ra_deg + ra_offset) % 360
    dec_deg = np.clip(dec_deg + dec_offset, -90, 90)

    return ra_deg, dec_deg, notional_density

def notional_radec_stellar_density(N_gal, frac=0.05, nside=64,
                                   dec_peak_density=0.0, dec_decay_scale=20.0,
                                   bulge_ra_deg=266.4, bulge_dec_deg=-29.0,
                                   bulge_scale_ra=10.0, bulge_scale_dec=5.0,
                                   bulge_amplitude=5.0, seed=None):
    """
    Generates a notional stellar density map based on Declination and a "bulge" component,
    then draws Poisson realizations of stellar positions.

    The notional stellar density is modeled with:
    1. A basic decline from a peak Declination (e.g., celestial equator).
    2. An elliptical 2D Gaussian "bulge" component centered near the Galactic Center's
       RA/Dec projection, to simulate increased density in that region.

    Parameters
    ----------
    N_gal : int
        The total number of galaxies in your galaxy sample.
    frac : float, optional
        The fraction of stars relative to the galaxy sample (N_star = frac * N_gal).
        Defaults to 0.05.
    nside : int, optional
        HEALPix nside parameter for the output map. Higher nside means finer resolution.
        Defaults to 64.
    dec_peak_density : float, optional
        The Declination (in degrees) where the background stellar density is highest.
        Defaults to 0.0 (celestial equator).
    dec_decay_scale : float, optional
        The e-folding scale (in degrees) for the exponential decay of stellar density
        away from `dec_peak_density`. A smaller value means a faster decay.
        Defaults to 20.0.
    bulge_ra_deg : float, optional
        Right Ascension (in degrees) of the center of the "bulge" component.
        Defaults to 266.4 degrees (approximate RA of Galactic Center).
    bulge_dec_deg : float, optional
        Declination (in degrees) of the center of the "bulge" component.
        Defaults to -29.0 degrees (approximate Dec of Galactic Center).
    bulge_scale_ra : float, optional
        The RA scale (standard deviation, in degrees) of the elliptical Gaussian bulge.
        Defaults to 10.0.
    bulge_scale_dec : float, optional
        The Dec scale (standard deviation, in degrees) of the elliptical Gaussian bulge.
        Defaults to 5.0.
    bulge_amplitude : float, optional
        Multiplicative factor for the bulge component. A higher value means a more
        pronounced bulge. Defaults to 5.0.
    seed : int or None, optional
        Optional seed for reproducibility. Defaults to None.

    Returns
    -------
    ra_deg : ndarray
        Right ascension values (degrees) for the generated stars.
    dec_deg : ndarray
        Declination values (degrees) for the generated stars.
    notional_stellar_map : ndarray
        The HEALPix map of the notional stellar density before Poisson sampling.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)

    # Get RA and Dec for each HEALPix pixel center
    theta, phi = hp.pix2ang(nside, ipix, nest=False)
    ra_pixel = np.degrees(phi)
    dec_pixel = 90.0 - np.degrees(theta)

    # --- 1. Declination Dependence ---
    # Exponential decay from dec_peak_density
    # Density is highest at dec_peak_density and drops off exponentially
    dec_density_component = np.exp(-np.abs(dec_pixel - dec_peak_density) / dec_decay_scale)

    # --- 2. Bulge Component (Elliptical 2D Gaussian) ---
    # Calculate distance from bulge center in RA and Dec
    delta_ra = (ra_pixel - bulge_ra_deg + 180) % 360 - 180 # Handle wrap-around at 0/360
    delta_dec = dec_pixel - bulge_dec_deg

    # Gaussian profile for the bulge
    bulge_component = bulge_amplitude * np.exp(
        -0.5 * (
            (delta_ra / bulge_scale_ra)**2 +
            (delta_dec / bulge_scale_dec)**2
        )
    )

    # --- Combine Components ---
    # Ensure all components are non-negative
    notional_density = dec_density_component + bulge_component
    notional_density[notional_density < 0] = 0.0 # Should not happen with current model, but good practice

    # Normalize the notional density to represent a probability distribution
    total_notional_density = np.sum(notional_density)
    if total_notional_density <= 0:
        raise ValueError("Notional stellar density has zero or negative total weight after masking.")
    prob_map = notional_density / total_notional_density

    # Target total number of stars
    N_star = int(frac * N_gal)

    # Expected stars per pixel
    expected_stars = prob_map * N_star

    # Poisson draw
    star_counts = rng.poisson(expected_stars)

    # Generate RA/Dec positions from Poisson counts
    mean_map = np.nan_to_num(star_counts, nan=0.0, posinf=0.0, neginf=0.0)
    mean_map[mean_map < 0] = 0.0

    # Repeat pixel indices according to counts
    ipix_sampled = np.repeat(np.arange(len(mean_map)), mean_map.astype(int))
    if len(ipix_sampled) == 0:
        return np.array([]), np.array([]), notional_density

    # Get RA/Dec for the sampled pixels
    theta_sampled_pix, phi_sampled_pix = hp.pix2ang(nside, ipix_sampled, nest=False)
    ra_sampled_pix = np.degrees(phi_sampled_pix)
    dec_sampled_pix = 90.0 - np.degrees(theta_sampled_pix)

    # Add small random offsets within the pixel for smoother distribution
    # These offsets are applied directly to RA/Dec
    # The resolution in RA/Dec varies with Dec, but for a notional map,
    # a fixed angular resolution approximation is often fine.
    # hp.nside2resol gives resolution in radians.
    pixel_resolution_deg = np.degrees(hp.nside2resol(nside))

    # Add uniform random offsets within approx 1/2 pixel size
    ra_offset = rng.uniform(-0.5 * pixel_resolution_deg, 0.5 * pixel_resolution_deg, size=len(ipix_sampled))
    dec_offset = rng.uniform(-0.5 * pixel_resolution_deg, 0.5 * pixel_resolution_deg, size=len(ipix_sampled))

    ra_deg = (ra_sampled_pix + ra_offset) % 360
    dec_deg = np.clip(dec_sampled_pix + dec_offset, -90, 90) # Ensure Dec stays within bounds

    return ra_deg, dec_deg, notional_density