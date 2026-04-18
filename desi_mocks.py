import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

from astropy.cosmology import Planck18 as cosmo

from contamination import *
from mask_and_randoms import *
from stitch_box import *
from utils import *
from halfdome import *


class desi_mock():

    desi_mock_basedir = '/global/cfs/cdirs/desi/survey/catalogs/'
    quijote_mock_basedir = '/pscratch/sd/r/rmfeder/quijote_dat/'
    halfdome_mock_basedir = '/global/cfs/cdirs/cmb/gsharing/halfdome/full_res/halos/'

    # real DESI data
    dr1_basepath = '/global/cfs/cdirs/desi/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/'

    def __init__(self, year=3, mock_type='AbacusSummit_v4_1', verbose=False):

        
        self.mock_dir = self.desi_mock_basedir+'Y'+str(year)+'/mocks/SecondGenMocks/'+mock_type+'/'

        if verbose:
            print('Mock directory is ', self.mock_dir)

        self.chi_interp = grab_chi_interp()


    def apply_redshift_selection(self, ra, dec, redshift, weight, zmin=None, zmax=None):

        print('Applying selection to input catalog..')

        mask = np.ones_like(redshift)

        if zmin is not None:
            mask *= (redshift > zmin)

        if zmax is not None:
            mask *= (redshift < zmax)
        
        which = np.where(mask)[0]

        sel_ra, sel_dec, sel_redshift, sel_weight = [x[which] for x in [ra, dec, redshift, weight]]
    
        print('length after selection is '+str(len(sel_redshift)))

        return sel_ra, sel_dec, sel_redshift, sel_weight


    def load_ezmock(self, mock_idx, galtype='ELG', zmin=None, zmax=None, apply_redshift_sel=False, downsamp_fac=None, gen_fkp=False, sel_fp=True):

        mock_fpath = self.mock_dir+'/'+galtype+'/EZmock_'+galtype+'_complete_AbacusSummit_base_c000_ph000_NScomb_'+str(mock_idx).zfill(4)+'.fits.gz'
        print('loading from mock fpath', mock_fpath)

        mockdat = fits.open(mock_fpath)

        cat_ra, cat_dec, cat_redshift, ran_num_0_1, cat_nbar_z, status = [mockdat[1].data[key] for key in ['RA', 'DEC', 'Z', 'RAN_NUM_0_1', 'NZ', 'STATUS']]

        if sel_fp:

            mask = ((status & (1 << 1)) != 0) & ((status & (1 << 1)) != 0)

        elif downsamp_fac is not None:
            p_downsamp = 1./downsamp_fac
            mask = (ran_num_0_1 < p_downsamp)

        cat_ra = cat_ra[mask]
        cat_dec = cat_dec[mask]
        cat_redshift = cat_redshift[mask]
        cat_nbar_z = cat_nbar_z[mask]

        print('after initial downselect, catalog has length ', len(cat_ra))
            
        if apply_redshift_sel:
            sel_ra, sel_dec, sel_redshift, sel_cat_nbar_z = self.apply_redshift_selection(cat_ra, cat_dec, cat_redshift, cat_nbar_z, zmin=zmin, zmax=zmax)
            
            return sel_ra, sel_dec, sel_redshift, sel_cat_nbar_z


        return cat_ra, cat_dec, cat_redshift, cat_nbar_z

    def load_desi_dat(self, skystr = 'NGC', galtype = 'QSO', mode='data', rand_idx=1, apply_redshift_sel=False, \
                     plot=False, inplace=False, zmin=None, zmax=None, combine_w_fkp=False):

        if mode=='data':
            filename = galtype+'_'+skystr+'_clustering.dat.fits'
        elif mode=='random':
            filename = galtype+'_'+skystr+'_'+str(rand_idx)+'_clustering.ran.fits'

        dat = fits.open(self.dr1_basepath+filename)
        
        cat_ra, cat_dec, cat_redshift = [dat['LSS'].data[key] for key in ['RA', 'DEC', 'Z']]

        cat_weight, cat_weight_fkp = [dat['LSS'].data[key] for key in ['WEIGHT', 'WEIGHT_FKP']]

        if combine_w_fkp:
            cat_weight *= cat_weight_fkp
                                      
        # cat_ra, cat_dec, cat_redshift, cat_weight = [dat['LSS'].data[key] for key in ['RA', 'DEC', 'Z', 'WEIGHT_FKP']]

        if apply_redshift_sel:

            if zmin is None and zmax is None:

                if galtype=='QSO':
                    zmin, zmax = 0.8, 3.1
                elif galtype=='LRG':
                    zmin, zmax = 0.4, 1.1
                elif 'ELG' in galtype:
                    zmin, zmax = 0.6, 1.6
            
            sel_ra, sel_dec, sel_redshift, sel_weight = self.apply_redshift_selection(cat_ra, cat_dec, cat_redshift, cat_weight, \
                                 zmin=zmin, zmax=zmax)
            
        if plot:
            plt.figure()
            _, bins, _ = plt.hist(cat_redshift, bins=30, histtype='step', color='k', label='Full sample')
            if apply_redshift_sel:
                plt.hist(sel_redshift, bins=bins, histtype='step', color='r', label='After selection')
            plt.legend()
            plt.xlabel('Redshift')
            plt.yscale('log')
            plt.show()

        if apply_redshift_sel:
            return sel_ra, sel_dec, sel_redshift, sel_weight
        else:
            return cat_ra, cat_dec, cat_redshift, cat_weight

        
    def load_desi_mock(self, mock_idx, galtype='QSO_complete_SGC', zmin=None, zmax=None, mode='data', rand_idx=0, apply_redshift_sel=False, \
                      plot=False, inplace=False):

        
        
        if mode=='data':
            mock_fpath = self.mock_dir+'mock'+str(mock_idx)+'/'+galtype+'_clustering.dat.fits'

        elif mode=='random':
            mock_fpath = self.mock_dir+'mock'+str(mock_idx)+'/'+galtype+'_'+str(rand_idx)+'_clustering.ran.fits'

        print('Opening from ', mock_fpath)

        mockdat = fits.open(mock_fpath)

        cat_ra, cat_dec, cat_redshift, cat_weight = [mockdat['LSS'].data[key] for key in ['RA', 'DEC', 'Z', 'WEIGHT_FKP']]

        if apply_redshift_sel:
            sel_ra, sel_dec, sel_redshift, sel_weight = self.apply_redshift_selection(cat_ra, cat_dec, cat_redshift, cat_weight, \
                                 zmin=zmin, zmax=zmax)
            
        if plot:
            plt.figure()
            _, bins, _ = plt.hist(cat_redshift, bins=30, histtype='step', color='k', label='Full sample')
            if apply_sel:
                plt.hist(sel_redshift, bins=bins, histtype='step', color='r', label='After selection')
            plt.legend()
            plt.xlabel('Redshift')
            plt.yscale('log')
            plt.show()

        if apply_sel:
            return sel_ra, sel_dec, sel_redshift, sel_weight
        else:
            return cat_ra, cat_dec, cat_redshift, cat_weight


    def load_quijote_galpos(self, mock_idx, with_RSD=False, replicate=False, rep_fac=3, ds_fac=1, randomize=False, seed=42):

        
        fpath = self.quijote_mock_basedir+str(mock_idx)+'/numpy/gal_Position.npy'
        if with_RSD:
            fpath = fpath.replace('Position', 'RSDPosition')
        
        print('fpath is ', fpath)
        galpos = np.load(fpath)

        boxdims = [np.max(galpos[:,i]) for i in range(3)]

        print('boxdims:', boxdims)

        boxlength = np.mean(boxdims)

        h = cosmo.h

        if replicate:

            if ds_fac != 1:

                # Randomly choose indices
                N_sub = galpos.shape[0]//ds_fac
                rng = np.random.default_rng(seed)
                idx = rng.choice(galpos.shape[0], size=N_sub, replace=False)
                galpos = galpos[idx]
            
            print('galpos initially has shape', galpos.shape)
            galpos = stitch_boxes_randomized(galpos, np.mean(boxlength), rep_fac=rep_fac, randomize=randomize)
            print('galpos now has shape', galpos.shape)

        return galpos


    def load_halfdome_mock(self, mockidx, n_sample=None):

        mockidx_use = 100+2*mockidx
        lightcone_fpath = self.halfdome_mock_basedir+'lightcone_'+str(mockidx_use)+'.hdf5'

        positions, redshifts = load_lightcone_subset(lightcone_fpath, n_sample=n_sample)

        return positions, redshifts

        
    def convert_to_posarray(self, ra, dec, redshift):

        cat_r = cosmo.comoving_distance(redshift).value  # in Mpc/h
        pos_data = np.array([ra, dec, cat_r])

        return pos_data

    def extend_selcat(ra_ext, dec_ext, z_ext):

        self.sel_ra.extend(ra_ext)
        self.sel_dec.extend(dec_ext)
        self.sel_redshift.extend(z_ext)


    def contaminate_catalog(self, pos_clean, pos_contam=None, ra_contam=None, dec_contam=None, redshift_contam=None):

        ''' 
        This would be for additive contamination (e.g., stars), though could also consider error in star/galaxy separation and its impact on selection function error.

        pos_fn : Generic position function generator function, could be customized

        contam_frac : relative to fiducial selected sample
        
        '''
        
        if pos_contam is None:
            print('Converting RA/DEC/redshift to RA/DEC/r_comoving')
            pos_contam = self.convert_to_posarray(ra_contam, dec_contam, redshift_contam)

        pos_comb = np.vstack([pos_clean, pos_contam])
                
        return pos_comb



def test_contamination_on_mocks(mock, test_params=None, compute_clean=True, **kwargs):

    ''' Parent function in progress '''

    if test_params is None:
        test_params = init_test_params(kwargs)
    else:
        print('Using fixed test parameters specified at input')

    print(test_params)

    # Finer bins at low k, coarser at high k. Specific for large scale clustering but could configure differetly
    kedges = np.unique(
        np.concatenate([
            np.linspace(0.001, 0.01, 4, endpoint=False),  # Fine bins below 0.01
            np.linspace(0.01, 0.1, 40)                   # Standard bins above
        ])
    )
    kbp = 0.5*(kedges[1:]+kedges[:-1])

    ez_ra, ez_dec, ez_Z, ez_nbar_z = dm.load_ezmock(mockidx+1, galtype, zmin=zmin, zmax=zmax, apply_sel=True, downsamp_fac=None, sel_fp=True)

    
    if compute_clean:

        result_clean = compute_plk(gal_posarray, gal_wfkp/np.mean(gal_wfkp), rand_pos, rand_fkp/np.mean(rand_fkp), kedges, plot_wedges=False, shotnoise=None, nmesh=nmesh)

    else:
        result_clean = None
    # contamination modifies the galaxy weights while keeping randoms as assumed
    result_wsys = compute_plk(gal_posarray, gal_wfkp_wsys/np.mean(gal_wfkp_wsys), rand_pos, rand_fkp/np.mean(rand_fkp), kedges, plot_wedges=False, shotnoise=None, nmesh=nmesh)

    return result_wsys, result_clean


def convert_to_ra_dec_distance(galpos_mpc_per_h, L_box_mpc_per_h):
    center_offset_mpc = L_box_mpc_per_h / 2.0
    x_relative = x_orig_mpc - center_offset_mpc
    y_relative = y_orig_mpc - center_offset_mpc
    z_relative = z_orig_mpc - center_offset_mpc

    r = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2) * u.Mpc
    
    ra_rad = np.arctan2(y_relative, x_relative)
    ra_deg = np.degrees(ra_rad)
    # Ensure RA is in [0, 360)
    ra_deg[ra_deg < 0] += 360

    dec_rad = np.arcsin(z_relative / r.value) # .value to remove units for arcsin
    dec_deg = np.degrees(dec_rad)

    return ra_deg, dec_deg, r

def apply_healpix_mask(galpos_mpc_per_h: np.ndarray, L_box_mpc_per_h: float, h_value: float, mask_path: str, plot_masked_skymap: bool = False) -> np.ndarray:
    """
    Converts galaxy Cartesian coordinates to RA/DEC, applies a HEALPix survey mask,
    and returns the original x/y/z coordinates for galaxies within the mask.

    Parameters
    ----------
    galpos_mpc_per_h : np.ndarray
        A 2D NumPy array (N_galaxies, 3) of galaxy positions (x, y, z)
        in comoving Mpc/h units. Assumes all coordinates are positive,
        and the observer is at the center of the L_box.
    L_box_mpc_per_h : float
        The side length of the cubic simulation box in comoving Mpc/h.
    h_value : float
        The dimensionless Hubble parameter, h (where H0 = 100 * h km/s/Mpc).
    mask_path : str
        The file path to the HEALPix FITS mask file. This mask should
        contain pixel values where non-zero typically indicates a
        valid survey region.
    plot_masked_skymap : bool, optional
        If True, a scatter plot of RA/DEC for the *masked* galaxies will be displayed.
        Defaults to False.

    Returns
    -------
    np.ndarray
        A 2D NumPy array (N_selected_galaxies, 3) containing the original
        x, y, z coordinates (in Mpc/h) of the galaxies that fall within the
        specified HEALPix mask.
    """
    print("--- Starting HEALPix Mask Application ---")
    print(f"Input box size: {L_box_mpc_per_h:.2f} Mpc/h")
    print(f"Using h = {h_value:.3f}")

    # Convert coordinates and box size from Mpc/h to Mpc
    x_orig_mpc = galpos_mpc_per_h[:, 0] * h_value
    y_orig_mpc = galpos_mpc_per_h[:, 1] * h_value
    z_orig_mpc = galpos_mpc_per_h[:, 2] * h_value
    L_box_mpc = L_box_mpc_per_h * h_value

    print(f"Converted box size: {L_box_mpc:.2f} Mpc")

    # Translate coordinates to be relative to the box center (observer at 0,0,0)
    center_offset_mpc = L_box_mpc / 2.0
    x_relative = x_orig_mpc - center_offset_mpc
    y_relative = y_orig_mpc - center_offset_mpc
    z_relative = z_orig_mpc - center_offset_mpc

    print(f"Number of input galaxies: {len(galpos_mpc_per_h)}")

    # Calculate spherical coordinates (RA/DEC)
    r = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2) * u.Mpc

    ra_rad = np.arctan2(y_relative, x_relative)
    ra_deg = np.degrees(ra_rad)
    # Ensure RA is in [0, 360)
    ra_deg[ra_deg < 0] += 360

    dec_rad = np.arcsin(z_relative / r.value) # .value to remove units for arcsin
    dec_deg = np.degrees(dec_rad)

    # --- Load and apply the HEALPix mask ---
    print(f"Loading HEALPix mask from: {mask_path}")
    try:
        # hp.read_map by default returns a map in RING ordering.
        # If your mask is in NESTED, you might want to specify nest=True here
        # or convert it later. For simple masks, the ordering often doesn't matter
        # as long as it's consistent in hp.ang2pix and the mask values.
        survey_mask = hp.read_map(mask_path, field=0, verbose=False)
        print(f"Mask NSIDE: {hp.npix2nside(len(survey_mask))}")
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}")
        return np.array([])
    except Exception as e:
        print(f"Error loading mask: {e}")
        return np.array([])

    # Get the NSIDE of the mask
    nside_mask = hp.npix2nside(len(survey_mask))

    # Convert galaxy RA/DEC to HEALPix pixel indices at the mask's resolution
    galaxy_pixels = hp.ang2pix(nside_mask, ra_deg, dec_deg, lonlat=True, nest=False) # Assuming mask is RING

    # Identify galaxies that fall within "valid" (non-zero) mask pixels
    # A common convention for masks is 0 for masked, 1 for unmasked.
    # If your mask uses hp.UNSEEN for masked values, that's also handled.
    valid_galaxies_indices = np.where(survey_mask[galaxy_pixels] != 0)[0]
    # You might want to be more explicit, e.g., survey_mask[galaxy_pixels] > 0
    # depending on how your mask values are defined.

    # Select the original galaxy positions that are within the mask
    galpos_masked = galpos_mpc_per_h[valid_galaxies_indices]

    print(f"Number of galaxies after mask application: {len(galpos_masked)}")

    # --- Plotting (optional) ---
    if plot_masked_skymap:
        plt.figure(figsize=(10, 6))
        # Plot only the RA/DEC of the galaxies that passed the mask
        plt.scatter(ra_deg, dec_deg, s=1, color='k', alpha=0.1)
        plt.xlabel("Right Ascension (degrees)")
        plt.ylabel("Declination (degrees)")
        plt.title(f"Galaxy Positions Before HEALPix Mask (N={len(galpos_masked)})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        # Plot only the RA/DEC of the galaxies that passed the mask
        plt.scatter(ra_deg[valid_galaxies_indices], dec_deg[valid_galaxies_indices], s=1, color='k', alpha=0.1)
        plt.xlabel("Right Ascension (degrees)")
        plt.ylabel("Declination (degrees)")
        plt.title(f"Galaxy Positions After HEALPix Mask (N={len(galpos_masked)})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    print("--- HEALPix Mask Application Complete ---")
    return galpos_masked
    
    


        