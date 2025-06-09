import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

from astropy.cosmology import Planck18 as cosmo

from contamination import *


class desi_mock():

    mock_basedir = '/global/cfs/cdirs/desi/survey/catalogs/'
    
    def __init__(self, year=3, mock_type='AbacusSummit_v4_1'):

        self.mock_dir = self.mock_basedir+'Y'+str(year)+'/mocks/SecondGenMocks/'+mock_type+'/'
        
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



def generate_random_catalog_from_data(data_ra, data_dec, data_z, data_nbar, n_randoms=10**6, P0=2600, nside=64, min_counts=10, seed=None):
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
    theta = np.radians(90. - data_dec)
    phi = np.radians(data_ra)
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    # Create a mask of valid pixels
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix, minlength=npix)
    valid_pix = np.where(counts >= min_counts)[0]
    mask = np.zeros(npix, dtype=bool)
    mask[valid_pix] = True

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

    return rand_ra, rand_dec, rand_z, rand_nbar, rand_fkp


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
    


        