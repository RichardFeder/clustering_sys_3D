import numpy as np

from mask_and_randoms import *
from plotting_fns import *
from contamination import *
from desi_mocks import *
from pscalc import *
from utils import *
from dust import *
from star_sim import *

def compute_pkmu_mocks(nmock, k_min=0.005, k_max=0.2, delta_k=0.01, mu_min=0., mu_max=1., nwedge=6, \
                      ells = (0, 2, 4, 6, 8), nmesh=256, with_RSD=False, boxsize=1000., with_sfd_contam=False, sfd_std=0.01, plot=False, \
                      with_stellar_contam=False, use_gaia=True, frac_stellar_contam=0.01, save_fname=None, mu_wedges=None, \
                       redshift_sel=False, zmin=0.4, zmax=1.0, replicate=False, rep_fac=3, nplot=5, ds_fac=100, randomize=False, \
                      mock_type='quijote', n_sample=20_000_000):

    dm = desi_mock()

    if mu_wedges is None:
        mu_wedges = np.linspace(mu_min, mu_max, nwedge)
    else:
        nwedge = len(mu_wedges)

    ell_max = np.max(ells)
    mu1 = 1./(ell_max//2 + 1)
    print('ell max, mu1 = ', ell_max, mu1)
    
    if with_sfd_contam:
        ebv_map, cl_sfd = gen_sfd_hp()
        delta_ebv = gen_delta_ebv_map_uncorr(cl_sfd, std=sfd_std)
        delta_n_over_n_uncorr = gen_dn_n_map(grf_map=delta_ebv)

        if plot:
            hp.mollview(delta_ebv, title='$\\delta E(B-V)$', unit='mag', min=-0.01, max=0.01)
            hp.mollview(delta_n_over_n_uncorr, title='$\\delta N/N$', unit=None, min=-0.1, max=0.1)
    else:
        sfd_std = None

    if with_stellar_contam:
        if use_gaia:
            stellar_map = load_gaia_stellar_density()
    else:
        frac_stellar_contam = None

    kedges = np.arange(k_min, k_max + delta_k, delta_k) 
    edges=(kedges, mu_wedges)
    nkbin = len(kedges)-1
    
    all_pkmu = np.zeros((nmock, nkbin, nwedge-1))
    all_plk = np.zeros((nmock, len(ells), nkbin))

    print('all pkmu has shape', all_pkmu.shape)
    print('all plk has shape', all_plk.shape)

    if redshift_sel:
        dcom_to_z_interp = gen_interp_fn_dcom_z()

    if replicate:
        boxsize_use = boxsize*rep_fac
    else:
        boxsize_use = boxsize

    for mockidx in range(nmock):

        if mock_type=='quijote':
            galpos = dm.load_quijote_galpos(mockidx, with_RSD=with_RSD, replicate=replicate, rep_fac=rep_fac, ds_fac=ds_fac, \
                                           randomize=randomize)
            redshift = None
            center_offset_mpc = None
            
        elif mock_type=='halfdome':
            galpos, redshift = dm.load_halfdome_mock(mockidx, n_sample=n_sample)
            center_offset_mpc = 0.
            
        print('max galpos here is ', [np.max(galpos[:,i]) for i in range(3)])

        ra_gal, dec_gal, r_gal = convert_to_ra_dec_distance(galpos, boxsize_use, center_offset_mpc=center_offset_mpc)
        posarray = np.array([ra_gal, dec_gal, r_gal])

        print([np.max(posarray[i]) for i in range(3)])

        if mockidx==0 and plot:
            plot_hexbin_density(ra_gal, dec_gal)
            plot_hist(r_gal, label='r gal [Mpc/h]')

        if redshift_sel:

            if redshift is None:
                redshift = comoving_distance_to_redshift(r_gal/cosmo.h, dcom_to_z_interp)

            radial_sel = np.ones_like(redshift)
            if zmin is not None:
                radial_sel *= (redshift > zmin)
            if zmax is not None:
                radial_sel *= (redshift < zmax)

            redshift_select = redshift[np.where(radial_sel)[0]]
            plt.figure(figsize=(5, 4))
            plt.hist(redshift, histtype='step', bins=np.linspace(0, 2, 50))
            plt.hist(redshift_select, histtype='stepfilled', alpha=0.3, bins=np.linspace(0, 2, 50))
            plt.xlabel('redshift')
            plt.show()
                
            print('radial selection includes ', np.sum(radial_sel)/len(redshift))
            sel_idx = np.where(radial_sel)[0]
            posarray = posarray[:,sel_idx]

        
            print('posarray now has shape', posarray.shape)

            if mockidx==0 and plot:
                plot_hexbin_density(posarray[0], posarray[1])

            
        weights = np.ones_like(posarray[0])
        if with_sfd_contam:
            sys_weights, _ = modify_fkp_weights(posarray[0], posarray[1], weights, delta_n_over_n_uncorr)
        else:
            sys_weights = weights

        if with_stellar_contam:

            if use_gaia:
                realization = poisson_star_map_from_fraction(stellar_map, len(posarray[0]), frac=frac_stellar_contam, mask=None)
                ra_star, dec_star = generate_poisson_radec_from_map(realization)

            else:
                # ra_star, dec_star, notional_map = notional_radec_stellar_density(
                #                                 N_gal=len(posarray[0]),
                #                                 frac=frac_stellar_contam,
                #                                 nside=64,
                #                                 dec_peak_density=0.0,      # Peak stellar density at the celestial equator
                #                                 dec_decay_scale=35.0,      # Gentle decay away from the equator
                #                                 bulge_ra_deg=0,        # RA of Galactic Center (approx)
                #                                 bulge_dec_deg=0,       # Dec of Galactic Center (approx)
                #                                 bulge_scale_ra=30.0,       # Broader in RA
                #                                 bulge_scale_dec=10.0,       # Narrower in Dec
                #                                 bulge_amplitude=1.0,      # More pronounced bulge
                #                                 seed=43
                #                             )
                ra_star, dec_star, notional_map = simple_halo_thick_disk_stellar_density(
                                                        N_gal=len(posarray[0]),
                                                        frac=frac_stellar_contam,
                                                        nside=64,
                                                        halo_power_law_exp=0.3,    # Adjust this: higher means faster drop-off from GC
                                                        thick_disk_amplitude=1.8, # Increase relative thick disk contribution
                                                        thick_disk_b_scale=25.0,   # Make the thick disk slightly more extended
                                                        seed=45
                                                    )

            if mockidx==0:
                
                plt.figure(figsize=(7, 4))
                plt.scatter(ra_star, dec_star, color='k', s=2)
                plt.xlabel('RA')
                plt.ylabel('DEC')
                plt.show()

            # assume stars follow same redshift distribution as tracer (for now)
            rng = np.random.default_rng(seed=42)  # optional: set seed for reproducibility
            rand_indices = rng.choice(len(posarray[2]), size=len(ra_star), replace=True)
            r_star = posarray[2,rand_indices]
    
            # r_star = np.random.uniform(0, boxsize/2., len(ra_star))
            star_array = np.array([ra_star, dec_star, r_star])
            weights_star = np.ones_like(ra_star)

            posarray = np.concatenate((posarray, star_array), axis=1)
            sys_weights = np.concatenate((sys_weights, weights_star))

            print('posarray now has shape', posarray.shape)
            print('weights', weights.shape)

            
        n_randoms = 5 * posarray.shape[1] # Typically more randoms than data

        if redshift_sel:

            chi_interp = grab_chi_interp()
            plot_chi_interp(chi_interp)

            ra_rand, dec_rand, r_rand, z_rand = generate_uniform_randoms(chi_interp, n_randoms, zmin=zmin, zmax=zmax, data_z=redshift_select)
            # ra_rand, dec_rand, r_rand = generate_randoms_matching_selection(n_randoms, boxsize*rep_fac, dcom_to_z_interp, zmin, zmax)
            randarray = np.array([ra_rand, dec_rand, r_rand])

            minr, maxr = np.min(posarray[2]), np.max(posarray[2])

            rbins = np.linspace(0.9*minr, 1.1*maxr, 100)

            plt.figure(figsize=(5, 4))
            plt.hist(r_rand, bins=rbins, histtype='step', label='Randoms')
            plt.hist(posarray[2], bins=rbins, histtype='step', label='Data')
            plt.xlabel('$\\chi$ [Mpc/h]')
            plt.yscale('log')
            plt.legend()
            plt.show()
            
        else:
            randoms_positions = np.random.uniform(0, boxsize, (n_randoms, 3)) # Same bounds as mock
            print('n randoms:', n_randoms)
            ra_rand, dec_rand, r_rand = convert_to_ra_dec_distance(randoms_positions, boxsize)
            randarray = np.array([ra_rand, dec_rand, r_rand])

        if mockidx==0:
            print('randarray has shape', randarray.shape)
            plot_hexbin_density(randarray[0], randarray[1])

        print('max rand array:', [np.max(randarray[i]) for i in range(3)])
        if mockidx==0:
            plot_hist(weights, label='Selection weights')


        weights_rand = np.ones_like(ra_rand)

        result = CatalogFFTPower(
                    data_positions1=posarray,
                    data_weights1=sys_weights,
                    randoms_positions1=randarray,
                    randoms_weights1=weights_rand,
                    nmesh=nmesh,
                    los='z', position_type='rdd',
                    resampler='tsc',
                    dtype='f8', \
                    ells=ells, \
                    edges=edges,
                )

        pkmu = result.wedges.get_power()

        if mockidx==0:
            kcen = result.wedges.k[:,0]

        all_plk[mockidx] = result.poles.power
        all_pkmu[mockidx] = pkmu

        if plot and mockidx<nplot:
            fig = plot_pkmu(kcen, pkmu, mu_wedges=mu_wedges)

        
    if save_fname is not None:
        np.savez('data/plk/'+save_fname, kcen=kcen, all_pkmu=all_pkmu, all_plk=all_plk, ells=ells, with_RSD=with_RSD, sfd_std=sfd_std, frac_stellar_contam=frac_stellar_contam, \
                mu_wedges=mu_wedges)

    return kcen, all_pkmu