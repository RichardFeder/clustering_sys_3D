import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

def compare_data_randoms_healpy(ra_data, dec_data, ra_rand, dec_rand, nside=128):
    """Plot HEALPix binned sky density maps and their ratio for data vs randoms."""
    # Convert RA/DEC to theta/phi (in radians) for healpy
    theta_data = np.radians(90. - dec_data)
    phi_data = np.radians(ra_data)
    
    theta_rand = np.radians(90. - dec_rand)
    phi_rand = np.radians(ra_rand)

    npix = hp.nside2npix(nside)

    # Fill HEALPix maps (counts per pixel)
    map_data = np.zeros(npix)
    map_randoms = np.zeros(npix)

    pix_data = hp.ang2pix(nside, theta_data, phi_data)
    pix_rand = hp.ang2pix(nside, theta_rand, phi_rand)

    for pix in pix_data:
        map_data[pix] += 1
    for pix in pix_rand:
        map_randoms[pix] += 1

    # Normalize maps
    map_data /= np.sum(map_data)
    map_randoms /= np.sum(map_randoms)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': 'mollweide'})
    
    hp.mollview(map_data, title='Data density (normalized)', sub=(1,3,1), fig=fig.number, cmap='viridis')
    hp.mollview(map_randoms, title='Random density (normalized)', sub=(1,3,2), fig=fig.number, cmap='viridis')

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(map_randoms > 0, map_data / map_randoms, 0)

    hp.mollview(ratio, title='Data / Randoms (angular mask)', sub=(1,3,3), fig=fig.number, cmap='coolwarm', min=0.5, max=1.5)
    
    plt.tight_layout()
    plt.show()

    return map_data, map_randoms, ratio

def compare_ps_recov(k, plk_contam, pkl_clean, figsize=(8, 4)):
    ''' Assume list of PS multipoles '''
    
    pk0, pk2, pk4 = pkl_contam
    pk0_clean, pk4_clean, pk4_clean = pkl_clean

    # --- 7. Plot comparison ---
    fig = plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.title("Power Spectrum Multipoles")
    plt.plot(k, pk0, label='Monopole contaminated', color='blue')
    plt.plot(k, pk2, label='Quadrupole contaminated', color='orange')
    plt.plot(k, pk4, label='Hexadecapole contaminated', color='green')
    plt.xlabel(r'$k$ [$h$/Mpc]')
    plt.ylabel(r'$P_\ell(k)$ [(Mpc/h)$^3$]')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.title("Fractional Change vs Clean")
    plt.plot(k, (pk0 - pk0_clean) / pk0_clean, label='Monopole', color='blue')
    plt.plot(k, (pk2 - pk2_clean) / (pk2_clean + 1e-8), label='Quadrupole', color='orange')
    plt.plot(k, (pk4 - pk4_clean) / (pk4_clean + 1e-8), label='Hexadecapole', color='green')
    plt.axhline(0, color='k', ls='--')
    plt.xlabel(r'$k$ [$h$/Mpc]')
    plt.ylabel(r'Fractional Change $\Delta P / P$')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

    return fig


def plot_poles_pk(result):
    # first plot is Pl(k), second is k*Pl(k)
    
    poles = result.poles
    print('Shot noise is {:.4f}.'.format(poles.shotnoise))
    print('Normalization is {:.4f}.'.format(poles.wnorm))
    ax = plt.gca()
    for ill, ell in enumerate(poles.ells):
        # Calling poles() removes shotnoise for ell == 0 by default;
        # Pass remove_shotnoise = False if you do not want to;
        # See get_power() for all arguments
        ax.plot(*poles(ell=ell, return_k=True, complex=False), label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]', fontsize=14)
    ax.set_ylabel(r'$P(k)$ [$(\mathrm{Mpc}/h)^{3}$]', fontsize=14)
    plt.show()

    poles[::2].plot(show=True);



def plot_contam(ra, dec, z, zerr):

    # Quick plot
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(z, bins=50)
    plt.title("Contam redshift")

    plt.subplot(1, 3, 2)
    plt.scatter(ra, dec, s=1, alpha=0.5)
    plt.title("Contam sky positions")

    plt.subplot(1, 3, 3)
    plt.hist(zerr, bins=50)
    plt.title("Redshift error distribution")
    plt.tight_layout()
    plt.show()

    return fig


    