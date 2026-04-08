import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D visualization (optional)
from matplotlib.cm import get_cmap
from astropy.cosmology import Planck18 as cosmo

def plot_hexbin_density(xpos, ypos, gridsize=200, cmap='plasma', mincnt=1, figsize=(7, 4), \
                       xlabel='RA', ylabel='DEC', return_fig=False):

    fig = plt.figure(figsize=(7, 4))
    plt.title('Randoms sky position after radial sel')
    plt.hexbin(xpos, ypos, gridsize=200, cmap='plasma', mincnt=1)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.show()

    if return_fig:
        return fig


def plot_pkmu(kcen, pkmu, mu_wedges=None, figsize=(5, 4), ylim=None, return_fig=False):

    
    fig = plt.figure(figsize=figsize)            
    for muidx in range(pkmu.shape[1]):

        if mu_wedges is not None:
            mu_label = str(np.round(mu_wedges[muidx], 2))+'$<\\mu<$'+str(np.round(mu_wedges[muidx+1], 2))
        else:
            mu_label = '$\\mu$ bin '+str(muidx)

        if muidx==0:
            linewidth = 3
        else:
            linewidth = 1.5

        kpkmu = kcen*pkmu[:,muidx]
        plt.plot(kcen, kcen*pkmu[:,muidx], label=mu_label, color='C'+str(muidx))
    
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('$k P(k,\\mu)$ [$(Mpc/h)^2$]', fontsize=12)
    plt.legend(ncol=2)
    plt.grid(alpha=0.2)
    plt.ylim(ylim)
    plt.show()

    if return_fig:
        return fig


def compare_pkmu_wsys(fpath_clean, fpath_wsys=None, \
                      title='$\\overline{n}=10^{-3}$, no RSD \n + Systematic 1% stellar contamination', \
                     ylim_ratio=[0.8, 2.0], ylim_ps=[300, 1500], cmap='jet', color_mu0='b', title_fs=14, \
                     figsize_ps=(10, 4), figsize_ratio=(5, 4), mu_wedges=None, loc=4, legend_fs=10, \
                     cleanlab='No systematics', syslab='With 10% stellar contamination', lab_fs=14, style=None):

    clean_res = np.load(fpath_clean)
    
    mu_wedges = clean_res['mu_wedges']
    print('mu wedges:', mu_wedges)
    kcen, pkmu_clean = clean_res['kcen'], clean_res['all_pkmu']
    
    if fpath_wsys is not None:
        sys_res = np.load(fpath_wsys)

        kcen, pkmu_wsys = sys_res['kcen'], sys_res['all_pkmu']
        ncols = 2
        axidx_lab = [0, 1]
    else:
        ncols = 1
        axidx_lab = [0]

    cmap = get_cmap(cmap)
    linsp = np.linspace(0.2, 1.0, len(mu_wedges)-1)
    print(linsp)
    colors = [cmap(i) for i in linsp]
    colors[0] = color_mu0

    
    fig_ps, ax = plt.subplots(figsize=figsize_ps, ncols=ncols, sharey=True)
    
    # fig_ps = plt.figure(figsize=figsize_ps)
    # plt.title(title, fontsize=title_fs)

    ax[0].set_title(cleanlab, fontsize=title_fs)
    
    for muidx in range(pkmu_clean.shape[2]):
        if muidx==0:
            linewidth = 3
            zorder=10
        else:
            linewidth = 2
            zorder=1

        if mu_wedges is None:
            mu_label = '$\\mu$ bin '+str(muidx)
        else:
            mu_label = str(np.round(mu_wedges[muidx], 2))+'$<\\mu<$'+str(np.round(mu_wedges[muidx+1], 2))
        ax[0].errorbar(kcen, kcen*np.mean(pkmu_clean[:,:,muidx], axis=0), yerr=kcen*np.std(pkmu_clean[:,:,muidx], axis=0)/np.sqrt(pkmu_clean.shape[0]-1), label=mu_label, color=colors[muidx], \
                    linewidth=linewidth, zorder=zorder)

        if fpath_wsys is not None:
            if muidx==0:
                ax[1].set_title(syslab, fontsize=title_fs)

            ax[1].errorbar(kcen, kcen*np.mean(pkmu_wsys[:,:,muidx], axis=0), yerr=kcen*np.std(pkmu_wsys[:,:,muidx], axis=0)/np.sqrt(pkmu_wsys.shape[0]-1), color=colors[muidx], \
                    linewidth=linewidth, linestyle='dashed', zorder=zorder)

    ax[0].set_ylabel('$k P(k,\\mu)$ [$(Mpc/h)^2$]', fontsize=lab_fs)

    for axidx in axidx_lab:
        ax[axidx].set_xlabel('k [h/Mpc]', fontsize=lab_fs)
        ax[axidx].set_ylim(ylim_ps)
        ax[axidx].grid(alpha=0.2)

    ax[0].legend(loc=loc, ncol=2, fontsize=legend_fs)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()

    fig_ratio = None
    if fpath_wsys is not None:

        if style is not None:
            plt.style.use(style)

        fig_ratio = plt.figure(figsize=figsize_ratio)
        plt.title(title, fontsize=title_fs)
        for muidx in range(pkmu_clean.shape[2]):
            if muidx==0:
                linewidth = 3
                zorder=10
            else:
                linewidth = 1.5
                zorder = 5

            
            mulab = str(np.round(mu_wedges[muidx], 2))+'$<\\mu<$'+str(np.round(mu_wedges[muidx+1], 2))
            ratio_wsys = pkmu_wsys[:,:,muidx]/pkmu_clean[:,:,muidx]
            mean_ratio, std_ratio = np.mean(ratio_wsys, axis=0), np.std(ratio_wsys, axis=0)
            plt.errorbar(kcen, mean_ratio, yerr=std_ratio/np.sqrt(ratio_wsys.shape[0]-1), label=mulab, zorder=zorder, color=colors[muidx], linewidth=linewidth, linestyle='dashed')
        
        plt.xlabel('k [h/Mpc]', fontsize=lab_fs)
        plt.ylabel('$\\frac{P(k,\\mu)^{wsys}}{P(k,\\mu)^{clean}}$', fontsize=20)
        plt.xscale('log')
        plt.legend(ncol=2)
        plt.grid(alpha=0.8)
        plt.ylim(ylim_ratio)
        plt.show()

        if style is not None:
            plt.style.use('default')
    

    return fig_ps, fig_ratio

            
def plot_hist(vals, figsize=(5, 4), bins=50, label='Weights'):
    plt.figure(figsize=figsize)
    plt.hist(vals, bins=bins)
    plt.yscale('log')
    plt.xlabel(label)
    plt.show()

def plot_chi_interp(chi_interp, zmin=0.01, zmax=2.0, nbin=1000):
        
    linsp = np.linspace(zmin, zmax, nbin)
    plt.figure()
    plt.plot(linsp, chi_interp(linsp))
    plt.xlabel('redshift')
    plt.ylabel('comoving distance')
    plt.show()

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


def generate_3d_angular_mask(
    angular_mask_path: str,
    L_box: float,
    N_voxels_per_side: int,
    mask_value_unmasked: int = 1, # Value in HEALPix map indicating unmasked regions
    plot_2d_slice: bool = False, # Option to plot a 2D slice of the 3D mask
    plot_3d_scatter: bool = False, # Option to plot a sparse 3D scatter of masked points
    zmin: float = None, # Minimum redshift for radial selection (inclusive)
    zmax: float = None, # Maximum redshift for radial selection (inclusive)
) -> np.ndarray:
    """
    Generates a 3D boolean mask for a cubic mock based on a 2D angular HEALPix mask.

    Assumes the observer is at the center (0,0,0) of the cubic volume.

    Parameters
    ----------
    angular_mask_path : str
        Path to the HEALPix FITS file containing the angular mask.
    L_box : float
        Side length of the cubic simulation box in physical units (e.g., Mpc/h).
        The coordinates will range from -L_box/2 to +L_box/2.
    N_voxels_per_side : int
        Number of voxels along each side of the cubic mock. The resulting 3D mask
        will have dimensions (N_voxels_per_side, N_voxels_per_side, N_voxels_per_side).
    mask_value_unmasked : int, optional
        The value in the HEALPix map that corresponds to an "unmasked" or valid region.
        Commonly 1 for binary masks. Defaults to 1.
    plot_2d_slice : bool, optional
        If True, a 2D slice (e.g., XY plane at Z=0) of the generated 3D mask
        will be displayed. Defaults to False.
    plot_3d_scatter : bool, optional
        If True, a sparse 3D scatter plot of the masked points will be displayed.
        This can be slow for large N_voxels_per_side. Defaults to False.

    Returns
    -------
    np.ndarray
        A 3D boolean NumPy array of shape (N_voxels_per_side, N_voxels_per_side, N_voxels_per_side)
        where True indicates that the voxel is within the angular mask, and False otherwise.
    """
    print(f"--- Generating 3D Angular Mask for {L_box:.2f} box, {N_voxels_per_side}^3 voxels ---")

    # 1. Load the angular HEALPix mask
    try:
        angular_mask = hp.read_map(angular_mask_path, field=0, verbose=False)
        nside_mask = hp.npix2nside(len(angular_mask))
        print(f"Loaded HEALPix mask (NSIDE={nside_mask}) from {angular_mask_path}")
    except FileNotFoundError:
        print(f"Error: Angular mask file not found at {angular_mask_path}")
        return np.array([])
    except Exception as e:
        print(f"Error loading HEALPix mask: {e}")
        return np.array([])


    hp.mollview(angular_mask, title="HEALPix Mask Visualization", cmap='binary', unit='Mask Value')

    # 2. Define the 3D coordinate grid for the mock
    # Coordinates range from -L_box/2 to +L_box/2
    coords = np.linspace(-L_box / 2, L_box / 2, N_voxels_per_side)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij') # 'ij' for (Z,Y,X) or (X,Y,Z) consistent with numpy array indexing

    # 3. Convert Cartesian (x, y, z) to Spherical (RA, Dec)
    # Ensure correct order for arctan2 (y, x) and arcsin (z/r)
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Handle the observer's exact position (r=0) to avoid division by zero in arcsin.
    # For a continuous grid, this is usually just one point. We can assign it a default
    # masked value or handle it as a special case. Here, we'll assign NaN to avoid error
    # and then mask it out.
    R[R == 0] = np.nan # Mark observer's point as NaN

    ra_rad = np.arctan2(Y, X)
    dec_rad = np.arcsin(Z / R) # Z / R will be NaN where R is NaN

    # Convert radians to degrees
    ra_deg = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    # Ensure RA is in [0, 360)
    ra_deg[ra_deg < 0] += 360

    # 4. Map (RA, Dec) to HEALPix pixel indices
    # Use 'lonlat=True' because RA/Dec are in degrees.
    # 'nest=False' assumes RING ordering for the mask (default for hp.read_map).
    # Adjust 'nest' if your mask is in NESTED ordering.
    galaxy_pixels = hp.ang2pix(nside_mask, ra_deg, dec_deg, lonlat=True, nest=False)

    # 5. Create the 3D mask by checking the angular mask values
    # Initialize 3D mask as boolean array
    mask_3d = np.zeros((N_voxels_per_side, N_voxels_per_side, N_voxels_per_side), dtype=bool)

    # Check if the pixel value in the angular mask corresponds to an unmasked region
    # Also handle NaN values from the observer's position (R=0)
    valid_pixels = ~np.isnan(galaxy_pixels) # Exclude NaN pixels (observer's position)
    
    # Apply the angular mask condition
    # Ensure galaxy_pixels is cast to integer for indexing
    mask_3d[valid_pixels] = (angular_mask[galaxy_pixels[valid_pixels].astype(int)] == mask_value_unmasked)

    # --- Apply optional radial selection based on zmin/zmax ---
    if (zmin is not None) or (zmax is not None):
        # Convert redshift boundaries to comoving distance in same units as L_box
        rmin = None
        rmax = None
        if zmin is not None:
            rmin = cosmo.comoving_distance(zmin).value * cosmo.h  # Mpc/h
        if zmax is not None:
            rmax = cosmo.comoving_distance(zmax).value * cosmo.h  # Mpc/h

        # Ensure positions at observer are excluded
        radial_keep = np.ones_like(mask_3d, dtype=bool)
        if rmin is not None:
            radial_keep[R < rmin] = False
        if rmax is not None:
            radial_keep[R > rmax] = False

        # Combine angular and radial selections
        mask_3d = mask_3d & radial_keep

    print(f"Generated 3D mask. Number of unmasked voxels: {np.sum(mask_3d)}")

    # --- Optional Visualization ---
    if plot_2d_slice:
        plt.figure(figsize=(7, 7))
        # Take a slice through the center (e.g., Z=0, or middle slice)
        center_slice_idx = N_voxels_per_side // 2
        plt.imshow(mask_3d[:, :, center_slice_idx].T, origin='lower', cmap='binary',
                   extent=[coords.min(), coords.max(), coords.min(), coords.max()])
        plt.title(f'2D Slice of 3D Mask (XY plane at Z={coords[center_slice_idx]:.2f})')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.colorbar(label='Masked (0) / Unmasked (1)')
        plt.grid(True, alpha=0.5)
        plt.show()

    if plot_3d_scatter:
        # This can be very slow and memory-intensive for large N_voxels_per_side.
        # Plotting only a sparse subset of points for visualization.
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get coordinates of masked points
        masked_indices = np.where(mask_3d)
        
        # To avoid plotting too many points for visualization, sample them
        # For example, plot only 1% of the masked points if there are many
        if len(masked_indices[0]) > 10000: # Adjust threshold as needed
            sample_size = 10000
            random_indices = np.random.choice(len(masked_indices[0]), sample_size, replace=False)
            x_plot = X[masked_indices][random_indices]
            y_plot = Y[masked_indices][random_indices]
            z_plot = Z[masked_indices][random_indices]
            print(f"Plotting a random sample of {sample_size} masked points for 3D visualization.")
        else:
            x_plot = X[masked_indices]
            y_plot = Y[masked_indices]
            z_plot = Z[masked_indices]
            print(f"Plotting all {len(masked_indices[0])} masked points for 3D visualization.")

        ax.scatter(x_plot, y_plot, z_plot, s=1, alpha=0.1, c='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Masked Volume')
        ax.set_xlim(coords.min(), coords.max())
        ax.set_ylim(coords.min(), coords.max())
        ax.set_zlim(coords.min(), coords.max())
        plt.show()

    return mask_3d


def generate_3d_completeness_map(
    hp_map: np.ndarray,
    L_box: float,
    N_voxels_per_side: int,
    zmin: float = None,
    zmax: float = None,
    plot_2d_slice: bool = False,
) -> np.ndarray:
    """
    Generates a 3D float completeness map from a continuous HEALPix map (e.g. fracarea).

    Each voxel is assigned the completeness value of its corresponding HEALPix pixel.
    NaN pixels in the input map (masked sky regions) map to 0.0 in the output.

    Parameters
    ----------
    hp_map : np.ndarray
        HEALPix map of completeness values (e.g. fracarea). Shape must be 12*nside**2.
        NaN entries are treated as fully masked (completeness = 0).
    L_box : float
        Side length of the cubic box in Mpc/h. Observer is at the center.
    N_voxels_per_side : int
        Number of voxels along each side.
    zmin, zmax : float, optional
        Redshift limits for a radial selection. Voxels outside this shell are set to 0.
    plot_2d_slice : bool, optional
        If True, plot the central XY slice of the 3D map.

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, N, N) with per-voxel completeness values in [0, 1].
    """
    nside = hp.npix2nside(len(hp_map))
    print(f"--- Generating 3D completeness map from NSIDE={nside} array, "
          f"{L_box:.0f} Mpc/h box, {N_voxels_per_side}^3 voxels ---")

    coords = np.linspace(-L_box / 2, L_box / 2, N_voxels_per_side)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

    R = np.sqrt(X**2 + Y**2 + Z**2)
    R[R == 0] = np.nan

    ra_deg = np.degrees(np.arctan2(Y, X))
    dec_deg = np.degrees(np.arcsin(Z / R))
    ra_deg[ra_deg < 0] += 360

    pixels = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True, nest=False)

    completeness_3d = np.zeros((N_voxels_per_side, N_voxels_per_side, N_voxels_per_side), dtype=np.float32)
    vals = hp_map[pixels.astype(int)]
    valid = np.isfinite(vals)
    completeness_3d[valid] = vals[valid].astype(np.float32)

    if (zmin is not None) or (zmax is not None):
        rmin = cosmo.comoving_distance(zmin).value * cosmo.h if zmin is not None else None
        rmax = cosmo.comoving_distance(zmax).value * cosmo.h if zmax is not None else None
        if rmin is not None:
            completeness_3d[R < rmin] = 0.0
        if rmax is not None:
            completeness_3d[R > rmax] = 0.0

    print(f"Generated 3D completeness map. Non-zero voxels: {np.sum(completeness_3d > 0)}")

    if plot_2d_slice:
        center = N_voxels_per_side // 2
        plt.figure(figsize=(7, 6))
        plt.imshow(completeness_3d[:, :, center].T, origin='lower', cmap='viridis',
                   extent=[coords.min(), coords.max(), coords.min(), coords.max()])
        plt.title(f'Completeness slice (XY plane, Z={coords[center]:.1f} Mpc/h)')
        plt.xlabel('X [Mpc/h]')
        plt.ylabel('Y [Mpc/h]')
        plt.colorbar(label='Completeness (fracarea)')
        plt.show()

    return completeness_3d
