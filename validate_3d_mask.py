"""
Validation and visualization for generate_3d_angular_mask.

Usage:
    mask_3d = generate_3d_angular_mask(angular_mask_path, L_box, N_voxels_per_side)
    angular_mask = hp.read_map(angular_mask_path)
    validate_3d_mask(mask_3d, angular_mask, L_box)
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def validate_3d_mask(mask_3d, angular_mask, L_box, n_check=200_000, mask_value_unmasked=1):
    """
    Validate and visualize a 3D boolean mask against its source 2D angular HEALPix mask.

    Tests
    -----
    1. Back-projection: every True voxel must map to an unmasked HEALPix pixel.
    2. Radial uniformity: the True fraction in spherical shells should be flat
       and equal to the angular mask sky fraction.
    3. Center slices (XY, XZ, YZ) to visually inspect the 3D structure.
    4. Mollweide back-projection vs. the original angular mask.

    Parameters
    ----------
    mask_3d : (N, N, N) bool ndarray
        Output of generate_3d_angular_mask.
    angular_mask : 1D ndarray
        The HEALPix angular mask (same one passed to generate_3d_angular_mask).
    L_box : float
        Box side length in whatever units positions are in.
    n_check : int
        Max number of masked voxels to sample for back-projection tests.
    mask_value_unmasked : int
        Value in angular_mask meaning "unmasked" (usually 1).
    """
    N = mask_3d.shape[0]
    coords = np.linspace(-L_box / 2, L_box / 2, N)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    nside = hp.npix2nside(len(angular_mask))

    # ------------------------------------------------------------------ #
    # TEST 1: Back-projection
    # Every voxel marked True must land in an unmasked angular pixel.
    # ------------------------------------------------------------------ #
    masked_flat = np.where(mask_3d.ravel())[0]
    rng = np.random.default_rng(42)
    if len(masked_flat) > n_check:
        masked_flat = rng.choice(masked_flat, size=n_check, replace=False)

    xm = X.ravel()[masked_flat]
    ym = Y.ravel()[masked_flat]
    zm = Z.ravel()[masked_flat]
    rm = R.ravel()[masked_flat]

    nonzero = rm > 0          # exclude observer voxel
    xm, ym, zm, rm = xm[nonzero], ym[nonzero], zm[nonzero], rm[nonzero]

    ra_back  = np.degrees(np.arctan2(ym, xm)) % 360
    dec_back = np.degrees(np.arcsin(np.clip(zm / rm, -1, 1)))
    pix_back = hp.ang2pix(nside, ra_back, dec_back, lonlat=True, nest=False)

    n_wrong = np.sum(angular_mask[pix_back] != mask_value_unmasked)
    print(f"[Test 1] Back-projection: {n_wrong}/{len(pix_back)} masked voxels "
          f"land in a MASKED angular pixel (should be 0).")

    # Also check that no True voxels were missed by checking False voxels too
    unmasked_flat = np.where(~mask_3d.ravel())[0]
    if len(unmasked_flat) > n_check:
        unmasked_flat = rng.choice(unmasked_flat, size=n_check, replace=False)

    xu = X.ravel()[unmasked_flat]
    yu = Y.ravel()[unmasked_flat]
    zu = Z.ravel()[unmasked_flat]
    ru = R.ravel()[unmasked_flat]
    nonzero_u = ru > 0
    xu, yu, zu, ru = xu[nonzero_u], yu[nonzero_u], zu[nonzero_u], ru[nonzero_u]

    ra_u  = np.degrees(np.arctan2(yu, xu)) % 360
    dec_u = np.degrees(np.arcsin(np.clip(zu / ru, -1, 1)))
    pix_u = hp.ang2pix(nside, ra_u, dec_u, lonlat=True, nest=False)
    n_should_be_true = np.sum(angular_mask[pix_u] == mask_value_unmasked)
    print(f"[Test 1] Back-projection: {n_should_be_true}/{len(pix_u)} False voxels "
          f"land in an UNMASKED angular pixel (should be 0 if mask is consistent).")

    # ------------------------------------------------------------------ #
    # TEST 2: Radial uniformity
    # Mask fraction in spherical shells should be constant ≈ sky fraction.
    # ------------------------------------------------------------------ #
    sky_frac = np.mean(angular_mask == mask_value_unmasked)
    print(f"\n[Test 2] Angular mask sky fraction: {sky_frac:.4f}")

    r_flat = R.ravel()
    m_flat = mask_3d.ravel().astype(float)

    # Use the inscribed sphere radius to stay inside the box
    r_max = L_box / 2
    r_edges = np.linspace(0, r_max, 25)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    shell_fracs = []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        in_shell = (r_flat >= r0) & (r_flat < r1)
        shell_fracs.append(m_flat[in_shell].mean() if in_shell.sum() > 0 else np.nan)

    deviation = np.nanmax(np.abs(np.array(shell_fracs) - sky_frac))
    print(f"[Test 2] Max deviation from expected sky fraction across shells: {deviation:.4f}")

    # ------------------------------------------------------------------ #
    # PLOT 1: Radial uniformity
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_centers, shell_fracs, 'o-', label='Measured')
    ax.axhline(sky_frac, color='r', ls='--', label=f'Expected ({sky_frac:.3f})')
    ax.set_xlabel('Radial distance [box units]')
    ax.set_ylabel('Fraction of voxels = True')
    ax.set_title('Radial shell mask fraction\n(should be flat at sky fraction)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # PLOT 2: Center slices XY / XZ / YZ
    # ------------------------------------------------------------------ #
    ci = N // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    slice_specs = [
        (mask_3d[:, :, ci].T, 'XY slice (Z=0)', 'X', 'Y'),
        (mask_3d[:, ci, :].T, 'XZ slice (Y=0)', 'X', 'Z'),
        (mask_3d[ci, :, :].T, 'YZ slice (X=0)', 'Y', 'Z'),
    ]
    for ax, (data, title, xl, yl) in zip(axes, slice_specs):
        ax.imshow(data, origin='lower', cmap='binary',
                  extent=[coords.min(), coords.max(), coords.min(), coords.max()],
                  vmin=0, vmax=1)
        # Inscribed sphere
        circle = plt.Circle((0, 0), L_box / 2, color='red', fill=False, lw=1.2,
                             ls='--', label='Inscribed sphere')
        ax.add_patch(circle)
        ax.set_title(title)
        ax.set_xlabel(f'{xl} [box units]')
        ax.set_ylabel(f'{yl} [box units]')
    axes[0].legend(fontsize=8)
    plt.suptitle('Center slices of 3D mask  (white=True, black=False)', y=1.02)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # PLOT 3: Mollweide — original mask vs. back-projected voxels
    # ------------------------------------------------------------------ #
    backproj_map = np.zeros(hp.nside2npix(nside))
    np.add.at(backproj_map, pix_back, 1)

    fig = plt.figure(figsize=(14, 4))
    hp.mollview(angular_mask, fig=fig, sub=(1, 2, 1),
                title='Input angular mask', cmap='binary', unit='mask value')
    hp.mollview(backproj_map, fig=fig, sub=(1, 2, 2),
                title='Back-projected True voxels (count)', cmap='plasma')
    plt.suptitle('Angular mask vs. back-projection of 3D mask\n'
                 '(footprints should match)', y=1.02)
    plt.show()

    # ------------------------------------------------------------------ #
    # PLOT 4: Thin radial shell in 3D — confirm angular shape
    # Project a shell at ~r_max/2 to confirm it looks like the angular mask.
    # ------------------------------------------------------------------ #
    r_shell_lo = 0.45 * r_max
    r_shell_hi = 0.55 * r_max
    in_shell_3d = (R >= r_shell_lo) & (R <= r_shell_hi)

    shell_mask_true  = in_shell_3d & mask_3d
    shell_mask_false = in_shell_3d & ~mask_3d

    xs_t = X[shell_mask_true].ravel()
    ys_t = Y[shell_mask_true].ravel()
    zs_t = Z[shell_mask_true].ravel()
    rs_t = R[shell_mask_true].ravel()

    ra_shell  = np.degrees(np.arctan2(ys_t, xs_t)) % 360
    dec_shell = np.degrees(np.arcsin(np.clip(zs_t / rs_t, -1, 1)))

    shell_map = np.zeros(hp.nside2npix(nside))
    pix_shell = hp.ang2pix(nside, ra_shell, dec_shell, lonlat=True, nest=False)
    np.add.at(shell_map, pix_shell, 1)

    fig = plt.figure(figsize=(14, 4))
    hp.mollview(angular_mask, fig=fig, sub=(1, 2, 1),
                title='Input angular mask', cmap='binary')
    hp.mollview(shell_map, fig=fig, sub=(1, 2, 2),
                title=f'Shell {r_shell_lo:.0f}–{r_shell_hi:.0f} voxels → sky',
                cmap='plasma')
    plt.suptitle('Shell back-projection (should trace the angular mask boundary)', y=1.02)
    plt.show()

    return {'shell_fracs': shell_fracs, 'r_centers': r_centers,
            'sky_frac': sky_frac, 'n_backproj_wrong': n_wrong}


def plot_lightcone_wedge(mask_3d, L_box, dec_slice_width=10., ra_slice_center=180., ra_slice_width=30.,
                         rmin=None, rmax=None, max_voxels=300_000):
    """
    Polar wedge and slice plots of the 3D lightcone mask.

    Makes four plots:
    1. RA polar wedge  — equatorial slice (|Dec| < dec_slice_width/2).
    2. Dec polar wedge — thin RA slice through the survey.
    3. Cartesian RA vs r — easier to read exact RA boundaries.
    4. XY center slice — shows the annular shell shape directly in box coordinates.

    Parameters
    ----------
    mask_3d : (N, N, N) bool ndarray
    L_box : float
        Box side length (Mpc/h).
    dec_slice_width : float
        Full width of the equatorial Dec slice in degrees (default 10).
    ra_slice_center : float
        RA center for the Dec-wedge slice in degrees (default 180).
    ra_slice_width : float
        Full width of the RA slice for the Dec wedge in degrees (default 30).
    rmin, rmax : float or None
        Expected inner/outer shell radii (Mpc/h). If provided, reference circles
        are drawn on the polar plots and lines on the Cartesian plot.
    max_voxels : int
        Subsample True voxels to at most this many before plotting.
    """
    N = mask_3d.shape[0]
    coords = np.linspace(-L_box / 2, L_box / 2, N)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Back-project all True voxels to RA / Dec / r
    idx = np.where(mask_3d.ravel() & (R.ravel() > 0))[0]
    rng = np.random.default_rng(42)
    if len(idx) > max_voxels:
        idx = rng.choice(idx, size=max_voxels, replace=False)

    xv = X.ravel()[idx]
    yv = Y.ravel()[idx]
    zv = Z.ravel()[idx]
    rv = R.ravel()[idx]

    ra  = np.degrees(np.arctan2(yv, xv)) % 360
    dec = np.degrees(np.arcsin(np.clip(zv / rv, -1, 1)))

    def _add_radial_circles(ax, rmin, rmax, rmax_plot):
        """Draw reference circles at rmin/rmax on a polar axis."""
        theta = np.linspace(0, 2 * np.pi, 360)
        if rmin is not None:
            ax.plot(theta, np.full_like(theta, rmin), 'r--', lw=1.2,
                    label=f'rmin={rmin:.0f}')
        if rmax is not None:
            ax.plot(theta, np.full_like(theta, rmax), 'g--', lw=1.2,
                    label=f'rmax={rmax:.0f}')
        ax.legend(fontsize=7, loc='lower right')

    # ------------------------------------------------------------------ #
    # PLOT 1 & 2: Polar wedges
    # ------------------------------------------------------------------ #
    eq_sel = np.abs(dec) < dec_slice_width / 2
    ra_eq  = ra[eq_sel]
    r_eq   = rv[eq_sel]

    ra_diff = ((ra - ra_slice_center + 180) % 360) - 180
    ra_sel  = np.abs(ra_diff) < ra_slice_width / 2
    dec_sel = dec[ra_sel]
    r_sel   = rv[ra_sel]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={'projection': 'polar'})

    ax = axes[0]
    ax.scatter(np.radians(ra_eq), r_eq, s=0.5, alpha=0.15, c='steelblue', rasterized=True)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_title(f'RA wedge  |Dec| < {dec_slice_width/2:.0f}°\n({eq_sel.sum():,} voxels)', pad=15)
    ax.set_rlabel_position(45)
    for ra_mark, lbl in [(0,'RA=0'), (90,'RA=90'), (180,'RA=180'), (270,'RA=270')]:
        ax.axvline(np.radians(ra_mark), color='gray', lw=0.7, ls='--', alpha=0.5)
    _add_radial_circles(ax, rmin, rmax, ax.get_rmax())

    ax2 = axes[1]
    ax2.scatter(np.radians(dec_sel), r_sel, s=0.5, alpha=0.15, c='darkorange', rasterized=True)
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    ax2.set_title(f'Dec wedge  |RA − {ra_slice_center:.0f}°| < {ra_slice_width/2:.0f}°\n'
                  f'({ra_sel.sum():,} voxels)', pad=15)
    ax2.set_rlabel_position(45)
    for dec_mark, lbl in [(-30,'Dec=−30'), (0,'Dec=0'), (30,'Dec=30'), (60,'Dec=60')]:
        ax2.axvline(np.radians(dec_mark), color='gray', lw=0.7, ls='--', alpha=0.5)
        ax2.text(np.radians(dec_mark), ax2.get_rmax() * 1.05, lbl, ha='center', fontsize=7)
    _add_radial_circles(ax2, rmin, rmax, ax2.get_rmax())

    plt.suptitle('Lightcone wedge plots  (observer at center, r = comoving distance)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # PLOT 3: Cartesian — RA vs r
    # ------------------------------------------------------------------ #
    fig, ax3 = plt.subplots(figsize=(12, 4))
    sc = ax3.scatter(ra[eq_sel], r_eq, c=dec[eq_sel], s=0.5, alpha=0.2,
                     cmap='RdBu_r', vmin=-dec_slice_width / 2, vmax=dec_slice_width / 2,
                     rasterized=True)
    plt.colorbar(sc, ax=ax3, label='Dec [deg]')
    if rmin is not None:
        ax3.axhline(rmin, color='r', ls='--', lw=1.2, label=f'rmin={rmin:.0f} Mpc/h')
    if rmax is not None:
        ax3.axhline(rmax, color='g', ls='--', lw=1.2, label=f'rmax={rmax:.0f} Mpc/h')
    if rmin is not None or rmax is not None:
        ax3.legend(fontsize=9)
    ax3.set_xlabel('RA [deg]')
    ax3.set_ylabel('Comoving distance [Mpc/h]')
    ax3.set_title(f'RA vs. comoving distance  |Dec| < {dec_slice_width/2:.0f}°')
    ax3.set_xlim(0, 360)
    ax3.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # PLOT 4: XY center slice — shows annular shell in box coordinates
    # This is the most direct view: the angular mask carves the shell into
    # sectors. Each filled arc should span exactly rmin→rmax.
    # ------------------------------------------------------------------ #
    ci = N // 2
    xy_slice = mask_3d[:, :, ci]   # Z=0 plane; shape (N, N)
    r_xy = R[:, :, ci]             # radius in that plane

    fig, axes4 = plt.subplots(1, 2, figsize=(13, 5))

    # Left: binary mask slice
    ax4a = axes4[0]
    im = ax4a.imshow(xy_slice.T, origin='lower', cmap='binary',
                     extent=[coords.min(), coords.max(),
                             coords.min(), coords.max()],
                     vmin=0, vmax=1)
    # Overlay rmin/rmax circles
    for r_ref, col, lbl in [(rmin, 'red', f'rmin={rmin:.0f}'), (rmax, 'limegreen', f'rmax={rmax:.0f}')]:
        if r_ref is not None:
            circle = plt.Circle((0, 0), r_ref, color=col, fill=False, lw=1.5, ls='--', label=lbl)
            ax4a.add_patch(circle)
    ax4a.set_xlim(coords.min(), coords.max())
    ax4a.set_ylim(coords.min(), coords.max())
    ax4a.set_aspect('equal')
    ax4a.set_xlabel('X [Mpc/h]')
    ax4a.set_ylabel('Y [Mpc/h]')
    ax4a.set_title('XY slice at Z=0  (white=True)')
    ax4a.legend(fontsize=8, loc='upper right')

    # Right: r-colored slice — shows that True voxels fill the shell uniformly
    r_display = np.where(xy_slice, r_xy, np.nan)
    ax4b = axes4[1]
    im2 = ax4b.imshow(r_display.T, origin='lower', cmap='plasma',
                      extent=[coords.min(), coords.max(),
                              coords.min(), coords.max()])
    plt.colorbar(im2, ax=ax4b, label='Comoving distance [Mpc/h]')
    for r_ref, col, lbl in [(rmin, 'white', f'rmin={rmin:.0f}'), (rmax, 'cyan', f'rmax={rmax:.0f}')]:
        if r_ref is not None:
            circle = plt.Circle((0, 0), r_ref, color=col, fill=False, lw=1.5, ls='--', label=lbl)
            ax4b.add_patch(circle)
    ax4b.set_xlim(coords.min(), coords.max())
    ax4b.set_ylim(coords.min(), coords.max())
    ax4b.set_aspect('equal')
    ax4b.set_xlabel('X [Mpc/h]')
    ax4b.set_ylabel('Y [Mpc/h]')
    ax4b.set_title('XY slice colored by r  (confirms shell boundaries)')
    ax4b.legend(fontsize=8, loc='upper right')

    plt.suptitle('XY center slice of 3D mask  (observer at origin)', fontsize=12)
    plt.tight_layout()
    plt.show()
