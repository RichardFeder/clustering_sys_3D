import numpy as np
import h5py
from astropy.cosmology import Planck18 as cosmo


def load_lightcone_positions_redshift(filepath, subsample_factor=100):
    """
    Loads Position and redshift data from a large HDF5 lightcone file,
    subsampling by the given factor.

    Parameters:
    -----------
    filepath : str
        Path to the lightcone HDF5 file.
    subsample_factor : int
        Factor by which to subsample the halos (e.g., 100 means keep 1 in 100).

    Returns:
    --------
    positions : (N, 3) numpy.ndarray
        Subsampled 3D positions of halos.
    redshifts : (N,) numpy.ndarray
        Subsampled redshifts of halos.
    """
    with h5py.File(filepath, 'r') as f:
        # Check that the required keys exist
        if 'Position' not in f or 'redshift' not in f:
            raise ValueError("HDF5 file missing required datasets: 'Position' or 'redshift'.")

        print('reading dataset shapes..')
        # Read dataset shapes
        N = f['Position'].shape[0]
        assert f['redshift'].shape[0] == N, "Mismatch between Position and redshift lengths."

        # Subsample indices
        print('subsampling by factor of ', subsample_factor)
        indices = np.arange(0, N, subsample_factor)

        # Load only selected entries
        print('Loading subsampled objects..')
        positions = f['Position'][indices]
        redshifts = f['redshift'][indices]

    return positions, redshifts


def load_lightcone_subset(filepath, n_sample=None, from_first_n=None):
    """
    Efficiently load a subsample of Position and redshift from a large halo lightcone.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 lightcone file.
    n_sample : int
        Total number of halos to load (after subsampling).
    from_first_n : int
        How many rows to read from the file before subsampling.

    Returns
    -------
    positions : (n_sample, 3) ndarray
        Subsampled 3D halo positions.
    redshifts : (n_sample,) ndarray
        Corresponding redshifts.
    """
    with h5py.File(filepath, 'r') as f:
        N_total = f['Position'].shape[0]

        if from_first_n is None:
            n_read = N_total
        else:
            n_read = min(from_first_n, N_total)

        print(f"Reading first {n_read:,} halos out of {N_total:,}")

        pos = f['Position'][:n_read]
        z = f['redshift'][:n_read]

    if n_sample is None:
        n_sample = n_read
    
    # Subsample in memory
    if n_sample >= n_read:
        # Return everything read
        return pos, z
    else:
        
        idx = np.random.choice(n_read, size=n_sample, replace=False)
        return pos[idx], z[idx]


HALFDOME_BASEDIR = '/global/cfs/cdirs/cmb/gsharing/halfdome/full_res/halos/'
N_HALFDOME_SIMS = 11  # lightcone_100, 102, ..., 120


def apply_completeness_subsampling(
    mockidx,
    completeness_3d,
    L_box,
    zmin=0.4,
    zmax=0.8,
    seed=None,
    n_sample=None,
    basedir=HALFDOME_BASEDIR,
):
    """
    Load a halfdome halo lightcone, apply a redshift cut, and sub-sample
    halos according to a voxelized 3D completeness map.

    For each halo the completeness value of its voxel is used as the
    acceptance probability: a uniform random draw in [0, 1) retains the
    halo if the draw is less than the completeness.

    Parameters
    ----------
    mockidx : int
        Simulation index in [0, 10].  Maps to lightcone_{100+2*mockidx}.hdf5.
    completeness_3d : np.ndarray, shape (N, N, N), float32
        Voxelized completeness map produced by
        ``plotting_fns.generate_3d_completeness_map``.  The grid covers
        [-L_box/2, L_box/2] Mpc/h along each axis with the observer at centre.
    L_box : float
        Side length of the completeness map box in Mpc/h.
    zmin, zmax : float
        Redshift selection window (inclusive).
    seed : int or None
        Random seed for reproducibility.
    n_sample : int or None
        If set, pre-subsample this many halos from the file before applying
        the redshift cut (passed directly to load_lightcone_subset).
    basedir : str
        Path to the directory containing the lightcone HDF5 files.

    Returns
    -------
    pos_obs : (M, 3) ndarray, float32
        Cartesian positions (Mpc, observer-centred) of the accepted halos.
    z_obs : (M,) ndarray, float32
        Redshifts of the accepted halos.
    completeness_vals : (M,) ndarray, float32
        Completeness value at each accepted halo's voxel.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Load halo catalog ------------------------------------------------
    mockidx_use = 100 + 2 * mockidx
    fpath = basedir + f'lightcone_{mockidx_use}.hdf5'
    print(f'Loading {fpath}')
    positions, redshifts = load_lightcone_subset(fpath, n_sample=n_sample)
    # positions: (N,3) float32 in Mpc, observer at origin

    # --- 2. Redshift selection -----------------------------------------------
    z_sel = (redshifts >= zmin) & (redshifts <= zmax)
    positions = positions[z_sel]
    redshifts = redshifts[z_sel]
    print(f'After z=[{zmin},{zmax}] cut: {len(redshifts):,} halos')

    # --- 3. Convert Mpc → Mpc/h for voxel lookup ----------------------------
    # Halfdome positions are in Mpc; completeness_3d grid is in Mpc/h.
    # pos_mpc_h = positions * cosmo.h  # (N,3) Mpc/h, observer-centred
    pos_mpc_h = positions  # (N,3) Mpc/h, observer-centred

    N = completeness_3d.shape[0]
    # Map [-L_box/2, L_box/2] → [0, N)
    vox = ((pos_mpc_h + L_box / 2.0) / L_box * N).astype(int)
    # Clip to valid range (handles rare halos just outside the box boundary)
    vox = np.clip(vox, 0, N - 1)

    # --- 4. Look up completeness per halo ------------------------------------
    c_vals = completeness_3d[vox[:, 0], vox[:, 1], vox[:, 2]]

    # --- 5. Acceptance / rejection sampling ---------------------------------
    draw = rng.uniform(0.0, 1.0, size=len(c_vals)).astype(np.float32)
    keep = draw < c_vals

    pos_obs = positions[keep]
    z_obs = redshifts[keep]
    completeness_vals = c_vals[keep]

    print(f'After completeness sub-sampling: {keep.sum():,} / {len(keep):,} halos '
          f'({100 * keep.mean():.1f}% kept)')

    return pos_obs, z_obs, completeness_vals
