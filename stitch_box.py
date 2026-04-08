import numpy as np

def stitch_boxes_randomized(initial_coords, box_size, rep_fac=3, randomize=True, seed=None):
    """
    Stitches a periodic cubic box into a larger volume with optional random flips and rotations.

    Args:
        initial_coords (np.ndarray): Shape (N, 3), galaxy positions in original box.
        box_size (float): Side length of the box (e.g., 1000 for 1 Gpc/h).
        rep_fac (int): Replication factor per axis (3 → 3x3x3).
        randomize (bool): Whether to randomly flip/rotate each replicate.
        seed (int or None): Seed for reproducibility.

    Returns:
        np.ndarray: Stitched and optionally randomized coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    all_coords = []

    for i in range(rep_fac):
        for j in range(rep_fac):
            for k in range(rep_fac):
                displacement = np.array([i, j, k]) * box_size
                coords = initial_coords.copy()

                if randomize:
                    # Apply random flips along axes
                    flips = np.random.choice([1, -1], size=3)
                    coords *= flips

                    # Optionally apply axis permutations
                    if np.random.rand() < 0.5:
                        coords = coords[:, [1, 0, 2]]  # Swap x and y
                    if np.random.rand() < 0.5:
                        coords = coords[:, [0, 2, 1]]  # Swap y and z

                # Displace to correct tile
                coords += displacement
                all_coords.append(coords)

    stitched_coords = np.concatenate(all_coords, axis=0)
    return stitched_coords

def stitch_boxes(initial_coords, box_size, rep_fac=3):
    """
    Stitches a periodic cubic box into a larger 3x3x3 volume.

    Args:
        initial_coords (np.ndarray): A NumPy array of shape (N, 3) containing
                                     the initial Cartesian coordinates (X, Y, Z)
                                     of the galaxies in the single box.
        box_size (float): The side length of the single cubic box (e.g., 1000 for 1 Gpc/h).

    Returns:
        np.ndarray: A NumPy array of shape (27*N, 3) containing the coordinates
                    of the galaxies in the stitched 3x3x3 volume.
    """
    if not isinstance(initial_coords, np.ndarray) or initial_coords.ndim != 2 or initial_coords.shape[1] != 3:
        raise ValueError("initial_coords must be a NumPy array of shape (N, 3).")

    # Create a list to hold the new sets of coordinates
    all_coords = []

    # Iterate through a 3x3x3 grid to place the boxes
    for i in range(rep_fac):
        for j in range(rep_fac):
            for k in range(rep_fac):
                # Calculate the displacement for the current box copy
                displacement = np.array([i * box_size, j * box_size, k * box_size])
                
                # Add the displacement to the initial coordinates
                new_coords = initial_coords + displacement
                all_coords.append(new_coords)

    # Concatenate all the coordinate sets into a single array
    stitched_coords = np.concatenate(all_coords, axis=0)
    
    return stitched_coords