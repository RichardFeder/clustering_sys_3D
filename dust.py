import os
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

def resolve_dustmaps_data_dir(explicit_data_dir=None):
    """Resolve dustmaps data dir so SFDQuery can find sfd/SFD_dust_*.fits files."""
    candidates = []

    if explicit_data_dir is not None:
        candidates.append(Path(explicit_data_dir).expanduser())

    env_dir = os.environ.get('DUSTMAPS_DATA_DIR')
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    # User's known pscratch location for dustmaps files.
    candidates.append(Path('/pscratch/sd/r/rmfeder/desi_sys'))

    # Repository-local fallback (when running from repo root).
    candidates.append(Path.cwd())

    seen = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)

        sfd_dir = base / 'sfd'
        ngp = sfd_dir / 'SFD_dust_4096_ngp.fits'
        sgp = sfd_dir / 'SFD_dust_4096_sgp.fits'
        if ngp.exists() and sgp.exists():
            return str(base)

    candidate_str = '\n'.join(str(c) for c in seen)
    raise FileNotFoundError(
        'Could not locate SFD dust map files in any candidate dustmaps data directory.\n'
        f'Searched bases:\n{candidate_str}\n'
        'Expected files under each base: sfd/SFD_dust_4096_ngp.fits and sfd/SFD_dust_4096_sgp.fits\n'
        'Set DUSTMAPS_DATA_DIR to the directory above sfd/ if needed.'
    )

def gen_sfd_hp(nside=256, calc_cl=True, dustmaps_data_dir=None):
    try:
        from dustmaps.config import config
        from dustmaps.sfd import SFDQuery
    except ImportError as exc:
        raise ImportError(
            'dustmaps is required to query SFD maps. Install it in your active environment '
            "or run variants without dust contamination."
        ) from exc

    data_dir = resolve_dustmaps_data_dir(dustmaps_data_dir)
    config['data_dir'] = data_dir
    print(f'Using dustmaps data_dir: {data_dir}')
    
    # Set HEALPix resolution
    npix = hp.nside2npix(nside)
    
    # Get pixel centers in RA/Dec
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    # Query the SFD map
    sfd = SFDQuery()
    ebv_map = sfd(coords)

    if calc_cl:
        cl_sfd = hp.anafast(ebv_map)

        return ebv_map, cl_sfd

    return ebv_map