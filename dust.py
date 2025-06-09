import os
import tempfile

import numpy as np
from matplotlib import pyplot as plt

import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from dustmaps.config import config
from dustmaps.sfd import SFDQuery

def gen_sfd_hp(nside=256, calc_cl=True):
    
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