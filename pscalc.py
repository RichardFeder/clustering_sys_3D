import tempfile
import numpy as np
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging

''' For things involving power spectrum estimation with pypower '''

# To activate logging
# setup_logging()


def compute_plk(posdata, weight_data, posrand, weight_rand, kedges, ells=(0,2,4), interlacing=2, nmesh=512, resampler='tsc', los='z', position_type='rdd', \
               plot=False, plot_wedges=False, mu_min=-1., mu_max=1., nwedge=6, shotnoise=None):

    
    result = CatalogFFTPower(data_positions1=posdata, data_weights1=weight_data,
                              randoms_positions1=posrand, randoms_weights1=weight_rand,\
                                 edges=(kedges, np.linspace(mu_min, mu_max, nwedge)), ells=ells, interlacing=interlacing,  # or whatever you estimate from your slice
                                     boxsize=None, boxcenter=None, nmesh=nmesh, resampler=resampler, wrap=False, 
                                         los=los, position_type=position_type, mpiroot=0, shotnoise=shotnoise)

    if plot_wedges:
        result.wedges.plot(show=True);
        

    return result