import os, sys
import numpy as np
from astropy.io import ascii



BASEDIRS = ["/uufs/astro.utah.edu/common/home/u6032461/DESI_LRG_SHAM",
            "/Users/aberti/Desktop/research/desi_lrg_sham",
            ]

DATADIRS = ["/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/dr9",
            "/Users/aberti/Desktop/research/desi_lrg_sham/data",
           ]

SIMDIRS  = ["/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/sims",
            "/Users/aberti/Desktop/research/sims",
           ]

MOCKDIRS = ["/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/mocks",
            "/Users/aberti/Desktop/research/desi_lrg_sham/mocks",
           ]


b = BASEDIRS[0]
i = 1
while not os.path.exists(b):
    b = BASEDIRS[i]
    i += 1    
BASEDIR = b

d = DATADIRS[0]
i = 1
while not os.path.exists(d):
    d = DATADIRS[i]
    i += 1
DATADIR = d

s = SIMDIRS[0]
i = 1
while not os.path.exists(s):
    s = SIMDIRS[i]
    i += 1
SIMDIR = s

m = MOCKDIRS[0]
i = 1
while not os.path.exists(m):
    m = MOCKDIRS[i]
    i += 1
MOCKDIR = m

#-- clustering rp bin parameters
nbins   = 19
rp_min  = 0.0333
rp_max  = 64.4098
rp_bins = np.logspace( np.log10(rp_min), np.log10(rp_max), nbins + 1 )
rp_mids = np.array([ 10**np.mean(np.log10(rp_bins)[i:i+2]) for i in np.arange(len(rp_bins)-1) ])

bin_file          = f"{BASEDIR}/clustering/rp_bins.txt"
bin_file_comoving = f"{BASEDIR}/clustering/rp_bins_comoving.txt"

cosmo_params = {"mdpl2":(67.77,0.307115)}

H0, Om0 = cosmo_params["mdpl2"]


def get_abs_mag_lim(zmin, band):
    if band=="Mr":
        abs_mag_limits = {0.3:-20.40, 0.4:-20.80, 0.5:-20.80, 0.6:-21.00}
    elif band=="Mz":
        abs_mag_limits = {0.3:None, 0.4:-21.60, 0.5:-21.60, 0.6:-21.85}
    elif band=="MW1":
        abs_mag_limits = {0.3:-22.25, 0.4:-22.25, 0.5:-22.85, 0.6:-23.15}
    else:
        raise Exception()
    assert(zmin in abs_mag_limits.keys())
    return abs_mag_limits[zmin]



def get_abs_mag_bins_clust(zmin, band, nbins=None):
    if nbins != None:
        assert(type(nbins)==int)
        if nbins < 1:
            raise Exception("nbins must be >= 1 if not 'None'")
    if band=="Mr":
        bins = np.round(np.concatenate([np.arange(-28.80,-22.41,-0.2),[-30.0]])[::-1],2)
    elif band=="Mz":
        bins = np.round(np.concatenate([np.arange(-21.60,-22.61,-0.25),[-30.0]])[::-1],2)
    elif band=="MW1":
        bins = np.round(np.concatenate([np.arange(-21.95,-24.66,-0.3),[-30.0]])[::-1],2)
    else:
        raise Exception()

    abs_mag_lim = get_abs_mag_lim(zmin, band)
    assert(abs_mag_lim != None)

    bins = bins[np.where(bins <= abs_mag_lim)]
    if type(nbins)==int:
        bins = np.concatenate([[bins[0]],bins[-nbins:]])
    return bins



def get_zsnap_data(sim_tag):
    """returns (zsnaps, snap_nums, snaps) for given simulation"""
    snaps = ascii.read(f"/{SIMDIR}/{sim_tag}/snaps.txt")
    if sim_tag=="bolshoip":
        zsnaps = [0.04521, 0.15524, 0.24244, 0.33028, 0.43152, 0.54338, 0.63265, 0.73292, 0.82922, 0.93682]
    elif sim_tag=="mdpl2":
        zsnaps = [0.04526, 0.14220, 0.24797, 0.33369, 0.42531, 0.52323, 0.62813, 0.74004, 0.81884, 0.94363]
    else:
        raise Exception(f"Simulation \'{sim_tag}\'' not implememnted!")
    snap_nums = [snaps["snapnum"][snaps["redshift"]==z][0] for z in zsnaps]

    return zsnaps, snap_nums, snaps



def get_boxsize(sim_tag):
    """returns boxsize (Mpc/h) for given simulation"""
    if sim_tag=="bolshoip":
        b = 250.
    elif sim_tag=="mdpl2":
        b = 1000.
    else:
        raise Exception(f"Simulation \'{sim_tag}\'' not implememnted!")
    return b



def get_sham_var_bins(sham_tag):
    """returns bins for (log of) SHAM mass proxy variable"""
    s = sham_tag.lower()
    if (s=="vpeak") | (s=="vmax") | (s=="vmax\@mpeak"):
        bins = np.arange(1,3.4,0.02)
    elif (s=="mvir") | (s=="mpeak"):
        bins = np.arange(9.5,14.6,0.1)
    else:
        raise Exception("Valid SHAM variables are \'Vpeak\', \'Vmax\', \'Vmax\@Mpeak\', \'Mpeak\', \'Mvir\'")
    cens = [ np.mean(bins[i:i+2]) for i in range(len(bins)-1) ]
    
    return bins, cens

