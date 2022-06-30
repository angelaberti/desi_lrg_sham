import os, sys
from pathlib import Path

import numpy as np

from params import SIMDIR, get_zsnap_data

from utils import *

from astropy.table import Table, Column, hstack, vstack, join
from astropy.io import fits, ascii
# from astropy.cosmology import FlatLambdaCDM
# cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
# from astropy.constants import c as c_speed
# from astropy import units as u


zmin        = float(sys.argv[1])
zrange      = np.round([zmin,zmin+0.1],2)
zmin, zmax  = zrange
quiet       = True if (("-q" in sys.argv) | ("--quiet" in sys.argv)) else False

sim_tag      = "mdpl2"
sham_tag     = "vpeak" #-- vmax, vmax\@mpeak
sham_tag_min = 125  #-- km/s


zsnaps, snap_nums, snaps = get_zsnap_data( sim_tag )
zsnap     = np.array(zsnaps)[ np.round(zsnaps,1)==zmin ][0]
# boxsize   = float(get_boxsize( sim_tag ))
# vol_sim   = boxsize**3
zsnap_tag = get_zsnap_tag(zsnap)

if not quiet:
    print(f"zsnap = {zsnap} ({sim_tag})")

a     = snaps["scale"][snaps["redshift"]==zsnap][0]
scale = f"{a:.5f}"
scale = scale.replace(".","p")
this_sim_dir = f"{SIMDIR}/{sim_tag}/CATALOGS/{sham_tag}min{int(sham_tag_min)}"
test_fname   = f"{this_sim_dir}/a{scale[:-1]}.npy"

if os.path.exists( test_fname ):
    if not quiet:
        print(f"{test_fname} exists; skipping...")
else:
    halocat = Table(np.load(f"{SIMDIR}/{sim_tag}/CATALOGS/hlist_{scale}.npy"))
    Path( this_sim_dir ).mkdir(parents=True, exist_ok=True)
    
    halocat = halocat[halocat[sham_tag] >= sham_tag_min]

    print(f"Saving in {this_sim_dir}...")
    np.save( f"{test_fname[:-4]}.npy", halocat )
    
    
 