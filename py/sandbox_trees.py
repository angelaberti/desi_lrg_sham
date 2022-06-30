import os, sys
from pathlib import Path

import numpy as np
# from scipy import interpolate
import itertools as it

# import matplotlib.pyplot as plt
# import matplotlib.backends.backend_pdf

from params import BASEDIR, DATADIR, SIMDIR, MOCKDIR, H0, Om0
from params import get_zsnap_data, get_boxsize, get_abs_mag_lim, get_sham_var_bins
# from params import nbins, rp_min, rp_max, rp_bins, rp_mids, bin_file
# from params import get_abs_mag_bins_clust

from utils import *

from astropy.table import Table, Column, hstack, vstack, join
from astropy.io import fits, ascii
# from astropy.cosmology import FlatLambdaCDM
# cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
# from astropy.constants import c as c_speed
# from astropy import units as u

# from datetime import datetime

from halocat_history import get_subhist_from_tree #, read_halocat, get_hist_from_tree, load_trees


sim_tag     = "mdpl2"
sham_tag    = "vpeak"
pct_scatter = 60
LOS_err     = 45

d        = "north"
zrange   = (0.4,0.5)
band     = "MW1"
zmag_lim = 20.7

quiet    = False


#-- housekeeping
zmin, zmax = zrange
zmag_tag   = get_zmag_tag(zmag_lim)

zsnaps, snap_nums, snaps = get_zsnap_data( sim_tag )
this_zsnap    = np.array(zsnaps)[ np.round(zsnaps,1)==zmin ][0]
zsnap_tag     = get_zsnap_tag(this_zsnap)
this_snap_num = np.array(snap_nums)[ np.round(zsnaps,1)==zmin ][0]
hist_len      = len(snaps) - this_snap_num

abs_mag_lim = get_abs_mag_lim(zmin, band)
if band=="MW1":
    abs_mag_lim_tag = get_MW1_tag( abs_mag_lim )
elif band=="Mr":
    abs_mag_lim_tag = get_Mr_tag( abs_mag_lim )
else:
    raise Exception()

ss = f"0{pct_scatter}" if pct_scatter < 10 else str(pct_scatter)

# boxsize = get_boxsize( sim_tag )
# vol_sim = boxsize**3
"""
1. Convert treefiles from ascii to npy in batches of every N_trees^(2/3) treefiles [DONE]
    e.g. trees_0_1_2.dat => trees_0_1_2.npy (trees 0_0_0 to 9_9_9)
    $ "python gen_lookup_tables.py mdpl2 # &" (# = 0 to 9)

2a. Create lookup table for each scale and each treefile (can be combined with previous step) [IN PROGRESS]
    e.g. a0p6142_tree_0_1_2.npy (all scales in a_snaps; trees 0_0_0 to 9_9_9)

# 2b. (NOT NEEDED) Combine lookup tables for every N_trees^(2/3) treefiles (N_trees^(2/3) = 100 for mdpl2)
#     e.g. a0p6142_trees#XX.npy (all scales in a_snaps; # = 0 to 9)
#     $ "python combine_lookup_tables.py # &" (# = 0 to 9; sim_tag already set to mdpl2)

3. Get full histories (from z=0) for each z=0 (sub)halo in batches of every N_trees^(2/3) treefiles
    e.g. ...HISTORIES/full/tree_0_1_2.npy (trees 0_0_0 to 9_9_9)
    $ "python get_halocat_hists.py # &" (# = 0 to 9)

4. Get subhistory from each scale for each (sub)halo in snapshot at that scale
    e.g. ...HISTORIES/a0p6142_tree_0_1_2.npy (all scales in a_snaps; trees 0_0_0 to 9_9_9)
    $ "python get_halo_subhists.py # &" (# = 0 to 9)
"""

z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]

if not quiet:
    print(f"snap_num        = {this_snap_num}")
    print(f"hist_len        = {hist_len}")
    print(f"abs_mag_lim_tag = {abs_mag_lim_tag}")
    print(f"zsnap           = {this_zsnap} ({sim_tag})")

mock_name = f"{MOCKDIR}/{sim_tag}/{sham_tag}/{zsnap_tag}_{zmag_tag}_{abs_mag_lim_tag}_scatter{ss}pct.npy"
if not quiet:
    print(f"Loading {mock_name}...\n")
mock = Table(np.load( mock_name ))

#-- load haloid (z=0 snapshot only) => treefile map
locs = Table(np.load(f"{SIMDIR}/{sim_tag}/locations.npy"))

this_haloid = mock["id"].data[5]
print(this_haloid)

this_treeid = locs["tree_id"][locs["halo_id"]==this_haloid].data[0]
if this_treeid < 10:
    tf = f"tree_0_0_{this_treeid}.npy"
elif this_treeid < 100:
    tf = "tree_0_{}_{}.npy".format(*[ dgt for dgt in str(this_treeid) ])
else:
    tf = "tree_{}_{}_{}.npy".format(*[ dgt for dgt in str(this_treeid) ])
this_tree_dict_fname = f"{SIMDIR}/{sim_tag}/TREES/{tf}"

this_tree_dict = np.load( this_tree_dict_fname, allow_pickle=True )

this_tree = this_tree_dict[ this_haloid ]
mvir_hist, rvir_hist, rs_hist, id_hist = get_subhist_from_tree(this_haloid, this_tree, quiet=quiet, N_steps=hist_len)

print(mvir_hist, rvir_hist, rs_hist, id_hist)

    # for scale in a_snaps:
    #     this_scale = scale
    #     hist_len   = snaps["snapnum"][snaps["scale"]==this_scale].data[0]
    #     print(f"this_scale = {this_scale}")
    #     print(f"hist_len   = {hist_len}")
    #     this_hist_idx = hist_idx[np.where(a_snaps==this_scale)[0][0]]
    #     print(this_hist_idx)

    #     this_chunkname = "a" + str(this_scale).replace(".","p") + "_trees" + "{}{}{}".format(*treefilename.split("_")[1:])[:3] + ".npy"
    #     if not quiet:
    #         print(f"  Loading chunk {this_chunkname}...")
        # this_chunk = Table(np.load(f"{history_dir}/{this_chunkname}"))
        
        # mask = this_chunk["id_hist"][:,0]==0
        # N_halos = len(this_chunk[mask])
        
        # if not quiet:
        #     print(f"    Histories needed for {N_halos} of {len(this_chunk)} halos...")
        
        # for i in range(N_halos):
        #     this_halo_id = this_chunk[mask]["halo_id"][i]
        #     this_tree_id = this_chunk[mask]["tree_id"][i]
    
        #     mvir_hist, rvir_hist, rs_hist, id_hist = get_subhist_from_tree(this_halo_id, this_treefile, this_tree_id, N_steps=hist_len)
        #     if i%1000==0 and i>0 and not quiet:
        #         print(f"    {i} / {N_halos}")
        #         print(f"\t{id_hist}")
                
        #     this_chunk["mvir_hist"][this_chunk["halo_id"]==this_halo_id] = mvir_hist
        #     this_chunk["rvir_hist"][this_chunk["halo_id"]==this_halo_id] = rvir_hist
        #     this_chunk["rs_hist"][this_chunk["halo_id"]==this_halo_id]   = rs_hist
        #     this_chunk["id_hist"][this_chunk["halo_id"]==this_halo_id]   = id_hist

        # if not quiet:
        #     print(f"  Saving chunk...\n")
        # np.save(f"{history_dir}/{this_chunkname}", this_chunk)



