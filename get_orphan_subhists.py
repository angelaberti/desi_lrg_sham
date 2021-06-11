import os
import sys
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack

import itertools as it

from halocat_history import load_trees, get_subhist_from_tree


j = int(sys.argv[1])

quiet = False


snap_dir = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"

#path_to_z0p0_histories = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"

#path_to_lookup_tables  = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/tree_lookup_tables"

path_to_treefiles = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/TREES"

history_dir  = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/HISTORIES"


snaps     = ascii.read(f"{snap_dir}/bolshoip_snaps.txt")
snap_nums = (148,145,141,138,134,129,125,122,118,115,112)

#hist_idx  = [178-i for i in snap_nums]

#z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]

treefilelist = ["tree_{}_{}_{}.dat".format(j,*i) for i in list(it.product("01234", repeat=2))]

for treefilename in treefilelist:
    if not quiet:
        print(f"Loading {treefilename}...")
    this_treefile = load_trees(f"{path_to_treefiles}/{treefilename}", quiet=True)
    
    for scale in a_snaps:
        this_scale = scale
        hist_len   = snaps["snapnum"][snaps["scale"]==this_scale].data[0]
        this_hist_idx = hist_idx[np.where(a_snaps==this_scale)[0][0]]

        this_chunkname = "a" + str(this_scale).replace(".","p") + "_trees" + "{}{}{}".format(*treefilename.split("_")[1:])[:3] + ".npy"
        if not quiet:
            print(f"  Loading chunk {this_chunkname}...")
        this_chunk = Table(np.load(f"{history_dir}/{this_chunkname}"))
        
        mask = this_chunk["id_hist"][:,0]==0
        N_halos = len(this_chunk[mask])
        
        if not quiet:
            print(f"    Histories needed for {N_halos} of {len(this_chunk)} halos...")
        
        for i in range(N_halos):
            this_halo_id = this_chunk[mask]["halo_id"][i]
            this_tree_id = this_chunk[mask]["tree_id"][i]
    
            mvir_hist, rvir_hist, rs_hist, id_hist = get_subhist_from_tree(this_halo_id, this_treefile, this_tree_id, N_steps=hist_len)
            if i%1000==0 and i>0 and not quiet:
                print(f"    {i} / {N_halos}")
                print(f"\t{id_hist}")
                
            this_chunk["mvir_hist"][this_chunk["halo_id"]==this_halo_id] = mvir_hist
            this_chunk["rvir_hist"][this_chunk["halo_id"]==this_halo_id] = rvir_hist
            this_chunk["rs_hist"][this_chunk["halo_id"]==this_halo_id]   = rs_hist
            this_chunk["id_hist"][this_chunk["halo_id"]==this_halo_id]   = id_hist

        if not quiet:
            print(f"  Saving chunk...\n")
        np.save(f"{history_dir}/{this_chunkname}", this_chunk)
