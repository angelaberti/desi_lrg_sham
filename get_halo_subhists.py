import os
import sys
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack

import itertools as it

#from halocat_history import load_trees, get_subhist_from_tree


j = int(sys.argv[1])

quiet = False


snap_dir = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"

path_to_z0p0_histories = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"

path_to_lookup_tables  = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/tree_lookup_tables"

path_to_treefiles = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/TREES"

history_save_dir  = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/HISTORIES"


snaps     = ascii.read(f"{snap_dir}/bolshoip_snaps.txt")
snap_nums = (148,145,141,138,134,129,125,122,118,115,112)

hist_idx  = [178-i for i in snap_nums]

z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]


history_chunks = [f"history_z0p0_trees{j}{k}X.npy" for k in range(5)]

for chunkname in history_chunks:
    if not quiet:
        print(f"Halo history chunk: {chunkname}")
    chunk = Table(np.load(f"{path_to_z0p0_histories}/{chunkname}"))

    for treefilename in np.unique(chunk["treefile"]):
        this_subchunk = chunk[chunk["treefile"]==treefilename]
        N_halos = len(this_subchunk)

        if not quiet:
            print(f"  Processing {N_halos} halos with treefile {treefilename}...")

        for scale in a_snaps:
            this_scale = scale
            hist_len   = snaps["snapnum"][snaps["scale"]==this_scale].data[0]
            this_hist_idx = hist_idx[np.where(a_snaps==this_scale)[0][0]]

            lookup_table_name = "a" + str(scale).replace(".","p") + ".npy"
            if not quiet:
                print(f"    Loading halo IDs at a={this_scale}...")
            lookup_table = Table(np.load(f"{path_to_lookup_tables}/{lookup_table_name}"))

            lookup_table_this_treefile = lookup_table[lookup_table["treefilename"]==treefilename]

            if not quiet:
                print(f"    Checking for placeholder columns for histories...")
            for cname in ("mvir_hist", "rvir_hist", "rs_hist"):
                if cname not in lookup_table_this_treefile.colnames:
                    lookup_table_this_treefile.add_column(Column([np.zeros(hist_len) for i in range(len(lookup_table_this_treefile))], name=cname))
            if "id_hist" not in lookup_table_this_treefile.colnames:
                lookup_table_this_treefile.add_column(Column([np.zeros(hist_len) for i in range(len(lookup_table_this_treefile))], name="id_hist", dtype=int))

            for i in range(N_halos):
                this_z0p0_halo_id = this_subchunk["id"][i]
                if i%10000==0 and i>0 and not quiet:
                    print(f"      {i} / {N_halos}")
                this_history = this_subchunk[i]
                
                mvir_hist = this_history["mvir_hist"][this_hist_idx:]
                rvir_hist = this_history["rvir_hist"][this_hist_idx:]
                rs_hist   = this_history["rs_hist"][this_hist_idx:]
                id_hist   = this_history["id_hist"][this_hist_idx:]

                lookup_idx = np.where(lookup_table_this_treefile["halo_id"]==id_hist[0])[0]
                
                lookup_table_this_treefile["mvir_hist"][lookup_idx] = mvir_hist
                lookup_table_this_treefile["rvir_hist"][lookup_idx] = rvir_hist
                lookup_table_this_treefile["rs_hist"][lookup_idx]   = rs_hist
                lookup_table_this_treefile["id_hist"][lookup_idx]   = id_hist

            lookup_table_this_treefile.remove_column("scale")
        
            outname = f"{history_save_dir}/a" + str(this_scale).replace(".","p") + "_trees" + "{}{}{}".format(*treefilename.split("_")[1:])[:3] + ".npy"
            if not quiet:
                print(f"    Saving as {outname}...\n")
            np.save(outname, lookup_table_this_treefile)
