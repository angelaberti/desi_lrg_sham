import os
import sys
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join

import itertools as it

from halocat_history import read_halocat, load_trees #, get_hist_from_tree

#path_to_treefiles = "/Users/aberti/Downloads"
path_to_treefiles = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/TREES"

#lt_save_dir       = "/Users/aberti/Desktop/Research/bolshoip_lookup_tables"
lt_save_dir       = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/tree_lookup_tables"

#snap_dir          = "/Users/aberti/Desktop/Research"
snap_dir          = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"


snaps     = ascii.read(f"{snap_dir}/bolshoip_snaps.txt")
snap_nums = (147,144,140,137,133,128,124,121,117,114,111)

z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]


treefilelist = [["tree_{}_{}_{}.dat".format(j,*i) for i in list(it.product("01234", repeat=2))] for j in range(5)]

for j,chunk in enumerate(treefilelist):
    lookup_tables = {}
    for i,snap in enumerate(snap_nums):
        lookup_tables[a_snaps[i]] = []

    # for tf,treefile in zip(("tree_0_4_2.dat","tree_2_3_0.dat"),(trees042,trees230)):
    for tf in chunk[:2]:
    #     this_treefile = treefile
        print(f"Loading treefile {tf}...")
        this_treefile = load_trees(f"{path_to_treefiles}/{tf}", quiet=True)

        this_treefile_halo_ids = list(this_treefile.keys())
        print(f"{len(this_treefile_halo_ids)}\t{tf}")
        
        for halo_id in this_treefile_halo_ids[:1000]:
            this_tree = this_treefile[halo_id]

            if "treefilename" in this_tree.colnames:
                this_tree.remove_column("treefilename")
            this_tree.add_column(Column([tf for j in range(len(this_tree))], name="treefilename"))
            if "tree_id" in this_tree.colnames:
                this_tree.remove_column("tree_id")
            this_tree.add_column(Column(halo_id*np.ones(len(this_tree), dtype=int), name="tree_id"))

            for i,snap in enumerate(snap_nums):
                #print(f"{i} / {len(snap_nums)}")
                lookup_tables[a_snaps[i]].append(this_tree[this_tree["Snap_num(31)"]==snap])
        
    for scale in list(lookup_tables.keys()):
        lookup_tables[scale] = vstack(lookup_tables[scale])
        for col in lookup_tables[scale].colnames[2:-2]: 
            this_table = lookup_tables[scale]
            this_table.remove_column(col)
        if "#scale(0)" in this_table.colnames:
            this_table.rename_column("#scale(0)", "scale")
        if "id(1)" in this_table.colnames:
            this_table.rename_column("id(1)", "halo_id")

        save_path = f"{lt_save_dir}/a" + str(scale).replace(".","p") + f"_trees{j}XX.npy"
        print(save_path)
        np.save(save_path, this_table)
