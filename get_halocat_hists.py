import os
import sys
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join

import itertools as it

from halocat_history import read_halocat, load_trees, get_hist_from_tree


j = int(sys.argv[1])

quiet = False

#snap_dir          = "/Users/aberti/Desktop/Research"
snap_dir          = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"

#path_to_locfile     = "/Users/aberti/Downloads"
path_to_locfile   = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"

#path_to_treefiles   = "/Users/aberti/Downloads"
path_to_treefiles = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip/TREES"

#history_save_dir  = "/Users/aberti/Downloads"
history_save_dir  = "/uufs/chpc.utah.edu/common/home/astro/dawson/aberti/bolshoip"

print(f"Loading treefile data...\n")

loc = Table(np.load(f"{path_to_locfile}/locations.npy"))

loc.rename_column("TreeRootID","id")
loc.rename_column("Filename","treefile")
loc.remove_columns(["FileID","Offset"])

## divide into chunks
z0p0_treefile_chunks = loc[[f"tree_{j}_" in x for x in loc["treefile"].data]]
z0p0_chunks = [z0p0_treefile_chunks[i] for i in [[f"{j}_{x}_" in y for y in z0p0_treefile_chunks["treefile"].data] for x in range(5)]]

print(f"History information saved in {history_save_dir}\n")

snaps = ascii.read(f"{snap_dir}/bolshoip_snaps.txt")
# snap_nums = (147,144,140,137,133,128,124,121,117,114,111)

hist_len = len(snaps)

# z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
# a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]

treefilelist = ["tree_{}_{}_{}.dat".format(j,*i) for i in list(it.product("01234", repeat=2))]
treefile_chunks = np.split(np.array(treefilelist), 5)
# print(treefile_chunks)

if not quiet:
    print(f"Processing z=0 (sub)halos with trees {j}XX...")

for k,z0p0_chunk,treefile_chunk in zip(range(len(z0p0_chunks)),z0p0_chunks,treefile_chunks):
    #print(k,z0p0_chunk[:10],treefile_chunk)
    if not quiet:
        print("  Adding placeholder columns for history information...")
    for cname in ("mvir_hist", "rvir_hist", "rs_hist"):
        if cname not in z0p0_chunk.colnames:
            z0p0_chunk.add_column(Column([np.zeros(hist_len) for i in range(len(z0p0_chunk))], name=cname))
    if "id_hist" not in z0p0_chunk.colnames:
        z0p0_chunk.add_column(Column([np.zeros(hist_len) for i in range(len(z0p0_chunk))], name="id_hist", dtype=int))

    for tf in treefile_chunk:
        if not quiet:
            print(f"    Loading treefile {tf}...")
        this_treefile = load_trees(f"{path_to_treefiles}/{tf}", quiet=True)

        ## iterate over all (sub)halo IDs with current treefile
        N_halos = len(z0p0_chunk[z0p0_chunk["treefile"]==tf])
        if not quiet:
            print(f"    z=0 (sub)halos with this treefile: {N_halos}")
        for i in range(N_halos):
            if i%10000==0 and not quiet:
                print(f"    {i} / {N_halos}")
            this_halo_id = z0p0_chunk[z0p0_chunk["treefile"]==tf]["id"][i]

            ## get history information for this (sub)halo
            mvir_hist, rvir_hist, rs_hist, id_hist = get_hist_from_tree(this_halo_id, this_treefile, quiet=quiet, N_steps=hist_len)

            z0p0_chunk["mvir_hist"][z0p0_chunk["id"]==this_halo_id] = mvir_hist
            z0p0_chunk["rvir_hist"][z0p0_chunk["id"]==this_halo_id] = rvir_hist
            z0p0_chunk["rs_hist"][z0p0_chunk["id"]==this_halo_id]   = rs_hist
            z0p0_chunk["id_hist"][z0p0_chunk["id"]==this_halo_id]   = id_hist

    save_path = f"{history_save_dir}/history_z0p0_trees{j}{k}X.npy"
    if not quiet:
        print(f"  Saving chunk {j}_{k}: {save_path}")
    np.save(save_path, z0p0_chunk)