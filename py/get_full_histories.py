import os, sys
from pathlib import Path

import numpy as np
import itertools as it

from params import BASEDIR, DATADIR, SIMDIR, MOCKDIR, H0, Om0
from params import get_zsnap_data #, get_boxsize, get_abs_mag_lim, get_sham_var_bins

from utils import *

from astropy.table import Table, Column, hstack, vstack, join
from astropy.io import fits, ascii

from halocat_history import get_hist_from_tree #, get_subhist_from_tree

chunk_idx = int(sys.argv[1])

quiet   = False
sim_tag = "mdpl2"

lookup_table_z0p0 = Table(np.load(f"{SIMDIR}/{sim_tag}/locations.npy"))

zsnaps, snap_nums, snaps = get_zsnap_data( sim_tag )

hist_len = len(snaps)

z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]

if chunk_idx==0:
    mask = lookup_table_z0p0["tree"] < 100
elif chunk_idx==9:
    mask = lookup_table_z0p0["tree"] >= 900
else:
    mask = ( lookup_table_z0p0["tree"].data >= (100*chunk_idx) ) & ( lookup_table_z0p0["tree"].data < 100*(chunk_idx+1) )

if not quiet:
    print(f"\nProcessing z=0 {sim_tag} (sub)halos with trees {chunk_idx}XX...")
this_chunk = lookup_table_z0p0[mask]

#-- get and save full (sub)halo histories for all halo IDs with given tree
for tree_num in np.unique(this_chunk["tree"]): #[::-1]:
    if chunk_idx==0:
        if tree_num < 10:
            tree_tag = f"tree_0_0_{tree_num}.npy"
        else:
            tree_tag = "tree_0_{}_{}.npy".format(*[ dgt for dgt in str(tree_num) ])
    else:
        tree_tag = "tree_{}_{}_{}.npy".format(*[ dgt for dgt in str(tree_num) ])

    save_path = f"{SIMDIR}/{sim_tag}/HISTORIES/full"
    Path( save_path ).mkdir(parents=True, exist_ok=True)
    save_as = f"{save_path}/{tree_tag}"

    if os.path.exists( save_as ):
        if not quiet:
            print(f"{save_as} found; skipping...")
        else:
            continue
    else:
        this_subchunk  = this_chunk[this_chunk["tree"]==tree_num]
        these_halo_ids = this_subchunk["halo_id"]
        N_halos = len(these_halo_ids)
        if not quiet:
            print(f"\nTree {tree_num}: {N_halos} (sub)halo IDs")

        #-- load treefile
        tf = f"{SIMDIR}/{sim_tag}/TREES/{tree_tag}" # + "tree_{}_{}_{}.npy".format(*[i for i in str(tree_num)])
        if not quiet:
            print(f"Loading {tf}...")
        this_treefile = np.load( tf, allow_pickle=True ).item()

        #-- placeholder columns for histories
        if not quiet:
            print("Adding placeholder columns for history information...")
        for cname in ("mvir_hist", "rvir_hist", "rs_hist"):
            if cname not in this_subchunk.colnames:
                if not quiet:
                    print(cname)
                this_subchunk.add_column( Column([ np.zeros(hist_len) for i in range(len(this_subchunk)) ], name=cname) )
        if "id_hist" not in this_subchunk.colnames:
            if not quiet:
                print("id_hist")
            this_subchunk.add_column( Column([ np.zeros(hist_len) for i in range(len(this_subchunk)) ], name="id_hist", dtype=int) )
    
        if not quiet:
            print("\nGetting full histories for individual (sub)halos...")
        for i,this_halo_id in enumerate(these_halo_ids):
            if ((i+1)%(int(1e4)))==0 and not quiet:
                print(f"{i+1} / {N_halos}")
    
            #-- get history information for this (sub)halo
            mvir_hist, rvir_hist, rs_hist, id_hist = get_hist_from_tree(this_halo_id, this_treefile, quiet=quiet, sim_tag=sim_tag)
    
            mask = these_halo_ids==this_halo_id
            this_subchunk["mvir_hist"][mask] = mvir_hist
            this_subchunk["rvir_hist"][mask] = rvir_hist
            this_subchunk["rs_hist"][mask]   = rs_hist
            this_subchunk["id_hist"][mask]   = id_hist
        
        if not quiet:
            print(f"Saving {save_as}...")
        np.save(save_as, this_subchunk)
        
