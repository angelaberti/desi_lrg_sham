import os, sys
from pathlib import Path

import numpy as np
import itertools as it

from params import BASEDIR, DATADIR, SIMDIR, MOCKDIR, H0, Om0
from params import get_zsnap_data #, get_boxsize, get_abs_mag_lim, get_sham_var_bins

from utils import *

from astropy.table import Table, Column, hstack, vstack, join
from astropy.io import fits, ascii

from halocat_history import get_subhist_from_tree

# chunk_idx = int(sys.argv[1])
# start_idx = int(sys.argv[2])
# end_idx   = int(sys.argv[3])

# assert((chunk_idx >= 0) & (chunk_idx <= 9))
# assert((start_idx >= 0) & (start_idx <= 99))
# assert((end_idx > 0) & (end_idx <= 99) & (end_idx > start_idx))

# tree_num_list = np.array([ "{}_{}_{}".format(chunk_idx,*i) for i in list(it.product("0123456789", repeat=2)) ])
quiet   = False
sim_tag = "mdpl2"

zsnaps, snap_nums, snaps = get_zsnap_data( sim_tag )

hist_idx = [len(snaps)-i for i in snap_nums]

z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]

#-- iterate over tree numbers (j_0_0 to j_9_9)
# if start_idx >= int(0.5*len(tree_num_list)):
#     iter_over = tree_num_list[::-1][len(tree_num_list)-(start_idx+1):]
# else:
#     iter_over = tree_num_list[start_idx:]    
iter_over = ["5_1_0"]
for i,tree_num in enumerate(iter_over):
    #-- check whether histories from all scales for this tree number already exist
    already_exists = True
    for scale in a_snaps:
        save_as = f"{SIMDIR}/{sim_tag}/HISTORIES/a" + str(scale).replace(".","p") + f"_tree_{tree_num}.npy"
        if not os.path.exists( save_as ):
            already_exists = False
            break

    if already_exists and not quiet:
        print(f"\nHistories from all scale factors for this treefile found; skipping tree_{tree_num}...")

    else:
        #-- load relevant full history
        hist_fname_this_tree = f"{SIMDIR}/{sim_tag}/HISTORIES/full/tree_{tree_num}.npy"
        if not quiet:
            print(f"\nLoading {hist_fname_this_tree}...")
        hist_full_this_tree = Table( np.load(hist_fname_this_tree) )
        if not quiet:
            print(f"Loading treefile {tree_num}...")
        this_treefile = np.load(f"{SIMDIR}/{sim_tag}/TREES/tree_{tree_num}.npy", allow_pickle=True).item()

        #-- iterate over scales
        for this_scale in a_snaps:
            save_as = f"{SIMDIR}/{sim_tag}/HISTORIES/a" + str(this_scale).replace(".","p") + f"_tree_{tree_num}.npy"

            if os.path.exists( save_as ):
                if not quiet:
                    fname = save_as.split("/")[-1]
                    print(f"{fname} found; skipping this scale...")
            else:
                hist_len = snaps["snapnum"][snaps["scale"]==this_scale].data[0]
                # print(f"\nHistory length: {hist_len}")
        
                this_hist_idx = hist_idx[np.where(a_snaps==this_scale)[0][0]]
                # print(f"\nthis_hist_idx = {this_hist_idx}")
    
                #-- load relevant lookup table for this tree number and scale
                this_lut_fname = f"{SIMDIR}/{sim_tag}/tree_lookup_tables/a" + str(this_scale).replace(".","p") + f"_tree_{tree_num}.npy"
                # if not quiet:
                #     print(f"\nLoading {this_lut_fname}...")
                this_lut = Table(np.load( this_lut_fname ))
                
                #-- number of (sub)halos in this lookup table
                N_halos = len(this_lut)
                if not quiet:
                    print(f"\nGetting histories for {N_halos} a={this_scale} {sim_tag} (sub)halos with tree {tree_num}...")
    
                #-- placeholder columns for histories
                for cname in ("mvir_hist", "rvir_hist", "rs_hist"):
                    if cname not in this_lut.colnames:
                        this_lut.add_column( Column([ np.zeros(hist_len+1) for h in range(N_halos) ], name=cname) )
                if "id_hist" not in this_lut.colnames:
                    this_lut.add_column( Column([ np.zeros(hist_len+1) for h in range(N_halos) ], name="id_hist", dtype=int) )
            
                #-- iterate over tree IDs
                N_tree_ids = len(np.unique(this_lut["tree_id"]))
                for t,this_tree_id in enumerate( np.unique(this_lut["tree_id"]) ):
                    these_subhalo_ids = this_lut["halo_id"][this_lut["tree_id"]==this_tree_id]
                    # if (t+1)%(int(1e4))==0 and not quiet:
                    #     print(f"{t+1} / {N_tree_ids}\tTree ID: {this_tree_id}")
                    #     print(f"Subhalos with this tree: {len(these_subhalo_ids)}")
                        
                    this_history = hist_full_this_tree[np.where(hist_full_this_tree["halo_id"]==this_tree_id)[0][0]]
                    
                    this_mvir_hist = this_history["mvir_hist"]
                    this_rvir_hist = this_history["rvir_hist"]
                    this_rs_hist   = this_history["rs_hist"]
                    this_id_hist   = this_history["id_hist"]
    
                    #-- iterate over (sub)halos with this tree ID
                    for s,this_subhalo_id in enumerate(these_subhalo_ids):
                        idx = np.where(this_lut["halo_id"]==this_subhalo_id)[0][0]
                        
                        # if (s+1)%(int(1e4))==0 and not quiet:
                        #     print(f"  (sub)halo {s+1} / {len(these_subhalo_ids)}")
                        #     print(f"  subhalo_id: {subhalo_id}; lookup table index = {idx}")
                            
                        if this_subhalo_id in this_id_hist:
                            # if (s+1)%(int(1e3))==0 and not quiet:
                            #     print(f"  ...(sub)halo ID {subhalo_id} found in id_hist")
                            this_mvir_hist = this_mvir_hist[this_hist_idx-1:]
                            this_rvir_hist = this_rvir_hist[this_hist_idx-1:]
                            this_rs_hist   = this_rs_hist[this_hist_idx-1:]
                            this_id_hist   = this_id_hist[this_hist_idx-1:]
        
                        else:
                            # if (s+1)%(int(1e3))==0 and not quiet:
                            #     print(f"  ...(sub)halo ID {subhalo_id} not found in id_hist; getting history from tree {this_tree_id}")
                            this_tree = this_treefile[this_tree_id]
                            this_mvir_hist, this_rvir_hist, this_rs_hist, this_id_hist = get_subhist_from_tree(this_subhalo_id, this_tree, quiet=quiet, N_steps=hist_len+1)
    
                        #-- add history information for this (sub)halo to lookup table for this scale
                        this_lut["mvir_hist"][idx] = this_mvir_hist
                        this_lut["rvir_hist"][idx] = this_rvir_hist
                        this_lut["rs_hist"][idx]   = this_rs_hist
                        this_lut["id_hist"][idx]   = this_id_hist
    
                #-- save lookup table for this tree number and this scale (now with added history information)
                if not quiet:
                    print(f"Saving {save_as}...")
                np.save(save_as, this_lut)
