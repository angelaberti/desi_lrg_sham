import os, sys
import numpy as np
from pathlib import Path

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join

import itertools as it

from params import BASEDIR, DATADIR, SIMDIR, MOCKDIR, H0, Om0, get_zsnap_data

from halocat_history import read_halocat, load_trees #, get_hist_from_tree


sim_tag   = sys.argv[1]
chunk_idx = int(sys.argv[2])
# convert_only = True if "--convert-only" in sys.argv else False
# if convert_only:
#     print("\nConverting treefiles from '.dat' to '.npy' format...")
# if ("--convert_only" in sys.argv):
#     raise Exception("Did you mean '--convert-only'?")
    
assert((sim_tag=="mdpl2") | (sim_tag=="bolshoip"))
assert((chunk_idx >= 0) & (chunk_idx <= 9))

quiet = False

zsnaps, snap_nums, snaps = get_zsnap_data(sim_tag)

TREEDIR = f"{SIMDIR}/{sim_tag}/TREES"

lookup_table_dir = f"{SIMDIR}/{sim_tag}/tree_lookup_tables"
Path( lookup_table_dir ).mkdir(parents=True, exist_ok=True)

z_snaps = [snaps["redshift"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]

#tf_list_chunks = np.array([["tree_{}_{}_{}".format(j,*i) for i in list(it.product("0123456789", repeat=2))] for j in range(10)])
tf_list = np.array([ "tree_{}_{}_{}".format(chunk_idx,*i) for i in list(it.product("0123456789", repeat=2)) ])
# if not quiet:
#     print(f"\nLet's do this...\n{tf_list}\n")
#this_chunk = tf_list_chunks[chunk_idx]

for tf in tf_list[::-1]:
    already_exists = True
    #-- check if there are *any* relevant scale factors for which a lookup table for this treefile doesn't already exist
    for scale in a_snaps:
        save_as = f"{lookup_table_dir}/a" + str(scale).replace(".","p") + f"_{tf}.npy"
        if not os.path.exists( save_as ):
            already_exists = False
            break

    if already_exists and not quiet:
        print(f"Lookup tables for all scale factors for this treefile found; skipping {tf}...")

    else:
        #-- load each treefile
        tfname_npy = f"{tf}.npy"
        tfpath_npy = f"{TREEDIR}/{tfname_npy}"
    
        #-- load npy version if it exists    
        if os.path.exists( tfpath_npy ):
            # if convert_only:
            #     print(f"{tfname_npy} found; skipping...")
            # else:
            if not quiet:
                print(f"Loading treefile {tfname_npy}...")
            this_treefile = np.load( tfpath_npy, allow_pickle=True ).item()
        #-- otherwise parse ascii file and save npy version
        else:
            if not quiet:
                print(f"Loading treefile {tf}.dat...")
            this_treefile = load_trees(f"{TREEDIR}/{tf}.dat", quiet=True)
            if not quiet:
                print(f"Saving as {tfpath_npy}")
            np.save( tfpath_npy, this_treefile, allow_pickle=True )
    
        #-- create empty lookup table dictionary
        lookup_table_by_snap = {}
        for i,snap in enumerate(snap_nums):
            lookup_table_by_snap[a_snaps[i]] = []
    
        # if not convert_only:
        #-- list of halo_ids in this treefile
        this_treefile_halo_ids = list(this_treefile.keys())
        N_halos = len(this_treefile_halo_ids)
        if not quiet:
            print(f"{N_halos} (sub)halos...")
            
        #-- iterate over list of halo_ids    
        for i,halo_id in enumerate(this_treefile_halo_ids):
            if ( (i+1)%(int(1e4))==0 ) and not quiet:
                print(f"{i+1} / {N_halos}")
            this_tree = this_treefile[halo_id] #-- tree table for this halo_id
    
            #-- add (optional) treefile and relevant z = 0 halo_id to this tree table
            if "treefilename" in this_tree.colnames:
                this_tree.remove_column("treefilename")
            # this_tree.add_column(Column([tf for j in range(len(this_tree))], name="treefilename"))
            if "tree_id" in this_tree.colnames:
                this_tree.remove_column("tree_id")
            this_tree.add_column(Column(halo_id*np.ones(len(this_tree), dtype=int), name="tree_id"))
    
            #-- add updated tree table to by scale facotr (snap) to dictionary
            for i,snap in enumerate(snap_nums):
                #print(f"{i} / {len(snap_nums)}")
                lookup_table_by_snap[a_snaps[i]].append(this_tree[this_tree["Snap_num(31)"]==snap])
      
        # if not convert_only:
        for scale in list(lookup_table_by_snap.keys()):
            lookup_table_by_snap[scale] = vstack( lookup_table_by_snap[scale] )
            for col in np.concatenate([ ["#scale(0)","Snap_num(31)"], lookup_table_by_snap[scale].colnames[2:-2] ]):
                this_table = lookup_table_by_snap[scale]
                this_table.remove_column(col)
            # if "#scale(0)" in this_table.colnames:
                # this_table.rename_column("#scale(0)")
                # this_table.rename_column("#scale(0)", "scale")
            if "id(1)" in this_table.colnames:
                this_table.rename_column("id(1)", "halo_id")
        
            save_as = f"{lookup_table_dir}/a" + str(scale).replace(".","p") + f"_{tf}.npy"
            if not quiet:
                print(f"Saving scale a = {scale}...")
            np.save(save_as, this_table)
    
