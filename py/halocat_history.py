import os
import numpy as np
from pathlib import Path

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join

from params import SIMDIR


def read_halocat(halocat_filepath, colnums=[1,5,6,10,11,12,15,16,17,18,19,59,60,61,62,63,69,70,71,72,73,74], quiet=True, sham_tag_min=125., N_chunks="all", chunk_length=50000):
    """
    Parses raw ascii (.dat) Rockstar halo files and saves selected columns in npy/table format
    colnums = [1,5,6,10,11,12,15,16,17,18,19,59,60,61,62,63,69,70,71,72,73,74]
    """
    if ".npy" not in halocat_filepath:
        if N_chunks != "all":
            assert( (type(N_chunks)==int) & (N_chunks > 0) )
            if not quiet:
                print(f"Reading {N_chunks} X {chunk_length} of {halocat_filepath}...")
        else:
            assert(os.path.exists(halocat_filepath))
            if not quiet:
                print(f"Reading all of {halocat_filepath}...")
        with open(halocat_filepath, "r") as f:
            #-- parse header
            line0 = f.readline()
            line0 = line0.strip()
            colnames = line0.split()

            #-- metadata for table
            names = [colnames[i][:-4].lower() if (colnames[i][-4]=="(") else colnames[i][:-3].lower() for i in colnums]
            dtype = [int if (("id" in colnames[i].lower()) or ("?" in colnames[i])) else float for i in colnums]

            if not quiet:
                for i in range(len(names)):
                    print(f"{i}\t{colnums[i]}\t{names[i]}")

            #-- skip over column definitions
            for i in range(63):
                line = f.readline()

            chunks = []
            chunk_num = 0
            while line != "":
                this_chunk = []
                for n in range(chunk_length):
                    line = f.readline().strip()
                    if line=="":
                        break
                    row = np.asarray(line.split())
                    this_chunk.append([float(i) if (("e" in i) or ("." in i)) else int(i) for i in [row[j] for j in colnums]])     
                this_chunk = Table( np.asarray(this_chunk), names=names, dtype=dtype )
                mask = this_chunk[ sham_tag ] > sham_tag_min
                chunks.append( this_chunk[mask] )
                if not quiet and chunk_num%100==0:
                    print(chunk_num)
                chunk_num += 1
                if (N_chunks != "all") & (chunk_num==N_chunks):
                    break
        f.close()
        halocat = vstack(chunks)
        if not quiet:
            print("Your simulation snapshot is ready!")

        save_dir_npy = "/" + os.path.join(*halocat_filepath.split("/")[1:-1])
        fname_npy    = "{}p{}.npy".format(*halocat_filepath.split("/")[-1].split(".")[:-1])
        sham_tag     = None
        if "vpeak" in halocat_filepath:
            sham_tag = "vpeak"
        elif ("vmax" in halocat_filepath) & ("mpeak" not in halocat_filepath):
            sham_tag = "vmax"
        elif "vmax\@mpeak" in halocat_filepath:
            sham_tag = "vmax\@mpeak"

        if sham_var != None:
            save_dir_npy += f"/{sham_var}min{int(sham_var_min)}"
            Path( save_dir_npy ).mkdir(parents=True, exist_ok=True)
            if not quiet:
                print(f"Saving {save_dir_npy}/{fname_npy}...\n{sham_tag} > {sham_tag_min}...")
                np.save(f"{save_dir_npy}/{fname_npy}", halocat[halocat[sham_tag] >= sham_tag_min])
        else:
            if not quiet:
                print(f"Saving {save_dir_npy}/{fname_npy}...")
            np.save(f"{save_dir_npy}/{fname_npy}", halocat)

        return Table(halocat)

    else:
        if os.path.exists( halocat_filepath ):
            if not quiet:
                print(f"Loading {halocat_filepath}...")
            halocat = np.load( halocat_filepath )
            return Table(halocat)
        else:
            print(f"{halocat_filepath} not found!")
            
    

def load_trees(treefilepath, quiet=False):
    """
    Parses raw ascii (.dat) Rockstar tree files and returns a dictionary of tables indexed by z=0 (a=1) halo_id;
    each table contains the full merger tree (history) for that z=0 halo_id
    
    """
    with open(treefilepath, "r") as f:
        #-- DISPLAY HEADER DESCRIPTIONS

        #-- parse header
        line0 = f.readline()
        line0 = line0.strip()
        colnames = line0.split()

        #-- only want to track certain columns for each tree
        colnums = (0,1,2,3,5,6,10,11,12,14,31)
        names = [colnames[i] for i in colnums]
        dtype = [int if (("id" in colnames[i].lower()) or ("?" in colnames[i]) or ("Snap" in colnames[i])) else float for i in colnums]

        # for i in colnums:
        #     print(f"{i}\t{colnames[i]}")

        #-- skip over column definitions
        for i in range(46):
            f.readline()

        #-- next line is total number of trees in this file
        N_trees = int(f.readline().strip())
        if not quiet:
            print(f"Trees in this tree file: {N_trees}")

        tree_dict = {}
        #-- next line is halo ID of first tree
        working_tree_id = int(f.readline().strip().split()[1])
        if not quiet:
            print(f"First tree: {working_tree_id}")
        rows = []
        tree_idx = 1

        for line in f:
            row = line.strip()
            if row=="":
                if not quiet:
                    print("Your ")
                break
            row = row.split()
            if row[0] != "#tree": #-- if NOT at start of a new tree
                rows.append( [row[i] for i in colnums] ) #-- add this line to running array for current tree
            else:
                tree_dict[working_tree_id] = Table(data=np.array(rows), names=names, dtype=dtype) #-- add previous tree to dictionary
                if not quiet:
                    print(f"Adding tree {working_tree_id} to dictionary...")
                last_tree_id = working_tree_id
                tree_idx += 1
                if tree_idx%1000==0 and not quiet:
                    print(f"{tree_idx} / {N_trees}")
                rows = [] #-- reset array to collect rows for next tree
                working_tree_id = int(row[1]) #-- number of tree about to be parsed
                if not quiet:
                    print(f"Next tree: {working_tree_id}") #-- ID of previously parsed tree

        tree_dict[working_tree_id] = Table(data=np.array(rows), names=names, dtype=dtype) #-- add final tree to dictionary
        if not quiet:
            print(f"Adding tree {working_tree_id} to dictionary...")
            print("\nDONE!")

        f.close()
    
    return tree_dict



def get_hist_from_tree(halo_id, tree_dict, quiet=True, sim_tag="mdpl2"):
    """
    Returns full (sub)halo history [mvir,rvir,rs,id] from z=0 for specified z=0 halo_id;
    length of full history = N_snaps of simulation
    
    """
    N_steps = len(Table(ascii.read(f"/{SIMDIR}/{sim_tag}/snaps.txt")))

    this_tree = tree_dict[halo_id]
    tbl = [this_tree[(this_tree["id(1)"]==halo_id)]]

    tbl.append(this_tree[(this_tree["desc_id(3)"]==halo_id) & (this_tree["mmp?(14)"]==1)])

    temp_id  = this_tree[(this_tree["desc_id(3)"]==halo_id) & (this_tree["mmp?(14)"]==1)]["id(1)"]
    sub_tree = this_tree[(this_tree["desc_id(3)"]==temp_id) & (this_tree["mmp?(14)"]==1)]

    tbl.append(sub_tree)
    
    j = 0
    while len(sub_tree) > 0:
        j += 1
        temp_id = sub_tree["id(1)"]
        sub_tree = this_tree[(this_tree["desc_id(3)"]==temp_id) & (this_tree["mmp?(14)"]==1)]
        tbl.append(sub_tree)

    h = vstack(tbl)

    fill = np.zeros(N_steps-len(h),dtype=int)

    mvir_hist = np.concatenate([h["mvir(10)"],fill])
    rvir_hist = np.concatenate([h["rvir(11)"],fill])
    rs_hist   = np.concatenate([h["rs(12)"],fill])
    id_hist   = np.concatenate([h["id(1)"],fill])

    return mvir_hist, rvir_hist, rs_hist, id_hist



def get_subhist_from_tree(halo_id, this_tree, quiet=True, N_steps=None):
    """
    Returns (sub)halo histories beginning from z > 0 (scale factor < 1);
    N_steps sets the length of the history to return
    """
    if N_steps==None:
        raise Exception("Must specify N_steps for subhistories!")
    
    tbl = [this_tree[(this_tree["id(1)"]==halo_id)]]

    tbl.append(this_tree[(this_tree["desc_id(3)"]==halo_id) & (this_tree["mmp?(14)"]==1)])

    temp_id  = this_tree[(this_tree["desc_id(3)"]==halo_id) & (this_tree["mmp?(14)"]==1)]["id(1)"]
    sub_tree = this_tree[(this_tree["desc_id(3)"]==temp_id) & (this_tree["mmp?(14)"]==1)]

    tbl.append(sub_tree)

    j = 0
    while len(sub_tree) > 0:
        j += 1
        temp_id = sub_tree["id(1)"]
        sub_tree = this_tree[(this_tree["desc_id(3)"]==temp_id) & (this_tree["mmp?(14)"]==1)]
        tbl.append(sub_tree)

    h = vstack(tbl)

    fill = np.zeros(N_steps-len(h),dtype=int)

    mvir_hist = np.concatenate([h["mvir(10)"],fill])
    rvir_hist = np.concatenate([h["rvir(11)"],fill])
    rs_hist   = np.concatenate([h["rs(12)"],fill])
    id_hist   = np.concatenate([h["id(1)"],fill])
    
    return mvir_hist, rvir_hist, rs_hist, id_hist

