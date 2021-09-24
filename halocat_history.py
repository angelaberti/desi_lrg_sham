import os
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join


def read_halocat(halocat_filepath, quiet=True):
    with open(halocat_filepath, "r") as f:

        if not quiet:
            print(f"Reading {halocat_filepath}...")

        ## parse header
        line0 = f.readline()
        line0 = line0.strip()
        colnames = line0.split()

        ## obtain metadata for table
        #colnums = (1,5,6,10,11,12,15,16,17,18,19,59,60,61,62,63,69,70,71,72,73,74)
        colnums = (1,6,10,16,17,18,19,60,62,70,74)
        names = [colnames[i][:-4].lower() if (colnames[i][-4]=="(") else colnames[i][:-3].lower() for i in colnums]
        dtype = [int if (("id" in colnames[i].lower()) or ("?" in colnames[i])) else float for i in colnums]

        if not quiet:
            for i in range(len(names)):
                print(f"{i}\t{colnums[i]}\t{names[i]}")

        ## skip over column definitions
        for i in range(63):
            line = f.readline()

        chunks = []
        chunk_num = 0
        while line != "":
            chunk = []
            for n in range(50000):
                line = f.readline().strip()
                if line=="":
                    break
                row  = np.asarray(line.split())
                chunk.append([float(i) if (("e" in i) or ("." in i)) else int(i) for i in [row[j] for j in colnums]])     
            chunks.append(Table(np.asarray(chunk), names=names, dtype=dtype))
            if not quiet and chunk_num%10==0:
                print(chunk_num)
            chunk_num += 1
            
        if not quiet:
            print("EOF")

        halocat = vstack(chunks)

        f.close()
        
    return halocat


def load_trees(treefilepath, quiet=False):
    with open(treefilepath, "r") as f:
        ## DISPLAY HEADER DESCRIPTIONS

        ## parse header
        line0 = f.readline()
        line0 = line0.strip()
        colnames = line0.split()

        ## only want to track certain columns for each tree
        colnums = (0,1,2,3,5,6,10,11,12,14,31)
        names = [colnames[i] for i in colnums]
        dtype = [int if (("id" in colnames[i].lower()) or ("?" in colnames[i]) or ("Snap" in colnames[i])) else float for i in colnums]

        # for i in colnums:
        #     print(f"{i}\t{colnames[i]}")

        ## skip over column definitions
        for i in range(45):
            f.readline()

        ## next line is total number of trees in this file
        N_trees = int(f.readline().strip())
        if not quiet:
            print(f"Trees in this tree file: {N_trees}")

        tree_dict = {}
        ## next line is halo ID of first tree
        working_tree_id = int(f.readline().strip().split()[1])
        if not quiet:
            print(f"First tree: {working_tree_id}")
        rows = []
        tree_idx = 1

        for line in f:
            row = line.strip()
            if row=="":
                if not quiet:
                    print("EOF")
                break
            row = row.split()
            if row[0] != "#tree": ## if NOT at start of a new tree
                rows.append([row[i] for i in colnums]) ## add this line to running array for current tree
            else:
                tree_dict[working_tree_id] = Table(data=np.array(rows), names=names, dtype=dtype) ## add previous tree to dictionary
                if not quiet:
                    print(f"Adding tree {working_tree_id} to dictionary...")
                last_tree_id = working_tree_id
                tree_idx += 1
                if tree_idx%1000==0 and not quiet:
                    print(f"{tree_idx} / {N_trees}")
                rows = [] ## reset array to collect rows for next tree
                working_tree_id = int(row[1]) ## number of tree about to be parsed
                if not quiet:
                    print(f"Next tree: {working_tree_id}") ## ID of previously parsed tree

        tree_dict[working_tree_id] = Table(data=np.array(rows), names=names, dtype=dtype) ## add final tree to dictionary
        if not quiet:
            print(f"Adding tree {working_tree_id} to dictionary...")
            print("\nDONE")

        f.close()
    
    return tree_dict



def get_hist_from_tree(halo_id, tree_dict, quiet=True, N_steps=178):
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



def get_subhist_from_tree(halo_id, tree_dict, tree_id, quiet=True, N_steps=None):
    if N_steps==None:
        raise Exception("Must specify N_steps for subhistories")
    
    this_tree = tree_dict[tree_id]
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

