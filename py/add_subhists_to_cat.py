import os, sys
from pathlib import Path

import numpy as np

import itertools as it

from params import BASEDIR, DATADIR, SIMDIR, MOCKDIR, H0, Om0
from params import get_zsnap_data, get_boxsize, get_abs_mag_lim

from utils import *

from astropy.table import Table, Column, hstack, vstack, join

sim_tag      = "mdpl2"
sham_tag     = "vpeak"
d            = "south"
zmag_lim     = 20.7

quiet       = True if (("-q" in sys.argv) | ("--quiet" in sys.argv)) else False
print_every = 10

#-- load galaxy parent sample statistics
#-- first define relelvant parameter values
this_mock_name = sys.argv[1]
zmin = float(sys.argv[2]) #= float(this_mock_name.split("_")[0][5:8].replace("p","."))

if "MW1" in this_mock_name:
    band = "MW1"
elif "Mz" in this_mock_name:
    band = "Mz"
else:
    raise Exception()

zmax   = zmin + 0.1
zrange = (zmin, zmax)


#-- housekeeping: galaxy parent sample
assert((band=="Mr") | (band=="Mz") | (band=="MW1"))
cat_tag     = get_cat_tag(d, zrange)
zmag_tag    = get_zmag_tag(zmag_lim)
zkcorrs,_,_ = get_zsnap_data("bolshoip")
abs_mag_lim = get_abs_mag_lim(zmin, band)
if band=="MW1":
    abs_mag_lim_tag = get_MW1_tag( abs_mag_lim )
elif band=="Mr":
    abs_mag_lim_tag = get_Mr_tag( abs_mag_lim )
elif band=="Mz":
    abs_mag_lim_tag = get_Mz_tag( abs_mag_lim )
else:
    raise Exception()
if not quiet:
    print(f"\nabs_mag_lim_tag = {abs_mag_lim_tag}")
    
#-- housekeeping: simulation
zsnaps, snap_nums, snaps = get_zsnap_data( sim_tag )
this_zsnap = np.array(zsnaps)[ np.round(zsnaps,1)==zmin ][0]
zsnap_tag  = get_zsnap_tag(this_zsnap)

if not quiet:
    print(f"zsnap = {this_zsnap} ({sim_tag})")


a     = snaps["scale"][snaps["redshift"]==this_zsnap][0]
scale = f"{a:.4f}"
scale = scale.replace(".","p")
if not quiet:
    print(f"scale = {a}")

this_mock_dir  = f"{MOCKDIR}/{sim_tag}/{sham_tag}/{d}"
galcat_fname   = f"{this_mock_dir}/{this_mock_name}"

if os.path.exists( galcat_fname ):
    if not quiet:
        print(f"Loading {galcat_fname}...")
    galcat = Table(np.load( galcat_fname ))
else:
    raise Exception(f"{galcat_fname} not found!")

if not quiet:
    print(len(galcat))
    
#-- update column name(s)
if "halo_id" not in galcat.colnames:
    galcat.rename_column("id", "halo_id")

save_as = f"{galcat_fname[:-4]}-hist.npy"

#-- add histories
cnames = galcat.colnames
if ("mvir_hist" not in cnames) | ("rvir_hist" not in cnames) | ("rs_hist" not in cnames) | ("id_hist" not in cnames):
    if not quiet:
        print(f"Adding history information to mock...")
    hist_filelist = np.array([ "a{}_tree_{}_{}_{}.npy".format(scale, *i) for i in list(it.product("0123456789", repeat=3)) ])
    chunks = []
    for n,this_hf in enumerate(hist_filelist):
        if (n+1)%print_every==0 and not quiet:
            print(f"{n+1} / {len(hist_filelist)}")
        this_histpath = f"{SIMDIR}/{sim_tag}/HISTORIES/{this_hf}"
        if os.path.exists( this_histpath ):
            if (n+1)%print_every==0 and not quiet:
                print(f"Loading {this_hf}...")
            this_hist = Table( np.load(this_histpath) )
    
            this_hist.remove_column("halo_id")
            this_hist.add_column(Column(this_hist["id_hist"][:,1], name="halo_id"), index=0)
    
            if (n+1)%print_every==0 and not quiet:
                print(f"Halos in this history chunk: {len(this_hist)}")
            this_chunk = join(galcat.copy(),this_hist,join_type="inner",keys="halo_id")
    
            if (n+1)%print_every==0 and not quiet:
                print(f"Matched histories for {len(this_chunk)} (sub)halos in a={a} mock\n")
            chunks.append(this_chunk)
        else:
            if (n+1)%print_every==0 and not quiet:
                print(f"{this_hf} not found; skipping...\n")

    galcat_with_hist = vstack( chunks )
    for col in ("mvir", "rvir", "rs", "id"):
        galcat_with_hist[f"{col}_hist"] = galcat_with_hist[f"{col}_hist"][:,1:]
    if not quiet:
        print(f"Histories added for {len(galcat_with_hist)} / {len(galcat)} (sub)halos")
        print(f"Saving {save_as}...")

    np.save( save_as, galcat_with_hist )
