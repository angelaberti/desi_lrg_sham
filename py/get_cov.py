import numpy as np
import os

from clustering import wp_from_rp_pi
from params import rp_mids

BASE = "/uufs/astro.utah.edu/common/home/u6032461/DESI_LRG_SHAM/clustering/zmaglim20p7"

Mr_subs = ["Mrn21p0-n20p8","Mrn21p2-n21p0","Mrn21p4-n21p2","Mrn30p0-n21p4","no_abs_mag_bins"]
Mr_subs = [f"Mrlimn20p8/{sub}" for sub in Mr_subs]
  
MW1_subs = ["MW1n22p55-n22p25","MW1n22p85-n22p55","MW1n23p15-n22p85","MW1n30p0-n23p15","no_abs_mag_bins"]
MW1_subs = [f"MW1limn22p25/{sub}" for sub in MW1_subs]

all_subs = np.concatenate([Mr_subs,MW1_subs,["no_abs_mag_cut"]])

END = "rp_pi_counts/z0p40-0p50_north_GXG.npy"

filelist = [f"{BASE}/{s}/{END}" for s in all_subs]

for f in filelist:
    print(f)    
    wp_from_rp_pi( f, rp_mids=rp_mids, quiet=False )
