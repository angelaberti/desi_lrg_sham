import os, sys
from pathlib import Path

import numpy as np

from params import BASEDIR, SIMDIR, MOCKDIR, get_zsnap_data
from halocat_history import read_halocat


sim_tag = "mdpl2"

# scales = [0.70160, 0.51450, 0.54980, 0.57470, 0.61420, 0.65650, 0.74980, 0.80130, 0.87550, 0.95670]
scales = [0.54980, 0.57470, 0.61420, 0.65650, 0.74980, 0.80130, 0.87550, 0.95670]

save_path = f"{SIMDIR}/{sim_tag}/CATALOGS/vpeakmin125"
Path( save_path ).mkdir(parents=True, exist_ok=True)

#-- default columns to save
#-- colnums = (1,6,10,11,12,16,17,18,19,60,62,70,74)
#-- expanded set of columns
colnums_full = (1,5,6,10,11,12,15,16,17,18,19,59,60,61,62,63,69,70,71,72,73,74)
to_remove = [1,6,11,13,15,16,18,19,20]

for a in scales:
    print(f"a = {a}")
    halocat_fname = f"hlist_{a:.5f}.list"
    halocat_fpath = f"{SIMDIR}/{sim_tag}/CATALOGS/{halocat_fname}"

    halocat  = read_halocat(halocat_fpath, N_chunks="all", vpeak_min=0, colnums=colnums_full, quiet=False)
    colnames = halocat.colnames
    print("Removing select columns...")
    for i in to_remove:
        halocat.remove_column( colnames[i] )

    scale = str(a).replace(".","p")
    print("Saving Vpeak >= 125 km/s catalog...")
    np.save( f"{save_path}/a{scale}.npy", halocat[halocat["vpeak"] >= 125] )
    
    print("Removing Rockstar ascii file...\n")
    os.remove( halocat_fname )
