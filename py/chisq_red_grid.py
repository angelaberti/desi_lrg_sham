import numpy as np

import os, sys
from pathlib import Path

from astropy.io import fits, ascii
from astropy.table import Table, Column#, join
from astropy.cosmology import FlatLambdaCDM
#from astropy.constants import c as c_speed
#from astropy import units as u

#import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams.update({"text.usetex": True})
# plt.rcParams["savefig.dpi"] = 200

sys.path.append("/Users/aberti/Desktop/research")
from plotutils import get_corners, fig_labels, get_colors

from params import BASEDIR, DATADIR, H0, Om0, get_zsnap_data
from utils import *
sys.path.append(BASEDIR)
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)


# python3 py/chisq_red_grid.py north Mr vpeak --zoom &
# python3 py/chisq_red_grid.py north MW1 vpeak --zoom &
# python3 py/chisq_red_grid.py north Mr vmax\\@mpeak --zoom &
# python3 py/chisq_red_grid.py north MW1 vmax\\@mpeak --zoom &

# python3 py/chisq_red_grid.py south Mr vpeak &
# python3 py/chisq_red_grid.py south MW1 vpeak &
# python3 py/chisq_red_grid.py south Mr vmax\\@mpeak &
# python3 py/chisq_red_grid.py south MW1 vmax\\@mpeak &


#-- define rp bins for clustering (data and mocks)
from params import nbins, rp_min, rp_max, rp_bins, rp_mids, bin_file
from params import get_abs_mag_bins_clust, get_abs_mag_lim


sim_tag = "mdpl2"
zsnaps, snap_nums, snaps = get_zsnap_data(sim_tag)


def cov_from_rp_pi( rp_pi_counts, cross="GXG", rp_mids=None, pimax=150., quiet=True ):
    """Calculate wp(rp) from jackknife samples and for multiple values of pimax"""
    if type(rp_pi_counts)==str:
        assert( os.path.exists(rp_pi_counts) )
        if not quiet:
            print(f"Loading {rp_pi_counts}")
        counts = np.load(rp_pi_counts, allow_pickle=True).item()
    else:
        counts = rp_pi_counts

    patches = list(counts[cross].keys())

    #-- sets of pair counts (DD, DR, RD, RR)
    count_sets = [ [counts[cross][p][pair_idx][0]["npairs"] for p in patches ] for pair_idx in range(4) ]
    jk_sets    = [ [(sum(count_sets[pair_idx])-this_set) for this_set in count_sets[pair_idx]] for pair_idx in range(4) ] 
    #-- total number counts (ND1, ND2, NR1, NR2)
    count_set_totals = [ [counts[cross][p][pair_idx][1] for p in patches ] for pair_idx in range(4) ]
    jk_set_totals    = [ [(sum(count_set_totals[pair_idx])-this_tot) for this_tot in count_set_totals[pair_idx]] for pair_idx in range(4) ]

    #-- format for Corrfunc convert_rp_pi_counts_to_wp()
    D1D2_sets, D1R2_sets, D2R1_sets, R1R2_sets = jk_sets
    ND1_tots,  ND2_tots,  NR1_tots,  NR2_tots  = jk_set_totals

    #-- get wp for each jackknife set
    wp_by_patch = []
    n_patches = len(patches)
    for patch_idx in np.arange(n_patches):
        ND1  = ND1_tots[patch_idx]
        ND2  = ND2_tots[patch_idx]
        NR1  = NR1_tots[patch_idx]
        NR2  = NR2_tots[patch_idx]
        D1D2 = D1D2_sets[patch_idx]
        D1R2 = D1R2_sets[patch_idx]
        D2R1 = D2R1_sets[patch_idx]
        R1R2 = R1R2_sets[patch_idx]
        
        wp_this_patch = ( D1D2/(ND1*ND2) - D1R2/(ND1*NR2) - D2R1/(ND2*NR1) ) / ( R1R2/(NR1*NR2) ) + 1
        wp_by_patch.append(wp_this_patch)
    
    #return np.array(wp_by_patch)
    #-- mean wp values for each jackknife sample (patch)
    wp_jk   = 2*np.sum([ np.transpose(wp_by_patch)[i::int(pimax)] for i in range(int(pimax)) ][:int(pimax)], axis=0)
    wp_mean = np.nanmean(wp_jk, axis=1)
    
    N_jk = len(wp_jk.T)
    cov  = np.zeros([len(rp_mids),len(rp_mids)])

    for rp_i in range(len(rp_mids)):
        wp_i = wp_jk[rp_i,:]
        wp_mean_i = wp_mean[rp_i]
        for rp_j in range(len(rp_mids)):
            wp_j = wp_jk[rp_j,:]
            wp_mean_j = wp_mean[rp_j]
            cov[rp_i][rp_j] = (1 - 1/N_jk)*sum((wp_i-wp_mean_i)*(wp_j-wp_mean_j))
    return cov
        


def chisq_red_array(d, band, sham_tag="vpeak", zoom_in=False, zrange=(0.4,0.5), sim_tag="mdpl2", pimax=150, h=0.7, zmag_lim=20.7):
    # for min_rp in rp_mins:
    #     for band in ("Mr","MW1"):
    # print(f"rp > {min_rp} Mpc\t{band}\tzoom = {zoom_in}")
    # d        = "south"
    # band     = "Mr"

    print(f"{d}\t{band}\t{sim_tag}\t{sham_tag}\nzoom_in = {zoom_in}")

    min_rp   = 0.0 #-- minimum rp (Mpc) to include in cov matrices
    cross    = "GXG"

    #-- housekeeping
    # print(f"min rp = {(min_rp):.3f} Mpc ({(h*min_rp):.3f} Mpc/h)")
    zmin, zmax   = zrange
    zsnap        = np.array(zsnaps)[ np.round(zsnaps,1)==zmin ][0]
    zsnap_tag    = get_zsnap_tag(zsnap)
    cat_tag      = get_cat_tag(d,zrange)
    zmag_tag     = get_zmag_tag(zmag_lim)
    abs_mag_lim  = get_abs_mag_lim(zmin,band)
    rp_include   = np.where(h*rp_mids > h*min_rp)[0]
    N_rp_include = len(rp_mids[rp_include])
    DOF          = N_rp_include - 2 #-- 2 model parameters
    # rpmin_tag    = "rpmin" + f"{h*rp_mids[rp_include][0]:.3f}"
    # rpmin_tag    = rpmin_tag.replace(".","p")
    zoom_tag     = "zoom" if zoom_in else "full"

    MW1lim = get_abs_mag_lim(zmin,"MW1")
    Mrlim  = get_abs_mag_lim(zmin,"Mr")

    if zoom_in:
        if (band=="Mr"):
            if (sham_tag=="vpeak"):
                pct_scatter_array = np.arange(45, 71, 1)
                LOS_err_array    = np.arange(15, 90, 5)
            elif (sham_tag=="vmax\\@mpeak"):
                pct_scatter_array = np.arange(65, 81, 1)
                LOS_err_array    = np.arange(15, 90, 5)
            else:
                raise Exception()
        elif (band=="MW1"):
            if (sham_tag=="vpeak"):
                pct_scatter_array = np.arange(35, 66, 1)
                LOS_err_array    = np.arange(0, 65, 5)
            elif (sham_tag=="vmax\\@mpeak"):
                pct_scatter_array = np.arange(50, 81, 1)
                LOS_err_array    = np.arange(0, 65, 5)
            else:
                raise Exception()
    else:
        pct_scatter_array = np.arange(5,151,5)
        LOS_err_array = np.arange(0,101,5)

    
    if band=="MW1":
        abs_mag_lim_tag = get_MW1_tag(MW1lim)
    elif band=="Mr":
        abs_mag_lim_tag = get_Mr_tag(Mrlim)
    else:
        raise Exception()

    # get_abs_mag_bin_tags(zmin, band)[0]
    # abs_mag_bin_tags = np.concatenate( [[get_abs_mag_bin_tags(zmin, band)[0]], get_abs_mag_bin_tags(zmin, band, bright_limit=True), ["no_abs_mag_bins"]] )
    abs_mag_bin_tags = np.concatenate( [get_abs_mag_bin_tags(zmin, band, bright_limit=True), ["no_abs_mag_bins"]] )

    # print(abs_mag_bin_tags)
    # raise Exception()

    save_as = f"{BASEDIR}/chisq_red/{d}_{sham_tag}_{abs_mag_lim_tag}_{zoom_tag}_{get_zsnap_tag(zsnap)}.npy"

    if os.path.exists( save_as ):
        print(f"{save_as} found; loading...\n")
        out = np.load( save_as, allow_pickle=True).item()
    else:
        out = {}

    for abs_mag_bin_tag in abs_mag_bin_tags:
        print(f"mag bin: {abs_mag_bin_tag}")
    
        chisq_matrix = np.zeros([len(pct_scatter_array),len(LOS_err_array)])
        summary = []

        #-- data
        fpath_data    = f"{BASEDIR}/clustering/{get_zmag_tag(zmag_lim)}/{abs_mag_lim_tag}/{abs_mag_bin_tag}"
        wp_fname_data = f"{fpath_data}/wp/{cat_tag}_{cross}_pimax{int(pimax)}.txt"
        # print(f"{pct_scatter}\t{LOS_err}\n{wp_fname}")
        assert(os.path.exists(wp_fname_data))
        data = ascii.read(wp_fname_data)
        rp_cen = h*data["rp_cen"]
        wp     = h*data["wp"]
        wp_err = h*data["wp_err"]

        rp_pi_counts_fname = f"{fpath_data}/rp_pi_counts/{cat_tag}_{cross}.npy"
        cov = cov_from_rp_pi(rp_pi_counts_fname, cross=cross, rp_mids=rp_mids, pimax=pimax)
        cov_inv = np.linalg.inv(cov)

        for u,pct_scatter in enumerate(pct_scatter_array):
            ss = str(pct_scatter)
            if pct_scatter < 10:
                ss = f"0{ss}"
            print(pct_scatter)

            #-- model
            fpath_model = f"{BASEDIR}/clustering/mocks/{sim_tag}/{sham_tag}/{zmag_tag}/{abs_mag_lim_tag}/{abs_mag_bin_tag}/{ss}pct_scatter"
            fname_model = f"{zsnap_tag}_{d}_{cross}_pimax{int(pimax)}.npy"
            if os.path.exists(f"{fpath_model}/{fname_model}"):
                model_all_sigz = np.load(f"{fpath_model}/{fname_model}", allow_pickle=True).item()
            else:
                raise Exception(f"{fpath_model}/{fname_model} not found!")

            for v,LOS_err in enumerate(LOS_err_array):
                model  = Table(np.asarray(model_all_sigz[f"sigz{int(LOS_err)}"]))
                wp_mod = h*model["wp"]

                chisq_red = np.sum([ [ (wp - wp_mod)[i]*cov_inv[i,j]*(wp - wp_mod)[j] for i in range( len(rp_mids[rp_include]) ) ] for j in range( len(rp_mids[rp_include]) ) ])/DOF

                chisq_matrix[u][v] = chisq_red

                summary.append( [chisq_red, pct_scatter, LOS_err] )

        summary = Table(data=np.array(summary), names=("chisq_red","pct_scatter_array","LOS_err"))
        out[ abs_mag_bin_tag ] = ( summary, chisq_matrix )


    print(f"Saving {save_as}...")
    np.save(save_as, out, allow_pickle=True)


assert(len(sys.argv) >= 4)

d, band, sham_tag = sys.argv[1:4]

zoom_in = True if ("--zoom" in sys.argv) else False

chisq_red_array( d, band, sham_tag=sham_tag, zoom_in=zoom_in )

