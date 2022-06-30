import os, sys
import numpy as np
from pathlib import Path
from astropy.table import Table, Column, vstack, hstack, join
from scipy import interpolate
import scipy.stats as stats
import itertools as it
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("./py")
from utils import *

sys.path.append("/Users/aberti/Desktop/research")
from plotutils import get_corners, fig_labels, get_colors, plot_settings
plt.rcParams.update(**plot_settings)

from params import BASEDIR, DATADIR, SIMDIR, H0, Om0
from params import get_boxsize, get_zsnap_data, get_sham_var_bins, get_abs_mag_lim, get_abs_mag_bins_clust
from params import nbins, rp_min, rp_max, rp_bins, rp_mids, bin_file
from mock_clust import _compile_chisq, sig_tanh, sig_lin, _get_popt, _model_wp, _plot, _plot_mag_bins


"""
python3 py/model_params.py MW1 0.4 &
python3 py/model_params.py MW1 0.5 -q &
python3 py/model_params.py MW1 0.6 -q &

python3 py/model_params.py Mr 0.4 &
python3 py/model_params.py Mr 0.5 &
python3 py/model_params.py Mr 0.6 &
"""


sham_scatter_keys = ["sham_lin4"] #,"sham_tanh")
N_chisq_list      = [1]
rp_use_ranges     = [(1.0,None)] #((0.2,None),(0.3,None),(0.2,20),(0.3,20))

band  = sys.argv[1]
zmin  = np.round(float(sys.argv[2]),1)
zmax  = zmin + 0.1
quiet = True if (("-q" in sys.argv) | ("--quiet" in sys.argv)) else False

zmag_lim = 20.7
sim_tag  = "mdpl2"
sham_tag = "vpeak"
d        = "south"
pimax    = 150
h        = 0.7

#========== BEGIN HOUSEKEEPING ==========#
z_snaps, snap_nums, snaps = get_zsnap_data(sim_tag)

hist_idx = [len(snaps)-i for i in snap_nums]
vol_sim  = get_boxsize(sim_tag)**3
a_snaps = [snaps["scale"][snaps["snapnum"]==sn].data[0] for sn in snap_nums]

sham_var_bins, sham_var_cens = get_sham_var_bins(sham_tag)

zrange      = zmin, zmax
cat_tag     = get_cat_tag(d,zrange)
abs_mag_lim = get_abs_mag_lim(zmin, band)
zsim        = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
zsnap_tag   = get_zsnap_tag(zsim)
zmag_tag    = get_zmag_tag(zmag_lim)

MW1lim = get_abs_mag_lim(zmin,"MW1")
Mrlim  = get_abs_mag_lim(zmin,"Mr")
Mzlim  = get_abs_mag_lim(zmin,"Mz")
if band=="MW1":
    abs_mag_lim_tag = get_MW1_tag(MW1lim)
elif band=="Mr":
    abs_mag_lim_tag = get_Mr_tag(Mrlim)
elif band=="Mz":
    abs_mag_lim_tag = get_Mz_tag(Mzlim)
else:
    raise Exception()

lf_name = f"{BASEDIR}/data/stats/lum_func_{band}_{cat_tag}_{zmag_tag}_{abs_mag_lim_tag}.npy"
lum_func  = np.load(lf_name).item()
lum_func_full = np.load(f"{BASEDIR}/data/stats/lum_func_{band}_{cat_tag}_{zmag_tag}_full.npy").item()
#=========== END HOUSEKEEPING ===========#


################################
##  Load simulation snapshot  ##
################################

a     = snaps["scale"][snaps["redshift"]==zsim][0]
scale = f"{a:.4f}"
scale = scale.replace(".","p")

quiet = False
sham_tag_min = 125

snap_fname = f"{SIMDIR}/{sim_tag}/CATALOGS/{sham_tag}min{int(sham_tag_min)}/a{scale}.npy"

if os.path.exists( snap_fname ):
    if not quiet:
        print(f"\n{now()} Loading simulation shapshot (zsim = {zsim}; {sham_tag} > {sham_tag_min})...")
    halocat = Table( np.load(snap_fname) )
else:
    raise Exception(f"\n{now()} Simulation shapshot (zsim = {zsim}; {sham_tag} > {sham_tag_min}) not found!")

halocat.sort( sham_tag )

halocat = halocat[::-1]


###################################################################
##  Infer halo number densities from target luminosity function  ##
###################################################################
if not quiet:
    print(f"\n{now()} Inferring halo number densities...")

halocat  = halocat[np.log10(halocat["parent_mvir"]) >= 11.5]
sham_var = halocat[sham_tag]

#-- get value of nh = nh( sham_var ) for each value of sham_var

#-- halo number counts in bins of (log_)sham_var
H,_ = np.histogram( np.log10(sham_var), bins=sham_var_bins )

#-- halo abundance versus (min log_)sham_var
cdf_log_sham_var = np.max(np.cumsum(H))-np.cumsum(H)

#-- interpolation function for nh of (log_)sham_var
nh_of_log_sham_var = interpolate.interp1d(sham_var_cens, np.array(cdf_log_sham_var)/vol_sim, fill_value="extrapolate")

#-- inferred halo number densities for sham_var values
nh_inferred = nh_of_log_sham_var( np.log10(sham_var) )

cname = "nh_inferred"
if cname in halocat.colnames:
    halocat.remove_column( cname )
halocat.add_column( Column(nh_inferred, name=cname) )


###################################################################################
##  Assign luminosities *without* scatter based on inferred halo number density  ##
###################################################################################
if not quiet:
    print(f"\n{now()} Assigning unscattered magnitudes...")

#-- get corresponding value of abs_mag for inferred nh values: abs_mag = abs_mag( ngal_eff=nh )
abs_mag_bins = lum_func_full["abs_mag_bins"]
ngal_eff     = lum_func_full["ng_eff"]

#-- interpolation function for abs_mag of ngal_eff
abs_mag_of_ngal_eff = interpolate.interp1d(ngal_eff, abs_mag_bins[::-1], fill_value="extrapolate")

nh_inferred = halocat["nh_inferred"]
abs_mag_inferred = abs_mag_of_ngal_eff( nh_inferred )

#-- add inferred magnitudes (with MONOTONIC Vcirc correspondence) to halo catalog
cname = f"{band}_no_scatter"
if cname in halocat.colnames:
    halocat.remove_column( cname )
halocat.add_column( Column(abs_mag_inferred, name=cname) )

halocat[f"{band}_no_scatter"][halocat[f"{band}_no_scatter"]==-np.inf] = np.inf


# sham_scatter_key = "sham_tanh"
# N_chisq          = 1
# rp_use_range     = (0.2,None)  # Mpc/h
combos = list(it.product(sham_scatter_keys, N_chisq_list, rp_use_ranges))
for (sham_scatter_key, N_chisq, rp_use_range) in combos:
    if not quiet:
        print(f"\n{now()}\nsham_scatter_key: {sham_scatter_key}\nN_chisq: {N_chisq}\nrp_use_range: {rp_use_range}")
    mock_scatter_tag = f"{sham_scatter_key.replace('_','-')}_chisq{N_chisq}"
    rp_use_tag       = get_rp_use_tag(rp_use_range)

    ##########################################################
    ##  Compute sigma array for mag-V$_{\rm circ}$ scatter  ##
    ##########################################################
    if not quiet:
        print(f"\n{now()} Getting optimal scatter parameters...")

    nsig_clip = 2.5

    popt = _get_popt(band, zmin, N_chisq=N_chisq, rp_use_tag=get_rp_use_tag(rp_use_range), quiet=quiet)
    # if not quiet:
    #     print(sham_scatter_key, popt[sham_scatter_key])

    if sham_scatter_key not in popt.keys():
        print(f"'{sham_scatter_key}' not possible for this combination of parameter values; continuing...")
    else:
        if "tanh" in sham_scatter_key:
            sigma_abs_mag_inferred = sig_tanh(halocat[f"{band}_no_scatter"], *popt[ sham_scatter_key ])
        elif "lin" in sham_scatter_key:
            sigma_abs_mag_inferred = sig_lin(halocat[f"{band}_no_scatter"], *popt[ sham_scatter_key ])

        #-- replace negative sigma values with zero if needed
        sigma_abs_mag_inferred[np.where(sigma_abs_mag_inferred < 0)[0]] = 0

        #-- add scatter to inferred magnitudes
        abs_mag_inferred_scattered = stats.truncnorm.rvs(-nsig_clip, nsig_clip, halocat[f"{band}_no_scatter"], sigma_abs_mag_inferred)


        ###########################################
        ##  Add scattered magnitudes to catalog  ##
        ###########################################
        if not quiet:
            print(f"\n{now()} Assigning scattered magnitudes...")

        #-- add inferred magnitudes (with SCATTERED Vcirc correspondence) to halo catalog
        cname = f"{band}_scattered"
        if cname in halocat.colnames:
            halocat.remove_column( cname )
        halocat.add_column( Column(abs_mag_inferred_scattered, name=cname) )

        halocat.sort( f"{band}_scattered" )

        abs_mag_inferred_sorted = halocat.copy()[f"{band}_no_scatter"]
        abs_mag_inferred_sorted.sort()

        if band in halocat.colnames:
            halocat.remove_column( band )
        halocat.add_column( Column(abs_mag_inferred_sorted, name=band) )


        ##########################################
        ##  Flag (sub)halos with mock galaxies  ##
        ##########################################

        #-- remove extra columns
        cols = (f"{band}_scattered", "nh_inferred", "galaxy")
        for c in cols:
            if c in halocat.colnames:
                halocat.remove_column(c)

        #-- sort catalog by model magnitude       
        halocat.sort( band )

        #-- add column for galaxy flag
        c = "galaxy"
        if c in halocat.colnames:
            halocat.remove_column(c)
        halocat.add_column(Column(np.zeros(len(halocat),dtype=bool), name=c))

        #-- number of (sub)halos to select based on number density from data
        Nhalo_select = int(np.max(lum_func["ng_eff"])*vol_sim)

        #-- flag halos with galaxies
        halocat["galaxy"][:Nhalo_select] = True


        #################################
        ##  Save mock with magnitudes  ##
        #################################

        # fname = f"{BASEDIR}/mocks/{sim_tag}/{sham_tag}/{d}/{zsnap_tag}_{zmag_tag}_{abs_mag_lim_tag}_{mock_scatter_tag}_{rp_use_tag}.npy"
        # if not quiet:
        #     print(fname)

        # np.save(fname, halocat)


        ################################
        ##  Iterate over LOS scatter  ##
        ################################

        los_scatter_key = "los_lin4"
        # for los_scatter_key in ("los_lin3","los_lin4"):
        if not quiet:
            print(f"\n{now()} los_scatter_key: {los_scatter_key}")

        kwargs = dict(DS=1, n_iter=1, boxsize=get_boxsize(sim_tag), BASEDIR=BASEDIR, rp_use_range=rp_use_range,
                      N_chisq=N_chisq, sham_scatter_key=sham_scatter_key, los_scatter_key=los_scatter_key,
                      cat   = halocat,
                      quiet = True,
                      save  = True,
                     )
        if not quiet:
            print(f"\n{now()} Plotting full model comparison...")
        _plot(band, zmin, **kwargs)
        if not quiet:
            print(f"\n{now()} Plotting magnitude-binned model comparison...")
        _plot_mag_bins(band, zmin, **kwargs)

print("\nDONE!")
