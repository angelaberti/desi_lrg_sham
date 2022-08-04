import os, sys
import numpy as np

from pathlib import Path

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join
from astropy.cosmology import FlatLambdaCDM

from scipy import interpolate
import scipy.stats as stats
from scipy.optimize import curve_fit

import itertools as it

import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("./py")
from utils import *

sys.path.append("/Users/aberti/Desktop/research")
from plotutils import get_corners, fig_labels, get_colors

from params import BASEDIR, DATADIR, SIMDIR, H0, Om0
from params import get_boxsize, get_zsnap_data, get_sham_var_bins, get_abs_mag_lim, get_abs_mag_bins_clust

cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

import Corrfunc
from Corrfunc.theory.wp import wp as wp_corrfunc

#-- bins for clustering (data and mocks)
from params import nbins, rp_min, rp_max, rp_bins, rp_mids, bin_file_comoving

zmag_lim = 20.7
sham_tag = "vpeak"


def sig_lin(mag, alpha, sig0):
    """
    Linear fit: 2 free parameters (alpha, sig0)
    sigma( mag ) = sig0 + alpha*mag
    """
    return sig0 + alpha*mag



def load_chisq_linear(rp_use_range=None, rp_use_tag=None, d="south", band=None, zmin=None, sim_tag=None, brightest_mag_bin_rp1Mpch=False):
    assert(sim_tag != None)

    out = []

    abs_mag_bins_clust = get_abs_mag_bins_clust(zmin, band, nbins=4)
    print(abs_mag_bins_clust)
    
    z_snaps,_,_ = get_zsnap_data(sim_tag)
    zsim      = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
    zsnap_tag = get_zsnap_tag(zsim)

    zsim = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
    zsnap_tag = get_zsnap_tag(zsim)

    abs_mag_lim = get_abs_mag_lim(zmin, band)
    if band=="Mz":
        abs_mag_lim_tag = get_Mz_tag(abs_mag_lim)
    elif band=="MW1":
        abs_mag_lim_tag = get_MW1_tag(abs_mag_lim)

    #-- load linear fit mag bin chisq data
    print(rp_use_tag)
    fname = f"{BASEDIR}/chisq/{rp_use_tag}/{d}_{sham_tag}_{abs_mag_lim_tag}_{zsnap_tag}.npy"
    print(fname)
    chisq_linear = np.load(fname, allow_pickle=True).item()

    #-- parse mag bin keys
    mag_bin_tags = list(chisq_linear.keys())

    #-- define mag bin centers
    mag_bin_edges = [-float(j[1:].replace("p",".")) for j in np.array([i.split("-") for i in mag_bin_tags]).T[1]]
            
    return chisq_linear



def calc_model_wp(band, zmin,
              cat                = None,
              d                  = "south",
              DS                 = 1,
              n_iter             = 1,
              nthreads           = 2,
              BASEDIR            = None,
              return_model_error = False,
              sham_tag           = "vpeak",
              sim_tag            = None,
              cat_tag            = None,
              zmag_tag           = "zmaglim20p7",
              boxsize            = None,
              abs_mag_lim_tag    = None,
              abs_mag_bin_tags   = None,
              pimax              = 150.,
              rp_use_range       = None,
              rp_use_tag         = None,
              brightest_mag_bin_rp1Mpch = None,
              bin_file           = None,
              popt_los           = None,
              quiet              = True,
             ):
    assert(sim_tag != None)
    
    z_snaps,_,_ = get_zsnap_data(sim_tag)
    zsim      = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
    zsnap_tag = get_zsnap_tag(zsim)

    if cat==None:
        raise Exception()
#         f = f"{BASEDIR}/mocks/{sim_tag}/{sham_tag}/{d}/{zsnap_tag}_{zmag_tag}_{abs_mag_lim_tag}_{mock_scatter_tag}_{rp_use_tag}"

#         galcat_fname = f"{f}_galcat_LRG-flagged.npy"
#         if not os.path.exists(galcat_fname):
#             galcat_fname = f"{f}.npy"
#         if not quiet:
#             print(f"Loading {galcat_fname}...\n")

#         cat = Table(np.load(galcat_fname))
    cat = cat[cat["galaxy"]==True]

    #-- compute sigma_los for specified parameterization
    #-- no los scatter parameters given
    if popt_los is None:
        sigma_los = np.zeros(len(cat))
        print(0,0)
    #-- linear fit for los scatter as function of model magnitude    
    else:
        m_los, b_los = popt_los
        sigma_los = m_los*cat[band].data + b_los
        # print(m_los, b_los)
    sigma_los[np.where(sigma_los < 0)[0]] = 0

    if abs_mag_bin_tags != None:
        model_mags = cat[band].data
        to_return  = {}
        for i,tag in enumerate(abs_mag_bin_tags):
            mag_min = -np.float(tag.split("-")[0].split("n")[-1].replace("p","."))
            mag_max = -np.float(tag.split("-")[-1].split("n")[-1].replace("p","."))
            mag_mask = (model_mags > mag_min) & (model_mags <= mag_max)
            out = []
            for (u,v,w) in (("x","y","z"),("y","z","x"),("z","x","y")):
                xx = cat[u]
                yy = cat[v]
                zz = cat[w]
                z_scatter = np.array([np.random.normal(zz, sigma_los, len(zz)) for idx in range(n_iter)])
                z_scatter[z_scatter < 0] = np.array(boxsize - np.abs(z_scatter[z_scatter < 0])%boxsize)
                z_scatter[z_scatter > boxsize] = np.array(z_scatter[z_scatter > boxsize]%boxsize)
                
                args = (boxsize, pimax, nthreads, bin_file, xx[mag_mask][::DS], yy[mag_mask][::DS])
                out.append([wp_corrfunc(*args, z_s[mag_mask][::DS], output_rpavg=False)["wp"] for z_s in z_scatter])

            out = np.concatenate(out).T

            wp_mean = np.mean(out, axis=1)
            std     = np.std(out, axis=1)
            model   = Table([rp_mids,wp_mean,std], names=("rp_cen","wp","wp_err"))
            
            if return_model_error:
                to_return[tag] = model
            else:
                to_return[tag] = wp_mean
        return to_return
    else:
        out = []
        for (u,v,w) in (("x","y","z"),("y","z","x"),("z","x","y")):
            xx = cat[u][::DS]
            yy = cat[v][::DS]
            zz = cat[w][::DS]
            z_scatter = np.array([np.random.normal(zz, sigma_los[::DS], len(zz)) for idx in range(n_iter)])
            z_scatter[z_scatter < 0] = np.array(boxsize - np.abs(z_scatter[z_scatter < 0])%boxsize)
            z_scatter[z_scatter > boxsize] = np.array(z_scatter[z_scatter > boxsize]%boxsize)

            out.append([wp_corrfunc(boxsize, pimax, nthreads, bin_file, xx, yy, z_s, output_rpavg=False)["wp"] for z_s in z_scatter])

        out = np.concatenate(out).T

        wp_mean = np.mean(out, axis=1)
        std     = np.std(out, axis=1)
        model   = Table([rp_mids,wp_mean,std], names=("rp_cen","wp","wp_err"))

        if return_model_error:
            return model
        else:
            return wp_mean
        

        
def plot_wp(band, zmin,
          cat                = None,
          DS                 = 1,
          n_iter             = 1,
          boxsize            = None,
          BASEDIR            = None,
          return_model_error = False,
          h                  = H0/100,
          d                  = "south",
          pimax              = 150.,
          zmag_lim           = 20.7,
          cross              = "GXG",
          quiet              = True,
          save               = True,
          rp_use_range       = None,
          rp_use_tag         = None,
          brightest_mag_bin_rp1Mpch = None,
          popt_los           = None,
          bin_file           = None,
          sim_tag            = None,
         ):
    
    assert(sim_tag != None)
    zmag_tag = get_zmag_tag(zmag_lim)
    MW1lim   = get_abs_mag_lim(zmin,"MW1")
    Mrlim    = get_abs_mag_lim(zmin,"Mr")
    Mzlim    = get_abs_mag_lim(zmin,"Mz")
    if band=="MW1":
        abs_mag_lim_tag = get_MW1_tag(MW1lim)
    elif band=="Mr":
        abs_mag_lim_tag = get_Mr_tag(Mrlim)
    elif band=="Mz":
        abs_mag_lim_tag = get_Mz_tag(Mzlim)
    else:
        raise Exception()

    zmax = zmin + 0.1
    # print(f"{band}\t({zmin:.1f},{zmax:.1f})\n")
    
    cat_tag = get_cat_tag(d,(zmin,zmin+0.1))
    if band=="MW1":
        abs_mag_lim_tag = get_MW1_tag( get_abs_mag_lim(zmin,"MW1") )
    elif band=="Mr":
        abs_mag_lim_tag = get_Mr_tag( get_abs_mag_lim(zmin,"Mr") )
    elif band=="Mz":
        abs_mag_lim_tag = get_Mz_tag( get_abs_mag_lim(zmin,"Mz") )
    else:
        raise Exception()

    #-- data
    wp_fname = f"{cat_tag}_{cross}_pimax{int(pimax)}.txt"
    fpath = f"{BASEDIR}/clustering/{zmag_tag}/{abs_mag_lim_tag}/no_abs_mag_bins/wp"
    if not os.path.exists(f"{fpath}/{wp_fname}"):
        fpath = f"{BASEDIR}/clustering/{zmag_tag}/v0.2/{abs_mag_lim_tag}/no_abs_mag_bins/wp"
        
    data = ascii.read(f"{fpath}/{wp_fname}")
    rp_cen = h*data["rp_cen"]
    wp     = h*data["wp"]
    wp_err = h*np.sqrt(data["wp_err"])

    rp_pi_counts_fname = f"{fpath[:-3]}/rp_pi_counts/{cat_tag}_{cross}.npy"
    cov = cov_from_rp_pi(rp_pi_counts_fname, cross=cross, rp_mids=rp_mids, pimax=pimax)
    cov_inv = np.linalg.inv(cov)
    
    kwargs_model_wp = dict(DS=DS, n_iter=n_iter, boxsize=boxsize, BASEDIR=BASEDIR, cat_tag=cat_tag, cat=cat,
                           return_model_error=return_model_error, abs_mag_lim_tag=abs_mag_lim_tag, sim_tag=sim_tag,
                           rp_use_range=rp_use_range, quiet=quiet, bin_file=bin_file, popt_los=popt_los,
                           brightest_mag_bin_rp1Mpch=brightest_mag_bin_rp1Mpch)
    models = {}

    args   = (band, zmin)
    wp_mod = calc_model_wp(*args, **kwargs_model_wp)
                
    rp_use_min, rp_use_max = rp_use_range
    if (rp_use_min != None) & (rp_use_max != None):
        rp_use_idx = np.where((rp_cen > rp_use_min) & (rp_cen < rp_use_max))[0]
    elif (rp_use_min != None) & (rp_use_max==None):
        rp_use_idx = np.where(rp_cen > rp_use_min)[0]
    elif (rp_use_min==None) & (rp_use_max != None):
        rp_use_idx = np.where(rp_cen < rp_use_max)[0]
    else:
        rp_use_idx = np.arange(len(rp_cen))

    N_rp  = len(rp_use_idx)
    N_dof = 4 #if "tanh" in sham_scatter_tag else 4

    #-- make the figure
    fig, ax = plt.subplots(1, 1, figsize=(8,7))

    ax.set_xlim(0.02, 50)
    ax.set_ylim(20, 140)
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_xlabel(fig_labels["rp"], fontsize=30)
    ax.set_ylabel(fig_labels["rpwp"], fontsize=30)

    if zmin==0.4:
        mag_label = r"$\ (^{0.43}M_{W1} < $" if (band=="MW1") else r"$\ (^{0.43}M_z < $"
    elif zmin==0.5:
        mag_label = r"$\ (^{0.52}M_{W1} < $" if (band=="MW1") else r"$\ (^{0.52}M_z < $"
    elif zmin==0.6:
        mag_label = r"$\ (^{0.63}M_{W1} < $" if (band=="MW1") else r"$\ (^{0.63}M_z < $"
    abs_mag_lim = get_abs_mag_lim(zmin, band)
    data_label = r"${\rm data}$" + mag_label + f"${abs_mag_lim})$"

    #-- plot data points
    ax.errorbar(rp_cen, rp_cen*wp, yerr=rp_cen*wp_err, label=data_label, marker="o", ms=5, color="k", linestyle="")

    #-- plot models
    colors = get_colors(2)
    #lines  = ("-", "dashdot", "--", ":")
    
    # for idx,k in enumerate(models.keys()):
    wp_mod = h*wp_mod
    chisq  = np.sum([ [ (wp - wp_mod)[i]*cov_inv[i,j]*(wp - wp_mod)[j] for i in rp_use_idx ] for j in rp_use_idx ])
    # chisq_dof = chisq/(N_rp - N_dof)
    # chisq_label = r"$\chi^2_{\nu}=\ $" + f"${int(chisq_dof)}$"
    chisq_label = r"$\chi^2=\ $" + f"${int(chisq)}$"
    model_label = r"${\rm model}$" + f" $(${rp_use_label}$)$" #;${chisq_label}$)$"
    ax.plot(rp_cen, rp_cen*wp_mod, label=model_label, lw=2, color=colors[0])
    # ax.set_title(chisq_label)
    print(int(chisq))
    
    zphot_label = f"${zmin}$" + r"$ < z_{\rm phot} < $" + f"${zmax}$"
    ax.text(get_corners(ax, logx=True, margin=0.05)["upper_left"][0], 100, zphot_label, ha="left", va="center", fontsize=26)

    ax.legend(fontsize=26, loc=2, handletextpad=0.25, numpoints=2, handlelength=1.5)
    ax.semilogx()
    if rp_use_min != None:
        ax.fill_between((ax.get_xlim()[0],rp_use_min), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)
    if rp_use_max != None:
        ax.fill_between((rp_use_max,ax.get_xlim()[-1]), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)

    plt.tight_layout()

    if brightest_mag_bin_rp1Mpch==True:
        fpath = f"{BASEDIR}/figures/wp_model-vs-data/rpmin0p1Mpch_brightest-mag-bin-rp1Mpch/{band}/{cat_tag[:-6]}"
    else:
        fpath = f"{BASEDIR}/figures/wp_model-vs-data/{band}/{cat_tag[:-6]}"
    fname = f"{abs_mag_lim_tag}_{rp_use_tag}.png"
    if not save:
        print(f"{fpath}/{fname}")
    elif save:
        print(f"\n{now()} Saving {fpath}/{fname}...")
        plt.savefig(f"{fpath}/{fname}", bbox_inches="tight", pad_inches=0.1, dpi=200)
    return

    

def plot_wp_mag_bins(band, zmin,
                   cat                = None,
                   sham_scatter_key   = None,
                   los_scatter_key    = None,
                   DS                 = 1,
                   n_iter             = 1,
                   boxsize            = None,
                   BASEDIR            = None,
                   return_model_error = False,
                   n_mag_bins_clust   = 4,
                   h                  = H0/100,
                   d                  = "south",
                   pimax              = 150.,
                   zmag_lim           = 20.7,
                   cross              = "GXG",
                   quiet              = False,
                   save               = True,
                   rp_use_range       = None,
                   rp_use_tag         = None,
                   brightest_mag_bin_rp1Mpch = None,
                   popt_los           = None,
                   bin_file           = None,
                   sim_tag            = None,
                  ):
    assert(sim_tag != None)
    zmag_tag = get_zmag_tag(zmag_lim)
    MW1lim   = get_abs_mag_lim(zmin,"MW1")
    Mrlim    = get_abs_mag_lim(zmin,"Mr")
    Mzlim    = get_abs_mag_lim(zmin,"Mz")
    if band=="MW1":
        abs_mag_lim_tag = get_MW1_tag(MW1lim)
    elif band=="Mr":
        abs_mag_lim_tag = get_Mr_tag(Mrlim)
    elif band=="Mz":
        abs_mag_lim_tag = get_Mz_tag(Mzlim)
    else:
        raise Exception()

    zmax = zmin + 0.1
    
    cat_tag = get_cat_tag(d,(zmin,zmin+0.1))
    if band=="MW1":
        abs_mag_lim_tag = get_MW1_tag( get_abs_mag_lim(zmin,"MW1") )
    elif band=="Mr":
        abs_mag_lim_tag = get_Mr_tag( get_abs_mag_lim(zmin,"Mr") )
    elif band=="Mz":
        abs_mag_lim_tag = get_Mz_tag( get_abs_mag_lim(zmin,"Mz") )
    else:
        raise Exception()
        
    abs_mag_bins_clust = get_abs_mag_bins_clust(zmin, band, nbins=n_mag_bins_clust)
    abs_mag_bin_tags   = get_abs_mag_bin_tags(zmin, band, nbins=n_mag_bins_clust)

    args = (band, zmin)
    kwargs_model_wp = dict(DS=DS, n_iter=n_iter, boxsize=boxsize, BASEDIR=BASEDIR, cat_tag=cat_tag, cat=cat,
                           return_model_error=return_model_error, abs_mag_lim_tag=abs_mag_lim_tag, bin_file=bin_file,
                           abs_mag_bin_tags=abs_mag_bin_tags, rp_use_range=rp_use_range, quiet=quiet, sim_tag=sim_tag,
                           brightest_mag_bin_rp1Mpch=brightest_mag_bin_rp1Mpch, popt_los=popt_los)
    models = calc_model_wp(*args, **kwargs_model_wp)
        
    rp_cen = h*rp_mids
    rp_use_min, rp_use_max = rp_use_range

    #-- make the figure
    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.set_xlim(0.02, 50)
    ax.set_ylim(1, 6e3)
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_xlabel(fig_labels["rp"], fontsize=30)
    ax.set_ylabel(fig_labels["wp"], fontsize=30)
    
    wp_fname = f"{cat_tag}_{cross}_pimax{int(pimax)}.txt"
    colors = get_colors(len(abs_mag_bin_tags))
    lines  = ("-", "dashdot", "--", ":", "-")

    for idx,abs_mag_bin_tag in enumerate(abs_mag_bin_tags):
        fpath = f"{BASEDIR}/clustering/{zmag_tag}/{abs_mag_lim_tag}/{abs_mag_bin_tag}/wp"
        
        data = ascii.read(f"{fpath}/{wp_fname}")
        rp_cen = h*data["rp_cen"]
        wp     = h*data["wp"]
        wp_err = h*np.sqrt(data["wp_err"])

        rp_pi_counts_fname = f"{fpath[:-3]}/rp_pi_counts/{cat_tag}_{cross}.npy"
        cov = cov_from_rp_pi(rp_pi_counts_fname, cross=cross, rp_mids=rp_mids, pimax=pimax)
        cov_inv = np.linalg.inv(cov)

        #-- plot models    
        wp_mod = h*models[abs_mag_bin_tag]
        
        if (brightest_mag_bin_rp1Mpch==True) and ("n30p0" in abs_mag_bin_tag):
            if (rp_use_max != None):
                rp_use_idx = np.where((rp_cen > 1.0) & (rp_cen < rp_use_max))[0]
            else:
                rp_use_idx = np.where(rp_cen > 1.0)[0]
        else:
            if (rp_use_min != None) & (rp_use_max != None):
                rp_use_idx = np.where((rp_cen > rp_use_min) & (rp_cen < rp_use_max))[0]
            elif (rp_use_min != None) & (rp_use_max==None):
                rp_use_idx = np.where(rp_cen > rp_use_min)[0]
            elif (rp_use_min==None) & (rp_use_max != None):
                rp_use_idx = np.where(rp_cen < rp_use_max)[0]
            else:
                rp_use_idx = np.arange(len(rp_cen))

        N_rp  = len(rp_use_idx)
        N_dof = 4 #if "tanh" in sham_scatter_tag else 4
        
        chisq  = np.sum([ [ (wp - wp_mod)[i]*cov_inv[i,j]*(wp - wp_mod)[j] for i in rp_use_idx ] for j in rp_use_idx ])
        #chisq_dof = chisq/(N_rp - N_dof)
        #chisq_label = r"$\chi^2_{\nu}=\ $" + f"${int(chisq_dof)}$"
        chisq_label = r"$\chi^2=\ $" + f"${int(chisq)}$"
        model_label = r"${\rm model}$" + f" $(${rp_use_label}$)$"#;${chisq_label}$)$"
        print(f"{abs_mag_bin_tag}\t{chisq:.1f}")
        ax.plot(rp_cen, wp_mod, color=colors[idx], label="", lw=2, ls=lines[idx])

        #-- plot data points
        mag_label = get_abs_mag_bin_label(abs_mag_bin_tag)
        ax.errorbar(rp_cen, wp, yerr=wp_err, label=mag_label, marker="o", ms=5, color=colors[idx], linestyle="")

    zphot_label = f"${zmin}$" + r"$ < z_{\rm phot} < $" + f"${zmax}$" + f"\n{rp_use_label}"
    ax.text(*get_corners(ax, log=True, margin=0.05)["upper_right"], zphot_label, ha="right", va="top", fontsize=26)

    ax.legend(fontsize=26, loc=3, handletextpad=0.25, labelspacing=0.3, numpoints=2, handlelength=1.5)
    ax.loglog()

    if rp_use_min != None:
        ax.fill_between((ax.get_xlim()[0],rp_use_min), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)
    if rp_use_max != None:
        ax.fill_between((rp_use_max,ax.get_xlim()[-1]), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)

    plt.tight_layout()

    if brightest_mag_bin_rp1Mpch==True:
        fpath = f"{BASEDIR}/figures/wp_model-vs-data/rpmin0p1Mpch_brightest-mag-bin-rp1Mpch/{band}/{cat_tag[:-6]}"
    else:
        fpath = f"{BASEDIR}/figures/wp_model-vs-data/{band}/{cat_tag[:-6]}"
    fname = f"{abs_mag_lim_tag}_{rp_use_tag}_mag-bins.png"
    if not save:
        print(f"{fpath}/{fname}")
    elif save:
        print(f"\n{now()} Saving {fpath}/{fname}...")
        plt.savefig(f"{fpath}/{fname}", bbox_inches="tight", pad_inches=0.1, dpi=200)
    return

        