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
# d        = "south"
# pimax    = 150


def _compile_chisq(N_chisq=1, rp_use_range=None, d="south", band=None, zmin=None, sim_tag=None):
    assert(sim_tag != None)
    rp_use_tag = get_rp_use_tag(rp_use_range, return_tag=True)

    out = []

    mag_bins_clust = get_abs_mag_bins_clust(zmin, band, nbins=4)
    z_snaps,_,_ = get_zsnap_data(sim_tag)
    zsim      = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
    zsnap_tag = get_zsnap_tag(zsim)

    zsim = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
    zsnap_tag = get_zsnap_tag(zsim)

    abs_mag_lim = get_abs_mag_lim(zmin, band)
    if band=="Mr":
        abs_mag_lim_tag = get_Mr_tag(abs_mag_lim)
    elif band=="Mz":
        abs_mag_lim_tag = get_Mz_tag(abs_mag_lim)
    elif band=="MW1":
        abs_mag_lim_tag = get_MW1_tag(abs_mag_lim)

    #-- load linear fit mag bin chisq data
    fname = f"{BASEDIR}/chisq/{rp_use_tag}/{d}_{sham_tag}_{abs_mag_lim_tag}_{zsnap_tag}.npy"
    chisq_linear = np.load(fname, allow_pickle=True).item()

    #-- parse mag bin keys
    mag_bin_tags = list(chisq_linear.keys())

    #-- define mag bin centers
    mag_bin_edges = [-float(j[1:].replace("p",".")) for j in np.array([i.split("-") for i in mag_bin_tags]).T[1]]
    mag_bin_cens  = [round(np.mean(mag_bins_clust[k:k+2]),3) for k in range(len(mag_bins_clust)-1)]

    #-- for each mag bin extract sigma_const(mag) and sigma_const_los(mag) of minimum N values of chisq
    for idx,bin_tag in enumerate(mag_bin_tags):
            bin_data = chisq_linear[bin_tag][0]
            bin_data.sort("chisq")
            chunk = bin_data[:N_chisq]
            if len(chunk) > 0:
                chunk.add_column(Column(data=[d for j in range(N_chisq)], name="field"), index=0)
                chunk.add_column(Column(zsim*np.ones(N_chisq), name="zsim"), index=1)
                chunk.add_column(Column(data=[band for j in range(N_chisq)], name="band"), index=2)
                chunk.add_column(Column(mag_bin_cens[idx]*np.ones(N_chisq), name="mag"), index=3)
                out.append(chunk)

    out = vstack(out)
    out["sigma_los"] = out["sigma_los"].astype(int)

    tmp = np.round(np.array(out["sigma_const"]),2)
    out.replace_column("sigma_const",tmp)

    tmp = np.round(np.array(out["chisq"]),1)
    out.replace_column("chisq",tmp)
    
    return out



def sig_tanh(mag, alpha, beta, m0, sig0=0.3):
    """
    Hyperbolic tangent fit: 3 free parameters (alpha, beta, m0)
    sigma( mag ) = sig0 + alpha*tanh( beta*(mag - m0) )
    Default values: sig0=0.3
    """
    return sig0 + alpha*np.tanh(beta*(mag - m0))


def sig_lin(mag, alpha, sig0):
    """
    Linear fit: 2 free parameters (alpha, sig0)
    sigma( mag ) = sig0 + alpha*mag
    """
    return sig0 + alpha*mag



def _get_popt(band, zmin, N_chisq=1, rp_use_range=None, sim_tag=None, quiet=True):
    assert(sim_tag != None)
    # if not quiet:
    #     print(f"zrange: ({zmin}, {zmin+0.1})")
    #     print(f"band: {band}")
    #     print(f"N min chi^2: {N_chisq}")
    #     print(f"{rp_use_range}\n")
    popt = {}
    scatter_key = f"{band}_zmin{str(zmin).replace('.','p')}_chisq{N_chisq}"

    kwargs = dict(band=band, zmin=zmin, N_chisq=N_chisq, sim_tag=sim_tag, rp_use_range=rp_use_range)
    out  = _compile_chisq(**kwargs)
    mask = (out["band"]==band) & (np.round(out["zsim"],1)==zmin)
    zsim = np.round(np.unique(out["zsim"][mask]),3)[0]

    t = []
    for mag in np.unique(out["mag"][mask]):
        t.append(out[mask & (out["mag"]==mag)])
    t = vstack(t)

    fig, axes = plt.subplots(1, 2, figsize=(12,5.5))

    ax = axes[0]
    ax.set_xlim(0.0,1.0)
    ax.set_xlabel("sigma SHAM (mag)")
    ax.set_ylim(0,125)
    ax.set_ylabel("sigma LOS (Mpc/$h$)")
    ax.grid()
    colors = get_colors(len(np.unique(t["mag"])))

    dd = []
    for color,mag in zip(colors,np.unique(t["mag"])):
        mask = t["mag"]==mag
        s = 100
        ax.scatter(np.mean(t["sigma_const"][mask]), np.mean(t["sigma_los"][mask]), color=color, label=f"{band} = {mag}", s=100)
        sig_v   = np.mean(t["sigma_const"][mask])
        sig_los = np.mean(t["sigma_los"][mask])
        w = np.mean(1/t["chisq"][mask])
        dd.append([mag, sig_v, sig_los, w])

    dd = np.array(dd).T
    ax.legend(frameon=True)

    mag_array = np.linspace(np.min(dd[0])-1,np.max(dd[0])+1,100)

    ax = axes[1]
    ax.set_xlim(np.min(dd[0])-1, np.max(dd[0])+1)
    ax.set_xlabel(band)
    ax.set_ylim(axes[0].get_xlim())
    ax.set_ylabel(axes[0].get_xlabel())
    ax.scatter(dd[0], dd[1], color="red")
 
    #-- tanh fit only possible if optimal scatter values for lowest three amgnitude bins *NOT* equal
    if ( (dd[1][1] != dd[1][2]) | (dd[1][1] != dd[1][3]) ) & (dd[1][0] < dd[1][1]):
        popt_sham_tanh,_ = curve_fit( sig_tanh, dd[0], dd[1], bounds=((0,0,-25),(1,2,-20)) )
        ax.plot(mag_array, sig_tanh(mag_array, *popt_sham_tanh), color="red", label="sigma SHAM tanh (4 bins)")
        popt["sham_tanh"] = popt_sham_tanh

    popt_sham_lin3,_  = curve_fit( sig_lin, dd[0][1:], dd[1][1:] )
    popt["sham_lin3"] = popt_sham_lin3

    popt_sham_lin4,_  = curve_fit( sig_lin, dd[0], dd[1] )
    popt["sham_lin4"] = popt_sham_lin4

    ax.plot(mag_array, sig_lin(mag_array, *popt_sham_lin4), color="k", ls="--", label="sigma SHAM linear (4 bins)")

    ax.legend(loc=2)

    ax = axes[1].twinx()
    ax.set_ylim(axes[0].get_ylim())
    ax.set_ylabel(axes[0].get_ylabel())
    ax.scatter(dd[0], dd[2], marker="x", color="black")

    #-- linear fit sigma_LOS(mag) to 3 dimmest mag bins
    popt_los_lin3,_  = curve_fit( sig_lin, dd[0][1:], dd[2][1:] )
    popt["los_lin3"] = popt_los_lin3
    #-- linear fit sigma_LOS(mag) to all 4 mag bins
    popt_los_lin4,_  = curve_fit( sig_lin, dd[0], dd[2] )
    popt["los_lin4"] = popt_los_lin4
    #-- single sigma_LOS value for each mag bin (excluding brightest)
    popt["los_binned3"] = np.mean(dd[2][1:])
    #-- single sigma_LOS value for each mag bin (excluding brightest)
    popt["los_binned4"] = np.mean(dd[2])

    # ax.plot(mag_array, sig_lin(mag_array, *popt_los_lin3), color="gray", lw=8, alpha=0.4, label="sigma LOS linear (3 bins)")
    ax.plot(mag_array, sig_lin(mag_array, *popt_los_lin4), color="black", lw=1.5, ls=":", label="sigma LOS linear (4 bins)")
    
    ax.legend(loc=4, markerfirst=False)
    ax.text(get_corners(ax)["upper_left"][0], 90, r"$z_{\rm sim}=\ $" + f"${zsim:.3f}$\n" + r"${\rm model}$ " + f"${N_chisq}$", ha="left", va="top", fontsize=20)
    plt.tight_layout()

    return popt



def _model_wp(band, zmin,
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
              N_chisq            = 1,
              rp_use_range       = None,
              bin_file           = None,
              sham_scatter_key   = None,
              los_scatter_key    = None,
              quiet              = True,
             ):
    assert(sim_tag != None)
    
    z_snaps,_,_ = get_zsnap_data(sim_tag)
    zsim      = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
    zsnap_tag = get_zsnap_tag(zsim)

    mock_scatter_tag = f"{sham_scatter_key.replace('_','-')}_chisq{N_chisq}"
    rp_use_tag       = get_rp_use_tag(rp_use_range)
    popt             = _get_popt(band, zmin, N_chisq=N_chisq, rp_use_range=rp_use_range, sim_tag=sim_tag)

    if cat==None:
        f = f"{BASEDIR}/mocks/{sim_tag}/{sham_tag}/{d}/{zsnap_tag}_{zmag_tag}_{abs_mag_lim_tag}_{mock_scatter_tag}_{rp_use_tag}"

        galcat_fname = f"{f}_galcat_LRG-flagged.npy"
        if not os.path.exists(galcat_fname):
            galcat_fname = f"{f}.npy"
        if not quiet:
            print(f"Loading {galcat_fname}...\n")

        cat = Table(np.load(galcat_fname))
    cat = cat[cat["galaxy"]==True]

    #-- compute sigma_los for specified parameterization
    if los_scatter_key==None:
        sigma_los = np.zeros(len(cat))
    elif "los_binned" in los_scatter_key:
        sigma_los = (popt[los_scatter_key])*np.ones(len(cat))
    else:
        sigma_los = sig_lin(cat[band].data, *popt[los_scatter_key])
    sigma_los[np.where(sigma_los < 0)[0]] = 0

    if abs_mag_bin_tags != None:
        model_mags = cat[band].data
        to_return  = {}
        for i,tag in enumerate(abs_mag_bin_tags):
            mag_min = -np.float(tag.split("-")[0].split("n")[-1].replace("p","."))
            mag_max = -np.float(tag.split("-")[-1].split("n")[-1].replace("p","."))
            # print(mag_min, mag_max)
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
        


def _plot(band, zmin,
          cat                = None,
          sham_scatter_key   = None,
          los_scatter_key    = None,
          N_chisq            = 1,
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
          popt               = None,
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
                           return_model_error=return_model_error, abs_mag_lim_tag=abs_mag_lim_tag,
                           sham_scatter_key=sham_scatter_key, los_scatter_key=los_scatter_key, sim_tag=sim_tag,
                           N_chisq=N_chisq, rp_use_range=rp_use_range, quiet=quiet, bin_file=bin_file)
    models = {}
    sham_scatter_tag = sham_scatter_key.replace("_","-")
    if los_scatter_key != None:
        fig_scatter_tag  = f"{sham_scatter_tag}_chisq{N_chisq}_{los_scatter_key.replace('_','-')}"
    else:
        fig_scatter_tag  = f"{sham_scatter_tag}_chisq{N_chisq}_los-None"

    args   = (band, zmin)
    wp_mod = _model_wp(*args, **kwargs_model_wp)
    models[sham_scatter_tag] = wp_mod
        
    rp_use_tag, rp_use_label = get_rp_use_tag(rp_use_range, return_tag=True, return_label=True)
        
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
    N_dof = 3 if "tanh" in sham_scatter_tag else 2

    #-- make the figure
    fig, ax = plt.subplots(1, 1, figsize=(8,7))

    ax.set_xlim(0.02, 50)
    ax.set_ylim(20, 140)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_xlabel(fig_labels["rp"], fontsize=24)
    ax.set_ylabel(fig_labels["rpwp"], fontsize=24)

    mag_label = r" $(M_{W1} <\ $" if (band=="MW1") else f" $(M_{band[1:]} <\ $"
    abs_mag_lim = get_abs_mag_lim(zmin, band)
    data_label = r"${\rm data}$" + mag_label + f"${abs_mag_lim})$"

    #-- plot data points
    ax.errorbar(rp_cen, rp_cen*wp, yerr=rp_cen*wp_err, label=data_label, marker="o", ms=5, color="k", linestyle="")

    #-- plot models
    colors = get_colors(3)
    lines  = ("-", "dashdot", "--", ":")
    
    for idx,k in enumerate(models.keys()):
        wp_mod = h*models[k]
        chisq  = np.sum([ [ (wp - wp_mod)[i]*cov_inv[i,j]*(wp - wp_mod)[j] for i in rp_use_idx ] for j in rp_use_idx ])
        chisq_dof = chisq/(N_rp - N_dof)
        chisq_label = r"$\chi^2_{\nu}=\ $" + f"${int(chisq_dof)}$"
        model_label = r"${\rm model}$" + f" $(${rp_use_label}$;\ ${chisq_label}$)$"
        ax.plot(rp_cen, rp_cen*wp_mod, color=colors[idx], label=model_label, lw=2, ls=lines[idx])

    zphot_label = f"${zmin}\ $" + r"$ < z_{\rm phot} < $" + f"$\ {zmax}$"
    ax.text(get_corners(ax, logx=True, margin=0.05)["upper_left"][0], 110, zphot_label, ha="left", va="top", fontsize=24)

    ax.legend(fontsize=20, loc=2, handletextpad=0.5)
    ax.semilogx()
    if rp_use_min != None:
        ax.fill_between((ax.get_xlim()[0],rp_use_min), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)
    if rp_use_max != None:
        ax.fill_between((rp_use_max,ax.get_xlim()[-1]), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)

    plt.tight_layout()

    fpath = f"{BASEDIR}/figures/wp_model-vs-data/{band}/{cat_tag[:-6]}"
    fname = f"{abs_mag_lim_tag}_{fig_scatter_tag}_{rp_use_tag}.png"
    if not save:
        print(f"{fpath}/{fname}")
    elif save:
        print(f"\n{now()} Saving {fpath}/{fname}...")
        plt.savefig(f"{fpath}/{fname}", bbox_inches="tight", pad_inches=0.1, dpi=200)
    return

    

def _plot_mag_bins(band, zmin,
                   cat                = None,
                   sham_scatter_key   = None,
                   los_scatter_key    = None,
                   N_chisq            = 1,
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
                   popt               = None,
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

    sham_scatter_tag = sham_scatter_key.replace("_","-")
    sham_scatter_tag = sham_scatter_key.replace("_","-")
    if los_scatter_key != None:
        fig_scatter_tag  = f"{sham_scatter_tag}_chisq{N_chisq}_{los_scatter_key.replace('_','-')}"
    else:
        fig_scatter_tag  = f"{sham_scatter_tag}_chisq{N_chisq}_los-None"
    args = (band, zmin)
    kwargs_model_wp = dict(DS=DS, n_iter=n_iter, boxsize=boxsize, BASEDIR=BASEDIR, cat_tag=cat_tag, cat=cat,
                           return_model_error=return_model_error, abs_mag_lim_tag=abs_mag_lim_tag, bin_file=bin_file,
                           abs_mag_bin_tags=abs_mag_bin_tags, rp_use_range=rp_use_range, quiet=quiet, sim_tag=sim_tag,
                           sham_scatter_key=sham_scatter_key, los_scatter_key=los_scatter_key, N_chisq=N_chisq)
    models = _model_wp(*args, **kwargs_model_wp)
    
    rp_use_tag, rp_use_label = get_rp_use_tag(rp_use_range, return_tag=True, return_label=True)
    
    rp_cen = h*rp_mids
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
    N_dof = 3 if "tanh" in sham_scatter_tag else 2

    #-- make the figure
    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.set_xlim(0.02, 50)
    ax.set_ylim(1, 6e3)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_xlabel(fig_labels["rp"], fontsize=24)
    ax.set_ylabel(fig_labels["wp"], fontsize=24)
    
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

        #-- plot data points
        mag_label = get_abs_mag_bin_label(abs_mag_bin_tag)
        ax.errorbar(rp_cen, wp, yerr=wp_err, label=mag_label, marker="o", ms=5, color=colors[idx], linestyle="")

        #-- plot models    
        wp_mod = h*models[abs_mag_bin_tag]
        chisq  = np.sum([ [ (wp - wp_mod)[i]*cov_inv[i,j]*(wp - wp_mod)[j] for i in rp_use_idx ] for j in rp_use_idx ])
        chisq_dof = chisq/(N_rp - N_dof)
        chisq_label = r"$\chi^2_{\nu}=\ $" + f"${int(chisq_dof)}$"
        ax.plot(rp_cen, wp_mod, color=colors[idx], label=chisq_label, lw=2, ls=lines[idx])

    zphot_label = f"${zmin}\ $" + r"$ < z_{\rm phot} < $" + f"$\ {zmax}$" + f"\n{rp_use_label}"
    ax.text(*get_corners(ax, log=True, margin=0.05)["upper_right"], zphot_label, ha="right", va="top", fontsize=24)

    ax.legend(fontsize=20, loc=3, handletextpad=0.5, labelspacing=0.3)
    ax.loglog()

    if rp_use_min != None:
        ax.fill_between((ax.get_xlim()[0],rp_use_min), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)
    if rp_use_max != None:
        ax.fill_between((rp_use_max,ax.get_xlim()[-1]), ax.get_ylim()[1]*np.ones(2), color="gray", alpha=0.1)

    plt.tight_layout()

    fpath = f"{BASEDIR}/figures/wp_model-vs-data/{band}/{cat_tag[:-6]}"
    fname = f"{abs_mag_lim_tag}_{fig_scatter_tag}_{rp_use_tag}_mag-bins.png"
    if not save:
        print(f"{fpath}/{fname}")
    elif save:
        print(f"\n{now()} Saving {fpath}/{fname}...")
        plt.savefig(f"{fpath}/{fname}", bbox_inches="tight", pad_inches=0.1, dpi=200)
    return
