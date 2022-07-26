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



# def _compile_chisq(N_chisq=1, rp_use_range=None, d="south", band=None, zmin=None, sim_tag=None, brightest_mag_bin_rp1Mpch=False):
#     assert(sim_tag != None)
#     rp_use_tag = get_rp_use_tag(rp_use_range, return_tag=True)
#     if brightest_mag_bin_rp1Mpch==True:
#         rp_use_tag += "_brightest-mag-bin-rp1Mpch"

#     out = []

#     mag_bins_clust = get_abs_mag_bins_clust(zmin, band, nbins=4)
#     z_snaps,_,_ = get_zsnap_data(sim_tag)
#     zsim      = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
#     zsnap_tag = get_zsnap_tag(zsim)

#     zsim = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
#     zsnap_tag = get_zsnap_tag(zsim)

#     abs_mag_lim = get_abs_mag_lim(zmin, band)
#     if band=="Mr":
#         abs_mag_lim_tag = get_Mr_tag(abs_mag_lim)
#     elif band=="Mz":
#         abs_mag_lim_tag = get_Mz_tag(abs_mag_lim)
#     elif band=="MW1":
#         abs_mag_lim_tag = get_MW1_tag(abs_mag_lim)

#     #-- load linear fit mag bin chisq data
#     fname = f"{BASEDIR}/chisq/{rp_use_tag}/{d}_{sham_tag}_{abs_mag_lim_tag}_{zsnap_tag}.npy"
#     chisq_linear = np.load(fname, allow_pickle=True).item()

#     #-- parse mag bin keys
#     mag_bin_tags = list(chisq_linear.keys())

#     #-- define mag bin centers
#     mag_bin_edges = [-float(j[1:].replace("p",".")) for j in np.array([i.split("-") for i in mag_bin_tags]).T[1]]
#     mag_bin_cens  = [round(np.mean(mag_bins_clust[k:k+2]),3) for k in range(len(mag_bins_clust)-1)]

#     #-- for each mag bin extract sigma_const(mag) and sigma_const_los(mag) of minimum N values of chisq
#     for idx,bin_tag in enumerate(mag_bin_tags):
#             bin_data = chisq_linear[bin_tag][0]
#             bin_data.sort("chisq")
#             chunk = bin_data[:N_chisq]
#             if len(chunk) > 0:
#                 chunk.add_column(Column(data=[d for j in range(N_chisq)], name="field"), index=0)
#                 chunk.add_column(Column(zsim*np.ones(N_chisq), name="zsim"), index=1)
#                 chunk.add_column(Column(data=[band for j in range(N_chisq)], name="band"), index=2)
#                 chunk.add_column(Column(mag_bin_cens[idx]*np.ones(N_chisq), name="mag"), index=3)
#                 out.append(chunk)

#     out = vstack(out)
#     out["sigma_los"] = out["sigma_los"].astype(int)

#     tmp = np.round(np.array(out["sigma_const"]),2)
#     out.replace_column("sigma_const",tmp)

#     tmp = np.round(np.array(out["chisq"]),1)
#     out.replace_column("chisq",tmp)
    
#     return out



# def _get_popt(band, zmin, N_chisq=1, mag_bin_cens=None, rp_use_range=(0.1,None), sim_tag="mdpl2", quiet=True, plot=False):
#     assert(sim_tag != None)
#     # if not quiet:
#     #     print(f"zrange: ({zmin}, {zmin+0.1})")
#     #     print(f"band: {band}")
#     #     print(f"N min chi^2: {N_chisq}")
#     #     print(f"{rp_use_range}\n")
#     popt = {}
#     scatter_key = f"{band}_zmin{str(zmin).replace('.','p')}_chisq{N_chisq}"

#     kwargs = dict(band=band, zmin=zmin, N_chisq=N_chisq, sim_tag=sim_tag, rp_use_range=rp_use_range)
#     out  = _compile_chisq(**kwargs)
#     mask = (out["band"]==band) & (np.round(out["zsim"],1)==zmin)
#     zsim = np.round(np.unique(out["zsim"][mask]),3)[0]

#     t = []
#     for mag in np.unique(out["mag"][mask]):
#         t.append(out[mask & (out["mag"]==mag)])
#     t = vstack(t)

#     if plot:
#         fig, axes = plt.subplots(1, 2, figsize=(12,5.5))

#         ax = axes[0]
#         ax.set_xlim(0.0,1.0)
#         ax.set_xlabel("sigma SHAM (mag)")
#         ax.set_ylim(0,125)
#         ax.set_ylabel("sigma LOS (Mpc/$h$)")
#         ax.grid()
#         colors = get_colors(len(np.unique(t["mag"])))

#     dd = []
#     colors = get_colors(4)
#     for color,mag in zip(colors,np.unique(t["mag"])):
#         mask = t["mag"]==mag
#         if plot:
#             s = 100
#             ax.scatter(np.mean(t["sigma_const"][mask]), np.mean(t["sigma_los"][mask]), color=color, label=f"{band} = {mag}", s=100)
#         sig_v   = np.mean(t["sigma_const"][mask])
#         sig_los = np.mean(t["sigma_los"][mask])
#         w = np.mean(1/t["chisq"][mask])
#         dd.append([mag, sig_v, sig_los, w])

#     dd = np.array(dd).T
#     if not mag_bin_cens==None:
#         assert(len(mag_bin_cens)==4)
#         dd[0] = mag_bin_cens

#     if plot:
#         ax.legend(frameon=True)

#     mag_array = np.linspace(np.min(dd[0])-1,np.max(dd[0])+1,100)

#     if plot:
#         ax = axes[1]
#         ax.set_xlim(np.min(dd[0])-1, np.max(dd[0])+1)
#         ax.set_xlabel(band)
#         ax.set_ylim(axes[0].get_xlim())
#         ax.set_ylabel(axes[0].get_xlabel())
#         ax.scatter(dd[0], dd[1], color="red")
 
#     #-- tanh fit only possible if optimal scatter values for lowest three amgnitude bins *NOT* equal
#     if ( (dd[1][1] != dd[1][2]) | (dd[1][1] != dd[1][3]) ) & (dd[1][0] < dd[1][1]):
#         popt_sham_tanh,_ = curve_fit( sig_tanh, dd[0], dd[1], bounds=((0,0,-25),(1,2,-20)) )
#         if plot:
#             ax.plot(mag_array, sig_tanh(mag_array, *popt_sham_tanh), color="red", label="sigma SHAM tanh (4 bins)")
#         popt["sham_tanh"] = popt_sham_tanh

#     popt_sham_lin3,_  = curve_fit( sig_lin, dd[0][1:], dd[1][1:] )
#     popt["sham_lin3"] = popt_sham_lin3

#     popt_sham_lin4,_  = curve_fit( sig_lin, dd[0], dd[1] )
#     popt["sham_lin4"] = popt_sham_lin4

#     if plot:
#         ax.plot(mag_array, sig_lin(mag_array, *popt_sham_lin4), color="k", ls="--", label="sigma SHAM linear (4 bins)")
#         ax.legend(loc=2)
#         ax = axes[1].twinx()
#         ax.set_ylim(axes[0].get_ylim())
#         ax.set_ylabel(axes[0].get_ylabel())
#         ax.scatter(dd[0], dd[2], marker="x", color="black")

#     #-- linear fit sigma_LOS(mag) to 3 dimmest mag bins
#     popt_los_lin3,_  = curve_fit( sig_lin, dd[0][1:], dd[2][1:] )
#     popt["los_lin3"] = popt_los_lin3
#     #-- linear fit sigma_LOS(mag) to all 4 mag bins
#     popt_los_lin4,_  = curve_fit( sig_lin, dd[0], dd[2] )
#     popt["los_lin4"] = popt_los_lin4
#     #-- single sigma_LOS value for each mag bin (excluding brightest)
#     popt["los_binned3"] = np.mean(dd[2][1:])
#     #-- single sigma_LOS value for each mag bin (excluding brightest)
#     popt["los_binned4"] = np.mean(dd[2])
    
#     if plot:
#         # ax.plot(mag_array, sig_lin(mag_array, *popt_los_lin3), color="gray", lw=8, alpha=0.4, label="sigma LOS linear (3 bins)")
#         ax.plot(mag_array, sig_lin(mag_array, *popt_los_lin4), color="black", lw=1.5, ls=":", label="sigma LOS linear (4 bins)")

#         ax.legend(loc=4, markerfirst=False)
#         ax.text(get_corners(ax)["upper_left"][0], 90, r"$z_{\rm sim}=\ $" + f"${zsim:.3f}$\n" + r"${\rm model}$ " + f"${N_chisq}$", ha="left", va="top", fontsize=20)
#         plt.tight_layout()

#     return popt



# def _model_wp(band, zmin,
#               cat                = None,
#               d                  = "south",
#               DS                 = 1,
#               n_iter             = 1,
#               nthreads           = 2,
#               BASEDIR            = None,
#               return_model_error = False,
#               sham_tag           = "vpeak",
#               sim_tag            = None,
#               cat_tag            = None,
#               zmag_tag           = "zmaglim20p7",
#               boxsize            = None,
#               abs_mag_lim_tag    = None,
#               abs_mag_bin_tags   = None,
#               pimax              = 150.,
#               N_chisq            = 1,
#               rp_use_range       = None,
#               bin_file           = None,
#               sham_scatter_key   = None,
#               los_scatter_key    = None,
#               quiet              = True,
#              ):
#     assert(sim_tag != None)
    
#     z_snaps,_,_ = get_zsnap_data(sim_tag)
#     zsim      = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
#     zsnap_tag = get_zsnap_tag(zsim)

#     mock_scatter_tag = f"{sham_scatter_key.replace('_','-')}_chisq{N_chisq}"
#     rp_use_tag       = get_rp_use_tag(rp_use_range)
#     popt             = _get_popt(band, zmin, N_chisq=N_chisq, rp_use_range=rp_use_range, sim_tag=sim_tag)

#     if cat==None:
#         f = f"{BASEDIR}/mocks/{sim_tag}/{sham_tag}/{d}/{zsnap_tag}_{zmag_tag}_{abs_mag_lim_tag}_{mock_scatter_tag}_{rp_use_tag}"

#         galcat_fname = f"{f}_galcat_LRG-flagged.npy"
#         if not os.path.exists(galcat_fname):
#             galcat_fname = f"{f}.npy"
#         if not quiet:
#             print(f"Loading {galcat_fname}...\n")

#         cat = Table(np.load(galcat_fname))
#     cat = cat[cat["galaxy"]==True]

#     #-- compute sigma_los for specified parameterization
#     if los_scatter_key==None:
#         sigma_los = np.zeros(len(cat))
#     elif "los_binned" in los_scatter_key:
#         sigma_los = (popt[los_scatter_key])*np.ones(len(cat))
#     else:
#         sigma_los = sig_lin(cat[band].data, *popt[los_scatter_key])
#     sigma_los[np.where(sigma_los < 0)[0]] = 0

#     if abs_mag_bin_tags != None:
#         model_mags = cat[band].data
#         to_return  = {}
#         for i,tag in enumerate(abs_mag_bin_tags):
#             mag_min = -np.float(tag.split("-")[0].split("n")[-1].replace("p","."))
#             mag_max = -np.float(tag.split("-")[-1].split("n")[-1].replace("p","."))
#             # print(mag_min, mag_max)
#             mag_mask = (model_mags > mag_min) & (model_mags <= mag_max)
#             out = []
#             for (u,v,w) in (("x","y","z"),("y","z","x"),("z","x","y")):
#                 xx = cat[u]
#                 yy = cat[v]
#                 zz = cat[w]
#                 z_scatter = np.array([np.random.normal(zz, sigma_los, len(zz)) for idx in range(n_iter)])
#                 z_scatter[z_scatter < 0] = np.array(boxsize - np.abs(z_scatter[z_scatter < 0])%boxsize)
#                 z_scatter[z_scatter > boxsize] = np.array(z_scatter[z_scatter > boxsize]%boxsize)
                
#                 args = (boxsize, pimax, nthreads, bin_file, xx[mag_mask][::DS], yy[mag_mask][::DS])
#                 out.append([wp_corrfunc(*args, z_s[mag_mask][::DS], output_rpavg=False)["wp"] for z_s in z_scatter])

#             out = np.concatenate(out).T

#             wp_mean = np.mean(out, axis=1)
#             std     = np.std(out, axis=1)
#             model   = Table([rp_mids,wp_mean,std], names=("rp_cen","wp","wp_err"))
            
#             if return_model_error:
#                 to_return[tag] = model
#             else:
#                 to_return[tag] = wp_mean
#         return to_return
#     else:
#         out = []
#         for (u,v,w) in (("x","y","z"),("y","z","x"),("z","x","y")):
#             xx = cat[u][::DS]
#             yy = cat[v][::DS]
#             zz = cat[w][::DS]
#             z_scatter = np.array([np.random.normal(zz, sigma_los[::DS], len(zz)) for idx in range(n_iter)])
#             z_scatter[z_scatter < 0] = np.array(boxsize - np.abs(z_scatter[z_scatter < 0])%boxsize)
#             z_scatter[z_scatter > boxsize] = np.array(z_scatter[z_scatter > boxsize]%boxsize)

#             out.append([wp_corrfunc(boxsize, pimax, nthreads, bin_file, xx, yy, z_s, output_rpavg=False)["wp"] for z_s in z_scatter])

#         out = np.concatenate(out).T

#         wp_mean = np.mean(out, axis=1)
#         std     = np.std(out, axis=1)
#         model   = Table([rp_mids,wp_mean,std], names=("rp_cen","wp","wp_err"))

#         if return_model_error:
#             return model
#         else:
#             return wp_mean
        
