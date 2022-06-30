import numpy as np

import os, sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from params import BASEDIR, DATADIR, get_zsnap_data, H0, Om0
zkcorrs,_,_ = get_zsnap_data("bolshoip")

from astropy.table import Table, Column, hstack, vstack, join
from astropy.io import fits, ascii
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
from astropy.constants import c as c_speed
from astropy import units as u

from datetime import datetime

from functions import sph_to_cart, load_cat, load_kcorr, get_patch_subs, get_survey_area #, make_cat, fmt_kcorr_idl
from masks import masks_from_maskbits, masks_from_fitbits, desi_lrg_mask, desi_lrg_mask_optical, get_coord_mask, cat_geo_mask, extra_masks

from params import get_abs_mag_lim, get_abs_mag_bins_clust
from utils import *

import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.wp import wp as wp_corrfunc
from Corrfunc.utils import convert_rp_pi_counts_to_wp



def run_cf( config_file, **cmd_kwargs ):
    band     = cmd_kwargs["band"]
    zrange   = cmd_kwargs["zrange"]
    quiet    = cmd_kwargs["quiet"]

    zmin, zmax = np.round(zrange,1)
 
    #-- read and parse configuration file
    with open(config_file, "r") as f:
        lines = f.readlines()
        f.close()
        
    kwargs = {}
    kwargs["quiet"] = quiet

    for l in lines:
        if (l != "\n") & ("#--" not in l) & (" = " in l):
            entry = [i.strip() for i in l.strip().split("=")]
            if entry[0][0] != "#":
                key, value = entry
                if key=="zrange":
                    value = value.split(",")
                    value = (float(value[0].strip("(")),float(value[1].strip(")")))
                else:
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    if ("True" in value):
                        value = True
                    elif ("False" in value):
                        value = False
                    elif ("None" in value):
                        value = None
                    elif ("." in value):
                        if ("[" in value) & ("]" in value):
                            value = value.strip("[").strip("]").split(",")
                            value = np.array( [float(i) for i in value] )
                        else:
                            value = float(value)
                    elif ("\"" not in value) & ("\'" not in value):
                        value = int(value)
                if type(value)==str:
                    value = value.strip("\"")
                if key=="d":
                    value = value.lower()
                kwargs[key] = value

    d = kwargs["d"]

    kwargs["zrange"] = zrange
    kwargs["band"]   = band

    mag_bins         = kwargs["mag_bins"]  # True or False
    n_mag_bins_clust = kwargs["n_mag_bins_clust"]  # only relevant if mag_bins==True
    if not mag_bins:
        abs_mag_bins = None
    #-- define magnitude bins for wp (if needed)
    else:
        abs_mag_bins = get_abs_mag_bins_clust(zmin, band, nbins=n_mag_bins_clust)
    kwargs["abs_mag_bins"] = abs_mag_bins  # None or array

    nbins  = kwargs["nbins"]
    rp_min = kwargs["rp_min"]
    rp_max = kwargs["rp_max"]
    
    #-- define rp bin centers (rp_mids) from rp_bins
    rp_bins = np.logspace( np.log10(rp_min), np.log10(rp_max), nbins + 1 )
    rp_mids = np.array([ 10**np.mean(np.log10(rp_bins)[i:i+2]) for i in np.arange(len(rp_bins)-1) ])
    
    kwargs["rp_bins"] = rp_bins
    kwargs["rp_mids"] = rp_mids
    
    kwargs["BASEDIR"] = BASEDIR
    kwargs["DATADIR"] = DATADIR
   
    kwargs["Mr_lim"]  = get_abs_mag_lim(zmin, "Mr") if (band=="Mr") else None
    kwargs["Mz_lim"]  = get_abs_mag_lim(zmin, "Mz") if (band=="Mz") else None
    kwargs["MW1_lim"] = get_abs_mag_lim(zmin, "MW1") if (band=="MW1") else None
    Mr_lim  = kwargs["Mr_lim"] 
    Mz_lim  = kwargs["Mz_lim"] 
    MW1_lim = kwargs["MW1_lim"] 

    debug         = kwargs["debug"]
    plot_zdists   = kwargs["plot_zdists"]
    #quiet         = kwargs["quiet"]

    zphot_tag     = kwargs["zphot_tag"]
    zmag_lim      = kwargs["zmag_lim"]
    extra_mask    = kwargs["extra_mask"]
    pimax         = kwargs["pimax"]
    rand_coeff    = kwargs["rand_coeff"]
    autocorr      = kwargs["autocorr"]
    cosmology     = kwargs["cosmology"]
    nthread       = kwargs["nthread"]

    do_GXG    = kwargs["do_GXG"]
    do_LXL    = kwargs["do_LXL"]
    do_LXLopt = kwargs["do_LXLopt"]
    do_LXnL   = kwargs["do_LXnL"]
    do_LXG    = kwargs["do_LXG"]

    print("")
    for k in kwargs.keys():
        print("{0:12} = {1}".format(k, kwargs[k]))
    
    #-- parameter checks
    if (Mr_lim != None):
        assert( (Mz_lim==None) & (MW1_lim==None) )
    elif (Mz_lim != None):
        assert( (Mr_lim==None) & (MW1_lim==None) )
    elif (MW1_lim != None):
        assert( (Mr_lim==None) & (Mz_lim==None) )
    # raise Exception("Mr_lim and MW1_lim cannot both be set; at least one must be None")

    if (not do_GXG) & (not do_LXL) & (not do_LXLopt) & (not do_LXnL) & (not do_LXG):
        raise Exception("No correlation functions to calculate!")
    
    # raise Exception()
    
    #-- optional down-sampling factor if debugging
    DS = 20 if debug else 1
    if debug and not quiet:
            print("########## RUNNING IN DEBUGGING MODE ##########")

    #-- more parameter checks and housekeeping
    for p in (BASEDIR,DATADIR):
        assert(os.path.exists(p))
    assert(len(zrange)==2)
    zmin, zmax = zrange[0], zrange[1]
    assert(zmax > zmin)
    zkcorr = zkcorrs[int(10*zrange[0])]

    assert( (d.lower()=="north") | (d.lower()=="south") )
    assert(zmag_lim <= 21.0)
    cat_tag = "z{:.2f}-{:.2f}_{}".format(zmin,zmax,d)
    cat_tag = cat_tag.replace(".","p")
    
    #-- load data catalog
    cat = load_cat(d=d, zrange=zrange, zphot_tag=zphot_tag)
    
    #-- add geometry mask to catalog
    if not quiet:
        print("Getting data geometry mask and jackknife patches...")
    cat, geo_mask = cat_geo_mask(cat, plot=False)

    #-- DESI LRG mask
    if not quiet:
        print("Getting DESI LRG mask...")
    lrg_mask = desi_lrg_mask(cat, d=d, zrange=zrange, zkcorr=zkcorr, quiet=True)

    #-- DESI optical LRG mask
    if not quiet:
        print("Getting DESI optical LRG mask...")
    lrg_mask_opt = desi_lrg_mask_optical(cat, d=d, zrange=zrange, zkcorr=zkcorr, quiet=True)


    #-- get redshift mask for unpadded catalogs
    if not quiet:
        print("Getting redshift mask...")
    zphot = cat[zphot_tag].data
    zphot_mask = (zphot >= zmin) & (zphot < zmax)
    
    #-- get mask for apparent z-band magnitude cut
    mz = 2.5*( 9 - np.log10(cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
    zmag_mask = (mz <= zmag_lim)
    # if extra_mask=="in_red_seq":
    #     zmag_mask = zmag_mask & reu_masks(cat, d=d, zrange=zrange, zkcorr=zkcorr, quiet=quiet, in_red_seq=True, out_red_seq=False, zphot_tag=zphot_tag)
    # elif extra_mask=="out_red_seq":
    #     zmag_mask = zmag_mask & reu_masks(cat, d=d, zrange=zrange, zkcorr=zkcorr, quiet=quiet, in_red_seq=False, out_red_seq=True, zphot_tag=zphot_tag)
    # if (extra_mask=="red_seq_Mr_vs_g-r_z0p4-0p5") | (extra_mask=="red_seq_MW1_vs_g-W1_z0p4-0p5"):
    if (extra_mask != None):
         zmag_mask = zmag_mask & extra_masks(cat, d=d, zrange=zrange, mask_name=extra_mask)

    zmag_tag = str(zmag_lim).replace(".","p")
    
    #-- optional absolute magnitude binning
    if (type(abs_mag_bins) != type(None)):
        mr  = 2.5*( 9 - np.log10(cat["FLUX_R"]/cat["MW_TRANSMISSION_R"]) )
        mz  = 2.5*( 9 - np.log10(cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
        mW1 = 2.5*( 9 - np.log10(cat["FLUX_W1"]/cat["MW_TRANSMISSION_W1"]) )
        _,Kr,Kz,KW1 = load_kcorr(d=d, zrange=zrange, zkcorr=zkcorr, quiet=quiet)[:4]
        if not quiet:
            print("Estimating absolute magnitudes from photo-zs...")

        DM  = cosmo.distmod(zphot).value
        Mr  = mr - DM - Kr
        Mz  = mz - DM - Kz
        MW1 = mW1 - DM - KW1
    try:
        N_mag_bins = len(abs_mag_bins)
    except:
        N_mag_bins = 2

    #-- get mask for (OPTIONAL) absolute r-band OR z-band OR W1-band magnitude cut
    if (type(Mr_lim)==float):
        assert((Mz_lim==None) & (MW1_lim==None))
        try:
            Mr
        except:
            mr = 2.5*( 9 - np.log10(cat["FLUX_R"]/cat["MW_TRANSMISSION_R"]) )
            Kr = load_kcorr(d=d, zrange=zrange, zkcorr=zkcorr, quiet=quiet)[1]
            Mr = mr - cosmo.distmod(zphot).value - Kr
        abs_mag_lim_mask = ( Mr <= Mr_lim )
        abs_mag_lim_tag  = get_Mr_tag( Mr_lim )
    elif (type(Mz_lim)==float):
        assert((Mr_lim==None) & (MW1_lim==None))
        try:
            Mz
        except:
            mz = 2.5*( 9 - np.log10(cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
            Kz = load_kcorr(d=d, zrange=zrange, zkcorr=zkcorr, quiet=quiet)[1]
            Mz = mz - cosmo.distmod(zphot).value - Kz
        abs_mag_lim_mask = ( Mz <= Mz_lim )
        abs_mag_lim_tag  = get_Mz_tag( Mz_lim )
    elif (type(MW1_lim)==float):
        assert((Mr_lim==None) & (Mz_lim==None))
        try:
            MW1
        except:
            mW1 = 2.5*( 9 - np.log10(cat["FLUX_W1"]/cat["MW_TRANSMISSION_W1"]) )
            KW1 = load_kcorr(d=d, zrange=zrange, zkcorr=zkcorr, quiet=quiet)[3]
            MW1 = mW1 - cosmo.distmod(zphot).value - KW1
        abs_mag_lim_mask = ( MW1 <= MW1_lim )
        abs_mag_lim_tag  = get_MW1_tag( MW1_lim )
    else:
        abs_mag_lim_mask = np.ones(len(zphot), dtype=bool)
        abs_mag_lim_tag  = "no_abs_mag_cut"
        
    #-- down-sample data vectors if debugging    
    if debug:
        abs_mag_lim_mask = abs_mag_lim_mask[::DS]

    print(band, abs_mag_lim_tag, zrange)
    #raise Exception()
    
    #-- clean up: define only needed fields from data catalog and delete the rest from memory
    gal_RA    = cat["RA"].data
    gal_DEC   = cat["DEC"].data
    gal_PATCH = cat["PATCH"].data
    gal_Z     = zphot
    gal_CZ    = cosmo.comoving_distance( zphot ).value

    patch_nums = np.unique( cat[geo_mask]["PATCH"] )
    N_patches  = len(patch_nums)

    data = Table( [gal_RA, gal_DEC, gal_CZ, gal_PATCH], names=("RA","DEC","CZ","PATCH") )
    del cat

    #-- down-sample data vectors if debugging    
    if debug:
        data      = data[::DS]
        zphot     = zphot[::DS]
        gal_Z     = gal_Z[::DS]
        gal_PATCH = gal_PATCH[::DS]
        
        geo_mask     = geo_mask[::DS]
        lrg_mask     = lrg_mask[::DS]
        lrg_mask_opt = lrg_mask_opt[::DS]
        zphot_mask   = zphot_mask[::DS]
        zmag_mask    = zmag_mask[::DS]

        patch_nums = patch_nums[:5]
        N_patches  = len(patch_nums)

        if (type(abs_mag_bins) != type(None)):
            MW1 = MW1[::DS]
            Mr  = Mr[::DS]

    #-- load and mask randoms, then clean up
    if not quiet:
        print("Loading randoms...")
    if d=="north":
        rand = np.load(f"{DATADIR}/random/north_full.npy")
    elif d=="south":
        rand = vstack([ Table(np.load(f"{DATADIR}/random/south_{i}.npy")) for i in np.arange(rand_coeff) ])
    rand_geo_mask = ~(rand["BITMASK"] | rand[f"COORDMASK"])
    
    rand_RA    = rand["RA"][rand_geo_mask].data
    rand_DEC   = rand["DEC"][rand_geo_mask].data
    rand_PATCH = rand["PATCH"][rand_geo_mask].data

    random = Table( [rand_RA, rand_DEC, rand_PATCH], names=("RA","DEC","PATCH") )
    del rand

    #-- down-sample randoms if debugging
    if debug:
        random     = random[::DS]
        rand_PATCH = random["PATCH"].data

    #-- set up colors, redshift bins for histograms, and figure objects if plotting data and random redshift distributions
    cm = plt.get_cmap("Dark2")
    colors = [cm(c) for c in np.arange(N_mag_bins)/N_mag_bins]

    dz = 0.005
    zphot_bins = np.concatenate([np.arange(zmin-dz,np.min(zphot[geo_mask & zmag_mask])-dz,-dz)[::-1],
                                 np.arange(zmin,zmax,dz),
                                 np.arange(zmax,np.max(zphot[geo_mask & zmag_mask])+dz,dz)])
    zphot_bins = np.unique( [round(i,3) for i in zphot_bins] )


    #-- loop over OPTIONAL absolute R/Z/W1-band magnitude bins (single execution if no binning)
    for mag_idx in np.arange(N_mag_bins)[:-1]:
        if (type(abs_mag_bins) != type(None)):
            mag_min, mag_max = abs_mag_bins[mag_idx], abs_mag_bins[mag_idx + 1]
            if not quiet:
                print(f"Absolute magnitude bin range: ({mag_min},{mag_max})")
            #-- if R-band absolute magnitude bins
            if (type(Mr_lim)==float):
                if not quiet:
                    print(f"Binning on R-band absolute magnitude")
                    print(f"This bin range: ({mag_min},{mag_max})")
                abs_mag_bin_tag  = f"Mrn{np.abs(mag_min)}-n{np.abs(mag_max)}"
                abs_mag_bin_tag  = abs_mag_bin_tag.replace(".","p")
                abs_mag_bin_mask = (Mr >= mag_min) & (Mr < mag_max)
            #-- if Z-band absolute magnitude bins
            if (type(Mz_lim)==float):
                if not quiet:
                    print(f"Binning on Z-band absolute magnitude")
                    print(f"This bin range: ({mag_min},{mag_max})")
                abs_mag_bin_tag  = f"Mrn{np.abs(mag_min)}-n{np.abs(mag_max)}"
                abs_mag_bin_tag  = abs_mag_bin_tag.replace(".","p")
                abs_mag_bin_mask = (Mz >= mag_min) & (Mz < mag_max)
            #-- if W1-band absolute magnitude bins
            elif (type(MW1_lim)==float):
                if not quiet:
                    print(f"Binning on W1-band absolute magnitude")
                    print(f"This bin range: ({mag_min},{mag_max})")
                abs_mag_bin_tag  = f"MW1n{np.abs(mag_min)}-n{np.abs(mag_max)}"
                abs_mag_bin_tag  = abs_mag_bin_tag.replace(".","p")
                abs_mag_bin_mask = (MW1 >= mag_min) & (MW1 < mag_max)
        #-- if NOT binning by absolute magnitude
        else:
            if not quiet:
                print(f"NO absolute magnitude binning")
            mag_min, mag_max = None, None
            abs_mag_bin_tag  = "no_abs_mag_bins"
            abs_mag_bin_mask = np.ones(len(zphot), dtype=bool)
            if (extra_mask != None):
                abs_mag_bin_tag = str(extra_mask)
                if not quiet:
                    print(f"Additional mask: {str(extra_mask)}")
    
        abs_mag_mask = abs_mag_lim_mask & abs_mag_bin_mask
        
        #-- MASKS
        # geo_mask         # masks bad sky regions (data)
        # zphot_mask       # strict redshift bin cut (vs padded)
        # zmag_mask        # OPTIONAL z-mag cut beyond mz < 21 OR misc mask
        # abs_mag_lim_mask # OPTIONAL absolute R-band or Z-band or W1-band magnitude cut
        # abs_mag_bin_mask # OPTIONAL masks for absolute magnitude bins
        # abs_mag_mask     # defined to be abs_mag_lim_mask & abs_mag_bin_mask
        # lrg_mask         # DESI LRG selection function (default)
        # lrg_mask_opt     # DESI optical LRG selection function
        # patch_mask       # for data jk samples (set below)
    
        # rand_geo_mask    # geo_mask for randoms 
        # rand_patch_mask  # for random jk samples (set below)

        rp_pi_counts_by_patch = {}
        if plot_zdists:
            figures = {}
        
        kwargs = dict(  N_patches       = N_patches,
                        patch_nums      = patch_nums,
                        random          = random,
                        gal_PATCH       = gal_PATCH,
                        geo_mask        = geo_mask,
                        d               = d,
                        zmag_mask       = zmag_mask,
                        abs_mag_mask    = abs_mag_mask,
                        zphot_mask      = zphot_mask,
                        lrg_mask        = lrg_mask,
                        lrg_mask_opt    = lrg_mask_opt,
                        data            = data,
                        rand_coeff      = rand_coeff,
                        gal_Z           = gal_Z,
                        zphot           = zphot,
                        plot_zdists     = plot_zdists,
                        zphot_bins      = zphot_bins,
                        abs_mag_bins    = abs_mag_bins,
                        mag_min         = mag_min,
                        mag_max         = mag_max,
                        zmag_lim        = zmag_lim,
                        mag_idx         = mag_idx,
                        do_GXG          = do_GXG,
                        do_LXG          = do_LXG,
                        do_LXL          = do_LXL,
                        do_LXLopt       = do_LXLopt,
                        do_LXnL         = do_LXnL,
                        quiet           = quiet,
                        colors          = colors,
                        autocorr        = autocorr,
                        cosmology       = cosmology,
                        pimax           = pimax,
                        nthread         = nthread,
                        rp_bins         = rp_bins,)

        for patch_idx in range(0,N_patches):
            if plot_zdists:
                patch, rp_pi_counts_this_patch, fig_this_patch = _rp_pi_counts_by_patch(patch_idx, **kwargs)
                figures[ patch ] = fig_this_patch
            else:
                patch, rp_pi_counts_this_patch = _rp_pi_counts_by_patch(patch_idx, **kwargs)
            rp_pi_counts_by_patch[ patch ] = rp_pi_counts_this_patch

        patches = list( rp_pi_counts_by_patch.keys() )
        pair_set_keys = list( rp_pi_counts_by_patch[ patches[0] ].keys() )
        
        rp_pi_counts_by_cross = {}
        for cross in pair_set_keys:
            rp_pi_counts_by_patch_this_cross = {}
            for patch in patches:
                rp_pi_counts_this_patch = rp_pi_counts_by_patch[ patch ][ cross ]
                rp_pi_counts_by_patch_this_cross[ patch ] = rp_pi_counts_this_patch
            rp_pi_counts_by_cross[ cross ] = rp_pi_counts_by_patch_this_cross

            #-- save (rp,pi) pair counts by cross
            out_path  = f"{BASEDIR}/clustering/zmaglim{zmag_tag}/{abs_mag_lim_tag}/{abs_mag_bin_tag}/rp_pi_counts/"
            Path( out_path ).mkdir(parents=True, exist_ok=True)
            out_fname = f"{cat_tag}_{cross}.npy"
            if not quiet:
                print(f"Saving {out_path + out_fname}...")
            np.save(out_path + out_fname, rp_pi_counts_by_cross, allow_pickle=True)

        #-- compute and save wp(rp) and error from (rp,pi) counts
        for cross in list( rp_pi_counts_by_cross.keys() ):
            save_path =  f"{BASEDIR}/clustering/zmaglim{zmag_tag}/{abs_mag_lim_tag}/{abs_mag_bin_tag}/wp/"
            fname = f"{cat_tag}_{cross}"
            wp_from_rp_pi( rp_pi_counts_by_cross[ cross ], save_path=save_path, fname=fname, rp_mids=rp_mids, pimax_array=np.arange(80.,160.,10.), quiet=quiet)


    if plot_zdists:
        plt.tight_layout()
        figdir = f"{BASEDIR}/figures/jk_redshift_distributions/zmaglim{zmag_tag}/"
        Path( figdir ).mkdir(parents=True, exist_ok=True)
        if not quiet:
            print("Saving redshift distribution plots...")
        pdf = matplotlib.backends.backend_pdf.PdfPages( figdir + f"{cat_tag}_{abs_mag_lim_tag}_{abs_mag_bin_tag}.pdf" )
        for key in figures.keys():
            this_fig = figures[key]
            this_fig.subplots_adjust(hspace=0.05)
            pdf.savefig( this_fig, bbox_inches="tight", pad_inches=0.1 )
        pdf.close()
    
    if not quiet:
        print("DONE!")




def _get_sph(DD):        
    RA1, DEC1, CZ1 = DD[0]["RA"], DD[0]["DEC"], DD[0]["CZ"]
    RA2, DEC2, CZ2 = DD[1]["RA"], DD[1]["DEC"], DD[1]["CZ"]
    return RA1, DEC1, CZ1.astype("<f8"), RA2, DEC2, CZ2.astype("<f8")



def _rp_pi_counts_by_patch(patch_idx, patch_nums=None, N_patches=None, random=None, gal_PATCH=None, geo_mask=None, d=None, zmag_mask=None, abs_mag_mask=None, zphot_mask=None, lrg_mask=None, lrg_mask_opt=None, data=None, rand_coeff=20, gal_Z=None, zphot=None, plot_zdists=True, figures=None, zphot_bins=None, abs_mag_bins=None, mag_min=None, mag_max=None, zmag_lim=None, mag_idx=None, do_GXG=True, do_LXG=True, do_LXL=True, do_LXLopt=True, do_LXnL=False, quiet=False, colors=None, autocorr=0, cosmology=2, nthread=2, pimax=None, rp_bins=None):
    patch = patch_nums[ patch_idx ]
    if not quiet:
        print("\n======= PATCH {0:3d} ({1:1d} of {2:1d}) =======".format(patch, patch_idx+1, N_patches))
    
    random_patch = random[ random["PATCH"]==patch ]
    Nrand_patch  = len(random_patch)
    
    patch_mask   = gal_PATCH==patch

    # print(f"geo_mask     : {len(geo_mask)}")
    # print(f"patch_mask   : {len(patch_mask)}")
    # print(f"zmag_mask    : {len(zmag_mask)}")
    # print(f"abs_mag_mask : {len(abs_mag_mask)}")
    # print(f"zphot_mask   : {len(zphot_mask)}")

    gal_mask     = geo_mask & patch_mask & zmag_mask & abs_mag_mask & zphot_mask
    gal_mask_pad = geo_mask & patch_mask & zmag_mask & abs_mag_mask


    #-- GALAXIES
    G   = np.array( data[ gal_mask ] )
    Gp  = np.array( data[ gal_mask_pad ] )
    NG  = len( G )
    NGp = len( Gp )
    
    if (NG > 0):
        #-- GALAXY RANDOMS
        NRG      = np.min( [rand_coeff*NG, Nrand_patch] )
        RG       = random_patch[:NRG]
        randG_Z  = np.random.choice( gal_Z[ gal_mask ], NRG )
        randG_CZ = cosmo.comoving_distance( randG_Z ).value
        RG.add_column( Column(randG_CZ, name="CZ") )
    
        #-- PADDED GALAXY RANDOMS
        NRGp      = np.min( [rand_coeff*NGp, Nrand_patch] )
        RGp       = random_patch[:NRGp]
        randGp_Z  = np.random.choice( gal_Z[ gal_mask_pad ], NRGp )
        randGp_CZ = cosmo.comoving_distance( randGp_Z ).value
        RGp.add_column( Column(randGp_CZ, name="CZ") )
    
    
        #-- LRGs
        L   = np.array( data[ gal_mask & lrg_mask ] )
        Lp  = np.array( data[ gal_mask_pad & lrg_mask ] )
        NL  = len( L )
        NLp = len( Lp )
    
        #-- LRG RANDOMS
        NRL      = np.min( [rand_coeff*NL, Nrand_patch] )
        RL       = random_patch[:NRL]
        randL_Z  = np.random.choice( gal_Z[ gal_mask & lrg_mask ], NRL )
        randL_CZ = cosmo.comoving_distance( randL_Z ).value
        RL.add_column( Column(randL_CZ, name="CZ") )
    
        #-- PADDED LRG RANDOMS
        NRLp      = np.min( [rand_coeff*NLp, Nrand_patch] )
        RLp       = random_patch[:NRLp]
        randLp_Z  = np.random.choice( gal_Z[ gal_mask_pad & lrg_mask ], NRLp )
        randLp_CZ = cosmo.comoving_distance( randLp_Z ).value
        RLp.add_column( Column(randLp_CZ, name="CZ") )
    
        
        #-- optical LRGs
        Lopt   = np.array( data[ gal_mask & lrg_mask_opt ] )
        Loptp  = np.array( data[ gal_mask_pad & lrg_mask_opt ] )
        NLopt  = len( Lopt )
        NLoptp = len( Loptp )
    
        #-- optical LRG RANDOMS
        NRLopt      = np.min( [rand_coeff*NLopt, Nrand_patch] )
        RLopt       = random_patch[:NRLopt]
        randLopt_Z  = np.random.choice( gal_Z[ gal_mask & lrg_mask_opt ], NRLopt )
        randLopt_CZ = cosmo.comoving_distance( randLopt_Z ).value
        RLopt.add_column( Column(randLopt_CZ, name="CZ") )
    
        #-- PADDED optical LRG RANDOMS
        NRLoptp      = np.min( [rand_coeff*NLoptp, Nrand_patch] )
        RLoptp       = random_patch[:NRLoptp]
        randLoptp_Z  = np.random.choice( gal_Z[ gal_mask_pad & lrg_mask_opt ], NRLoptp )
        randLoptp_CZ = cosmo.comoving_distance( randLoptp_Z ).value
        RLoptp.add_column( Column(randLoptp_CZ, name="CZ") )


        #-- not-LRGs
        nL   = np.array( data[ gal_mask & ~lrg_mask ] )
        nLp  = np.array( data[ gal_mask_pad & ~lrg_mask ] )
        NnL  = len( nL )
        NnLp = len( nLp )
    
        #-- not-LRG RANDOMS
        NRnL      = np.min( [rand_coeff*NnL, Nrand_patch] )
        RnL       = random_patch[:NRnL]
        randnL_Z  = np.random.choice( gal_Z[ gal_mask & ~lrg_mask ], NRnL )
        randnL_CZ = cosmo.comoving_distance( randnL_Z ).value
        RnL.add_column( Column(randnL_CZ, name="CZ") )
    
        #-- PADDED not-LRG RANDOMS
        NRnLp      = np.min( [rand_coeff*NnLp, Nrand_patch] )
        RnLp       = random_patch[:NRnLp]
        randnLp_Z  = np.random.choice( gal_Z[ gal_mask_pad & ~lrg_mask ], NRnLp )
        randnLp_CZ = cosmo.comoving_distance( randnLp_Z ).value
        RnLp.add_column( Column(randnLp_CZ, name="CZ") )
    
    
        #-- Set up optional figures for data/random redshift distributions
        if plot_zdists:
            fig_this_patch, axes = plt.subplots(3, 2, figsize=(12,12), sharex=True)
            for ax in np.concatenate(axes):
                ax.tick_params(axis="both", which="major", labelsize=10)
                # ax.set_xlim(zphot_bins[0]-0.5*dz, zphot_bins[-1]+0.5*dz)
                # ax.set_xticks(np.arange(zmin-0.06, zmax+0.06, 0.02))
                ax.set_xlim(zphot_bins[0]-0.005, zphot_bins[-1]+0.005)
            axes[2][0].set_xlabel(r"$z_{\rm phot}$ (data)", fontsize=12)
            axes[2][1].set_xlabel("random redshifts", fontsize=12)
            
            if (type(abs_mag_bins) != type(None)):
                label = r"${:.1f} < MW1 < {:.1f}$".format(mag_min, mag_max)
            else:
                label = r"no $MW1$ cut; $m_z<{}$".format(zmag_lim)
            color = colors[mag_idx]
            
            axes[0][0].hist( zphot[gal_mask_pad], bins=zphot_bins, histtype="step", lw=2, color=color)
            axes[0][0].hist( zphot[gal_mask],     bins=zphot_bins, alpha=0.15, color=color)
            axes[0][1].hist( randGp_Z, bins=zphot_bins, histtype="step", lw=2, label=label, color=color)
            axes[0][1].hist( randG_Z,  bins=zphot_bins, alpha=0.1, color=color)
    
            axes[1][0].hist( zphot[gal_mask_pad & lrg_mask], bins=zphot_bins, histtype="step", lw=2, color=color)
            axes[1][0].hist( zphot[gal_mask & lrg_mask],     bins=zphot_bins, alpha=0.15, color=color)
            axes[1][1].hist( randLp_Z, bins=zphot_bins, histtype="step", lw=2, color=color)
            axes[1][1].hist( randL_Z,  bins=zphot_bins, alpha=0.1, color=color)
    
            axes[2][0].hist( zphot[gal_mask_pad & ~lrg_mask], bins=zphot_bins, histtype="step", lw=2, color=color)
            axes[2][0].hist( zphot[gal_mask & ~lrg_mask],     bins=zphot_bins, alpha=0.15, color=color)
            axes[2][1].hist( randnLp_Z, bins=zphot_bins, histtype="step", lw=2, color=color)
            axes[2][1].hist( randnL_Z,  bins=zphot_bins, alpha=0.1, color=color)
    
            axes[0][1].legend(fontsize=12, loc=3) #, ncol=2)
    
            if mag_idx==0:
                xmin, xmax = axes[0][0].get_xlim()
                ymin, ymax = axes[0][0].get_ylim()
                axes[0][0].text( xmin+0.05*(xmax-xmin), 0.05*ymax, f"{d} patch {patch}", ha="left", va="center", fontsize=12 )
                for row,gal_type in zip((0,1,2),("all galaxies","LRGs","non-LRGs")):
                    ymin, ymax = axes[row][0].get_ylim()
                    axes[row][0].text( xmax-0.05*(xmax-xmin), 0.95*ymax, gal_type, ha="right", va="center", fontsize=12 )
    
        
        #-- Create pair samples for passing to Corrfunc
        if do_GXG:
            #-- gal X gal_padded
            D1D2 = ( G,  Gp  )
            D1R2 = ( G,  RGp )
            D2R1 = ( Gp, RG  )
            R1R2 = ( RG, RGp )
            GXG  = (D1D2, D1R2, D2R1, R1R2)
    
            NGXG = (NG, NGp, NRG, NRGp)
            if not quiet:
                print("N_gal          = {0:8d}".format(NG))
                print("N_rand         = {0:8d}\n".format(NRG))
                print("N_gal (pad)    = {0:8d}".format(NGp))
                print("N_rand (pad)   = {0:8d}\n".format(NRGp))
        
        if do_LXG:
            #-- LRG X gal_padded
            D1D2 = ( L,  Gp )
            D1R2 = ( L,  RGp )
            D2R1 = ( Gp, RL )
            R1R2 = ( RL, RGp )
            LXG  = (D1D2, D1R2, D2R1, R1R2)
    
            NLXG = (NL, NGp, NRL, NRGp)
            if not quiet:
                print("N_LRG          = {0:8d}".format(NL))
                print("N_rand         = {0:8d}\n".format(NRL))
    
        if do_LXL:
            #-- LRG X LRG_padded
            D1D2 = ( L,  Lp )
            D1R2 = ( L,  RLp )
            D2R1 = ( Lp, RL )
            R1R2 = ( RL, RLp )
            LXL  = (D1D2, D1R2, D2R1, R1R2)
    
            NLXL = (NL, NLp, NRL, NRLp)
            if not quiet:
                print("N_LRG (pad)    = {0:8d}".format(NLp))
                print("N_rand (pad)   = {0:8d}\n".format(NRLp))
    
        if do_LXLopt:
            #-- LRG X LRG_padded
            D1D2 = ( Lopt,  Loptp )
            D1R2 = ( Lopt,  RLoptp )
            D2R1 = ( Loptp, RLopt )
            R1R2 = ( RLopt, RLoptp )
            LXLopt = (D1D2, D1R2, D2R1, R1R2)
    
            NLXLopt = (NLopt, NLoptp, NRLopt, NRLoptp)
            if not quiet:
                print("N_LRGopt (pad) = {0:8d}".format(NLoptp))
                print("N_rand (pad)   = {0:8d}\n".format(NRLoptp))

        if do_LXnL:
            #-- LRG X not-LRG_padded
            D1D2  = ( L,  nLp )
            D1R2  = ( L,  RnLp )
            D2R1  = ( nLp, RL )
            R1R2  = ( RL, RnLp )
            LXnL  = (D1D2, D1R2, D2R1, R1R2)
    
            NLXnL = (NL, NnLp, NRL, NRnLp)
            if not quiet:
                print("N_notLRG (pad) = {0:8d}".format(NnLp))
                print("N_rand (pad)   = {0:8d}\n".format(NRnLp))
    
        rp_pi_counts_this_patch = {}
        
        pair_set_list = []
        pair_set_keys = []
        if do_GXG:
            pair_set_list.append( (GXG,NGXG) )
            pair_set_keys.append( "GXG" )
        if do_LXG:
            pair_set_list.append( (LXG,NLXG) )
            pair_set_keys.append( "LXG" )
        if do_LXL:
            pair_set_list.append( (LXL,NLXL) )
            pair_set_keys.append( "LXL" )                
        if do_LXLopt:
            pair_set_list.append( (LXLopt,NLXLopt) )
            pair_set_keys.append( "LXLopt" )                
        if do_LXnL:
            pair_set_list.append( (LXnL,NLXnL) )
            pair_set_keys.append( "LXnL" )
        for n,pair_set in enumerate( pair_set_list ):
            pair_data   = pair_set[0]
            pair_totals = pair_set[1]
            if not quiet:
                print(f"Pair {n+1} of {len(pair_set_keys)}: {pair_set_keys[n]}")
            results_this_pair_set = []
            for idx,label in enumerate( ("D1D2","D1R2","D2R1","R1R2") ):
                pair = pair_data[idx]
                if not quiet:
                    print(f"  {label}:" + "{0:10d} X {1:10d}".format( len(pair[0]), len(pair[1]) ))
                RA1, DEC1, CZ1, RA2, DEC2, CZ2 = _get_sph( pair )
    
                counts_this_pair = DDrppi_mocks(autocorr, cosmology, nthread, pimax, rp_bins, RA1, DEC1, CZ1, weights1=None, RA2=RA2, DEC2=DEC2, CZ2=CZ2, weights2=None, is_comoving_dist=True, verbose=False)
                results_this_pair_set.append( (counts_this_pair, pair_totals[idx]) )
        
            rp_pi_counts_this_patch[ pair_set_keys[n] ] = results_this_pair_set
            
        if plot_zdists:
            return (patch, rp_pi_counts_this_patch, fig_this_patch)
        else:
            return (patch, rp_pi_counts_this_patch)




def wp_from_rp_pi( rp_pi_counts, save_path=None, fname=None, rp_mids=None, pimax_array=np.arange(80.,160.,10.), quiet=True):
    """Calculate wp(rp) from jackknife samples and for multiple values of pimax"""
    if type(rp_pi_counts)==str:
        assert( os.path.exists(rp_pi_counts) )
        if not quiet:
            print(f"Loading {rp_pi_counts}")
        counts = np.load(rp_pi_counts, allow_pickle=True).item()
        save_path = rp_pi_counts.split("/")
        save_path[-2] = "wp"
        fname = save_path[-1][:-4]
        save_path = "/".join(save_path[:-1]) + "/"
    else:
        counts = rp_pi_counts
        if (save_path==None) | (fname==None):
            raise Exception("\nMust specify 'save_path' and 'fname' for wp output;\nfname should end with '_{cross_type}', e.g. '_GXG', '_LXL', '_LXLopt', '_LXnL'")
        if (save_path[-1]!="/"):
            save_path = save_path + "/"
        if fname[-4:]==".txt":
            fname = fname[:-4]
    Path( save_path ).mkdir(parents=True, exist_ok=True)
    fname = fname.split("_")
    cross = fname[-1]
    fname = "_".join(fname[:-1])

    if not quiet:
        print(f"save_path = {save_path}\n")
        print(f"fname = {fname}\n")
        print(f"cross = {cross}\n")

    # patches = list(counts[cross].keys())
    patches = list(counts.keys())
    if not quiet:
        print(f"patches = {patches}\n")

    #-- sets of pair counts (DD, DR, RD, RR)
    # count_sets = [ [counts[cross][p][pair_idx][0]["npairs"] for p in patches ] for pair_idx in range(4) ]
    count_sets = [ [counts[p][pair_idx][0]["npairs"] for p in patches ] for pair_idx in range(4) ]
    jk_sets    = [ [(sum(count_sets[pair_idx])-this_set) for this_set in count_sets[pair_idx]] for pair_idx in range(4) ] 
    #-- total number counts (ND1, ND2, NR1, NR2)
    # count_set_totals = [ [counts[cross][p][pair_idx][1] for p in patches ] for pair_idx in range(4) ]
    count_set_totals = [ [counts[p][pair_idx][1] for p in patches ] for pair_idx in range(4) ]
    jk_set_totals    = [ [(sum(count_set_totals[pair_idx])-this_tot) for this_tot in count_set_totals[pair_idx]] for pair_idx in range(4) ]

    #-- format for Corrfunc convert_rp_pi_counts_to_wp()
    D1D2_sets, D1R2_sets, D2R1_sets, R1R2_sets = jk_sets
    ND1_tots,  ND2_tots,  NR1_tots,  NR2_tots  = jk_set_totals

    #-- get wp for each jackknife set and multiple values of pimax
    pimax = np.max(pimax_array)

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

    for this_pimax in pimax_array:
        wp_jk   = 2*np.sum([ np.transpose(wp_by_patch)[i::int(pimax)] for i in range(int(pimax)) ][:int(this_pimax)], axis=0)
        wp_mean = np.nanmean(wp_jk, axis=1)
        N_jk    = len(wp_jk.T)
        cov     = np.zeros([len(rp_mids),len(rp_mids)])
        for rp_i in range(len(rp_mids)):
            wp_i = wp_jk[rp_i,:]
            wp_mean_i = wp_mean[rp_i]
            for rp_j in range(len(rp_mids)):
                wp_j = wp_jk[rp_j,:]
                wp_mean_j = wp_mean[rp_j]
                cov[rp_i][rp_j] = (1 - 1/N_jk)*sum((wp_i-wp_mean_i)*(wp_j-wp_mean_j))
        wp_err = [cov[i,i] for i in range(len(rp_mids))]
        out    = Table( data=[rp_mids, wp_mean, wp_err], names=("rp_cen","wp","wp_err") )
        this_fname = f"{fname}_{cross}_pimax{int(this_pimax)}.txt"
        if not quiet:
            print(f"Saving {this_fname}")
        ascii.write(out, save_path + this_fname, delimiter="\t", overwrite=True)
        
        
        
