import os, sys
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join

from scipy import interpolate
import scipy.stats as stats

sys.path.append("./py")
from utils import *

from params import BASEDIR, DATADIR, MOCKDIR, get_zsnap_data, get_sham_var_bins, get_abs_mag_lim

fname = sys.argv[1]
zmin  = float(fname.split("_")[0][5:8].replace("p","."))

if "MW1" in fname:
    band = "MW1"
elif "Mz" in fname:
    band = "Mz"
else:
    raise Exception()

zrange   = (zmin, zmin + 0.1)
sim_tag  = "mdpl2"
sham_tag = "vpeak"
d        = "south"
zmag_lim = 20.7
quiet    = True if (("-q" in sys.argv) | ("--quiet" in sys.argv)) else False

z_snaps, snap_nums, snaps = get_zsnap_data(sim_tag)

zmin, zmax  = zrange
cat_tag     = get_cat_tag(d,zrange)
abs_mag_lim = get_abs_mag_lim(zmin, band)
zsim        = np.array(z_snaps)[ np.round(z_snaps,1)==zmin ][0]
zsnap_tag   = get_zsnap_tag(zsim)
zmag_tag    = get_zmag_tag(zmag_lim)

MW1lim = get_abs_mag_lim(zmin, "MW1")
Mrlim  = get_abs_mag_lim(zmin, "Mr")
Mzlim  = get_abs_mag_lim(zmin, "Mz")
if band=="MW1":
    abs_mag_lim_tag = get_MW1_tag(MW1lim)
elif band=="Mr":
    abs_mag_lim_tag = get_Mr_tag(Mrlim)
elif band=="Mz":
    abs_mag_lim_tag = get_Mz_tag(Mzlim)
else:
    raise Exception()
    

#-- load mock with rest-frame magnitudes and history information (and parent halo Mvir)
fpath = f"{MOCKDIR}/{sim_tag}/{sham_tag}/{d}"
if os.path.exists( f"{fpath}/{fname}" ):
    if not quiet:
        print(f"Loading {fpath}/{fname}...")
    cat = Table(np.load( f"{fpath}/{fname}" ))
else:
    raise Exception(f"{fpath}/{fname} not found!")



#-- add placeholder columns
for colname in ("zchar","zacc","zform"):
    if colname in cat.colnames:
        cat.remove_column(colname)
    cat.add_column(Column(99*np.zeros(len(cat)), name=colname))

#-- initialize counters for zstarve assignment algorithm
DSprint = int(10e4)

Mh_exceeds_12_yes = 0
Mh_exceeds_12_no  = 0

hist_len = len(cat["mvir_hist"][0])

these_snaps     = snaps[:hist_len]
these_scales    = these_snaps["scale"].data
these_redshifts = these_snaps["redshift"].data
this_scale      = these_snaps["scale"][-1]

print(f"a_sim: {this_scale:.5f}")
print(f"z_sim: {zsim}")


#-- calculate zstarve and add to mock
for i in range(len(cat)):
    #-- only calculate zstarve for halos with galaxies
    if cat["galaxy"][i]==True:
        htype = "Host Halo" if cat["upid"][i]==-1 else "Subhalo"
        include = False
            
        this_mvir_hist = cat["mvir_hist"][i]
        this_rvir_hist = cat["rvir_hist"][i]
        this_rs_hist   = cat["rs_hist"][i]
        a_acc          = cat["acc_scale"][i]        
    
        #-- z_char
        mh_max = np.max(np.log10(this_mvir_hist))
        if mh_max >= 12:
            #include = True
            Mh_exceeds_12_yes += 1
            idx        = np.where(np.log10(this_mvir_hist) >= 12)[0][-1]
            z_char_idx = hist_len-(idx+1)
            z_char     = these_redshifts[z_char_idx]
            if not quiet and (i%DSprint==0 or include):
                print(f"\n========== {htype} {i} ==========")
                print("idx\tlog Mh\tscale\tredshift")
            for j in np.concatenate([[0],np.arange(np.max([1,idx-2]),np.min([hist_len-1,idx+3]))]):
                z_char_idx = hist_len-(j+1)
                mark="*" if j==idx else " "
                if not quiet and (i%DSprint==0 or include):
                    print(f"  {mark}{j}{mark}\t{round(np.log10(this_mvir_hist[j]),3)}\t{these_scales[z_char_idx]}\t{these_redshifts[z_char_idx]}")
            if not quiet and (i%DSprint==0 or include):
                print(f"z_char = {z_char}")
        else:
            Mh_exceeds_12_no += 1
            z_char = zsim
            if not quiet and i%DSprint==0:
                print(f"max Mh = {round(mh_max,3)}")
        if not quiet and i%DSprint==0:
            print(f"z_char = {z_char}")
            
        #-- z_acc and z_form
        if cat["upid"][i] == -1: #-- host halos
            z_acc = zsim
            cvir  = (this_rvir_hist[0])/(this_rs_hist[0])
        else: #-- subhalos
            z_acc = 1/a_acc - 1
            z_form_idx = np.where(a_acc==these_scales[::-1])[0][0]
            if not quiet and (i%DSprint==0 or include):
                print("\nidx\tscale\trvir\trs")
                for j in np.arange(np.max([0,z_form_idx-2]),np.min([hist_len,z_form_idx+3])):
                    mark="*" if j==z_form_idx else " "
                    print(f"{mark}{j}{mark}\t{these_scales[::-1][j]}\t{this_rvir_hist[j]}\t{this_rs_hist[j]}")
            cvir = this_rvir_hist[z_form_idx]/this_rs_hist[z_form_idx]
            if not quiet and (i%DSprint==0 or include):
                print(f"cvir at a_acc={a_acc}: {round(cvir,3)}")
        z_form = np.round(cvir / (4.1*a_acc) - 1, 5)
        if not quiet and (i%DSprint==0 or include):
            print(f"z_form = {z_form}")
            print(f"z_acc  = {z_acc:.4f}")
    
        cat["zchar"][i] = z_char
        cat["zacc"][i]  = z_acc
        cat["zform"][i] = z_form

print(f"\nHalos achieving 10^12 Msun/h:     {Mh_exceeds_12_yes}")
print(f"Halos not achieving 10^12 Msun/h: {Mh_exceeds_12_no}")
print(f"Correct total number of halos?    {Mh_exceeds_12_no+Mh_exceeds_12_yes==len(cat)}\n")

z_starve = [np.max(i) for i in np.array([cat["zchar"],cat["zacc"],cat["zform"]]).T]

if "zstarve" in cat.colnames:
    cat.remove_column("zstarve")
cat.add_column(Column(z_starve, name="zstarve"))


#-- assign rest-frame colors
zstarve_bins = np.arange(0,16.1,0.01)
zstarve_cens = [np.mean(zstarve_bins[i:i+2]) for i in range(len(zstarve_bins)-1)]

log_zstarve_bins = np.logspace(np.log10(0.1*np.floor(10*zsim)),np.log10(15),40)
log_zstarve_cens = [np.log10(np.mean((10**log_zstarve_bins)[i:i+2])) for i in range(len(log_zstarve_bins)-1)]

color_cdf_name = f"{DATADIR}/stats/cdf_r-{band[1:]}_{cat_tag}_{zmag_tag}_{abs_mag_lim_tag}.npy"
color_cdf = Table(np.load(color_cdf_name))
color_cens = color_cdf["color_cens"]

abs_mag_bins = [ -1*float(i.split("-")[0][1:].replace("p",".")) for i in color_cdf.colnames[1:] ]
#-- add final magnitude bright limit
abs_mag_bins.append( -1*float(color_cdf.colnames[-1].split("-")[1][1:].replace("p",".")) )

color_cname = f"r-{band[1:]}"
if color_cname not in cat.colnames:
    cat.add_column(Column(np.zeros(len(cat)), name=color_cname))

for i in range(len(abs_mag_bins)-1):
    abs_mag_min, abs_mag_max = abs_mag_bins[i:i+2]

    abs_mag_bin_label = "n{:.2f}-n{:.2f}".format(np.abs(abs_mag_min), np.abs(abs_mag_max))
    abs_mag_bin_label = abs_mag_bin_label.replace(".","p")

    this_bin_color_cdf = color_cdf[abs_mag_bin_label]

    abs_mag_assigned = cat[f"{band}"]
    z_starve = cat["zstarve"]
    
    abs_mag_mask = (abs_mag_assigned < abs_mag_min) & (abs_mag_assigned >= abs_mag_max)
    
    this_bin_zstarve = z_starve[abs_mag_mask]
    H,_ = np.histogram(this_bin_zstarve, bins=log_zstarve_bins)
    this_bin_zstarve_cdf = np.cumsum(H)/np.cumsum(H)[-1]

    #-- interpolated halo CDF position as function of z_starve
    #-- input: z_starve
    #-- infer: halo CDF position
    halo_cdf_of_zstarve = interpolate.interp1d(log_zstarve_cens, this_bin_zstarve_cdf, fill_value="extrapolate")

    #-- interpolated r-? color as function of r-? CDF position
    #-- input: color CDF position
    #-- infer: color
    color_of_color_cdf = interpolate.interp1d(this_bin_color_cdf, color_cens, fill_value="extrapolate")

    #-- inferred halo CDF position for given z_starve
    halo_cdf_inferred = halo_cdf_of_zstarve(this_bin_zstarve)

    #-- assigned (inferred) color for given color CDF position
    halo_color_inferred = color_of_color_cdf(halo_cdf_inferred)

    cat[f"r-{band[1:]}"][abs_mag_mask] = halo_color_inferred

colors = cat[f"r-{band[1:]}"]

cat[f"r-{band[1:]}"][colors==np.inf] = np.nanmax(colors[colors != np.inf])

mask = (colors >= np.nanmin(colors)) & (colors <= np.nanmax(colors))
cat[f"r-{band[1:]}"][~mask] = np.nanmin(colors)



################################
##        LRG FLAGGING        ##
################################

#-- add columns for LRG flagging
for c in ("LRG_IR", "LRG_opt"):
    if c in cat.colnames:
        cat.remove_column(c)
    cat.add_column(Column(np.zeros(len(cat), dtype=bool), name=c))

#-- get model magnitudes and colors for LRG flagging
model_mags    = cat[band]
model_mag_min = np.min(model_mags)
model_mag_max = np.max(model_mags)

model_colors    = cat[f"r-{band[1:]}"]
model_color_min = np.min(model_colors)
model_color_max = np.max(model_colors)


### OPTICAL LRG SELECTION ###

lrgfrac_opt_fname = f"{DATADIR}/stats/lrgfrac_optical-select_{band[1:]}-band_{cat_tag}.npy"
print(f"Loading {lrgfrac_opt_fname}...")

mag_bins_opt, color_bins_opt, H_lrg_opt, H_full_opt = np.load(lrgfrac_opt_fname, allow_pickle=True)

if model_mag_min < np.min(mag_bins_opt):
    mag_bins_opt[0] = np.floor(model_mag_min)
if model_mag_max > np.max(mag_bins_opt):
    mag_bins_opt[-1] = model_mag_max

if model_color_min < np.min(color_bins_opt):
    color_bins_opt[0] = np.floor(model_color_min)
if model_color_max > np.max(color_bins_opt):
    color_bins_opt[-1] = model_color_max

lrg_frac_opt = H_lrg_opt/H_full_opt

N_bins = (len(color_bins_opt)-1)*(len(mag_bins_opt)-1)

k = 1
cat["LRG_opt"] = False

for i in range(len(color_bins_opt)-1):
    color_mask = (model_colors > color_bins_opt[i]) & (model_colors <= color_bins_opt[i+1])
    for j in range(len(mag_bins_opt)-1):
        mag_mask = (model_mags > mag_bins_opt[j]) & (model_mags <= mag_bins_opt[j+1])
        bin_mask = mag_mask & color_mask
        N_mock   = len(cat[bin_mask])
        if N_mock > 0:
            indices = np.where(bin_mask)[0]            
            N_gal_opt = H_full_opt[i,j]
            N_lrg_opt = H_lrg_opt[i,j]
            
            targ_lrg_opt = 0
            if (N_lrg_opt > 0) & (N_gal_opt > 0):
                targ_lrg_opt = int(N_mock*lrg_frac_opt[i,j])

            if (targ_lrg_opt > 0):
                if (targ_lrg_opt < N_mock):
                    selected = indices[np.random.choice(len(indices), targ_lrg_opt, replace=False)]
                    cat["LRG_opt"][selected] = True
                else:
                    cat["LRG_opt"][indices] = True

            if (targ_lrg_opt > 0) and (k%100==0):
                print(f"{k}/{N_bins}\t{color_bins_opt[i]:.2f}\t{mag_bins_opt[j]:.2f}\t{N_mock}\t{targ_lrg_opt}")
        k += 1


### IR LRG SELECTION ###

lrgfrac_IR_fname = f"{DATADIR}/stats/lrgfrac_IR-select_{band[1:]}-band_{cat_tag}.npy"
print(f"Loading {lrgfrac_IR_fname}...")

mag_bins_IR, color_bins_IR, H_lrg_IR, H_full_IR = np.load(lrgfrac_IR_fname, allow_pickle=True)

if model_mag_min < np.min(mag_bins_IR):
    mag_bins_IR[0] = np.floor(model_mag_min)
if model_mag_max > np.max(mag_bins_IR):
    mag_bins_IR[-1] = model_mag_max

if model_color_min < np.min(color_bins_IR):
    color_bins_IR[0] = np.floor(model_color_min)
if model_color_max > np.max(color_bins_IR):
    color_bins_IR[-1] = model_color_max

lrg_frac_IR = H_lrg_IR/H_full_IR

N_bins = (len(color_bins_IR)-1)*(len(mag_bins_IR)-1)

k = 1
cat["LRG_IR"] = False

for i in range(len(color_bins_IR)-1):
    color_mask = (model_colors > color_bins_IR[i]) & (model_colors <= color_bins_IR[i+1])
    for j in range(len(mag_bins_IR)-1):
        mag_mask = (model_mags > mag_bins_IR[j]) & (model_mags <= mag_bins_IR[j+1])
        bin_mask = mag_mask & color_mask
        N_mock   = len(cat[bin_mask])
        if N_mock > 0:
            indices = np.where(bin_mask)[0]            
            N_gal_IR = H_full_IR[i,j]
            N_lrg_IR = H_lrg_IR[i,j]
            
            targ_lrg_IR = 0
            if (N_lrg_IR > 0) & (N_gal_IR > 0):
                targ_lrg_IR = int(N_mock*lrg_frac_IR[i,j])

            if (targ_lrg_IR > 0):
                if (targ_lrg_IR < N_mock):
                    selected = indices[np.random.choice(len(indices), targ_lrg_IR, replace=False)]
                    cat["LRG_IR"][selected] = True
                else:
                    cat["LRG_IR"][indices] = True

            if (targ_lrg_IR > 0) and (k%100==0):
                print(f"{k}/{N_bins}\t{color_bins_IR[i]:.2f}\t{mag_bins_IR[j]:.2f}\t{N_mock}\t{targ_lrg_IR}")
        k += 1

fname_out = f"{fname[:-4]}_LRG-flagged.npy"
print(f"Saving {fname_out}...")

np.save(f"{fpath}/{fname_out}", cat)
