import os, sys
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c as c_speed
from astropy import units as u

BASEDIR = "/Users/aberti/Desktop/research/desi_lrg_sham"
sys.path.append(BASEDIR)

# from halocat_history import read_halocat#, load_trees, get_hist_from_tree, get_subhist_from_tree
from params import BASEDIR, DATADIR, H0, Om0, zsnaps
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

import Corrfunc
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.wp import wp as wp_corrfunc


#-- define bins for clustering (data and mocks)
nbins   = 19
rp_min  = 0.0333
rp_max  = 64.4098
rp_bins = np.logspace( np.log10(rp_min), np.log10(rp_max), nbins + 1 )
rp_mids = np.array([ 10**np.mean(np.log10(rp_bins)[i:i+2]) for i in np.arange(len(rp_bins)-1) ])
binfile = f"{BASEDIR}/clustering/rp_bins.txt"

#-- correlation function parameters
autocorr   = 0
cosmology  = 2
nthreads   = 2
pimax      = 150
rand_coeff = 20

#-- data parameters
d          = "north"
zmin, zmax = 0.4, 0.5
mz_lim     = 20.7

#-- housekeeping
ztag      = "{:.2f}-{:.2f}".format(zmin, zmax)
ztag      = ztag.replace(".","p")
zsnap     = zsnaps[int(10*zmin)]
zsnap_tag = str(zsnap).replace(".","p")
zlim_tag  = str(mz_lim).replace(".","p")


def _cart_to_sph(x, y, z):
    rho   = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/rho)
    phi   = np.arcsin(y/np.sqrt(x**2 + y**2))
    
    cz    = rho
    delta = np.rad2deg(np.pi/2. - theta)
    alpha = np.rad2deg(phi)
    
    return alpha, delta, cz


halocat_vpeak_select = Table(np.load(f"{BASEDIR}/sims/vpeak_select_zsnap{zsnap_tag}_zmaglim{zlim_tag}.npy"))

cat     = halocat_vpeak_select
boxsize = 250.

#-- stack simulation volume and translate coordinates for conversion to RA/Dec/dcom
D = cosmo.comoving_distance(zsnap).value

#-- wide mock data
#-- wide mock data
z2 = np.concatenate( [cat["x"], cat["x"], cat["x"]] ) - boxsize/2
y2 = np.concatenate( [cat["y"], cat["y"], cat["y"]] ) - boxsize/2
x2 = np.concatenate( [np.array(cat["z"])-1.5*boxsize,
                      np.array(cat["z"])-0.5*boxsize,
                      np.array(cat["z"])+0.5*boxsize] ) + D

wide = np.abs(x2 - D) < (boxsize/2 + pimax)
z2 = z2[wide]
y2 = y2[wide]
x2 = x2[wide]

alpha2, delta2, cz2 = _cart_to_sph(x2,y2,z2)

#-- narrow mock data
narrow = np.abs(x2 - D) < boxsize/2
x1 = x2[narrow]
y1 = y2[narrow]
z1 = z2[narrow]

alpha1, delta1, cz1 = _cart_to_sph(x1,y1,z1)

ND1 = len(alpha1)
ND2 = len(alpha2)

NR1 = rand_coeff*ND1
NR2 = rand_coeff*ND2

#-- narrow randoms
RZ1 = np.random.uniform(-boxsize/2, boxsize/2, NR1)
RY1 = np.random.uniform(-boxsize/2, boxsize/2, NR1)
RX1 = np.random.uniform(-boxsize/2, boxsize/2, NR1) + D

#-- wide randoms
RZ2 = np.random.uniform(-boxsize/2, boxsize/2, NR2)
RY2 = np.random.uniform(-boxsize/2, boxsize/2, NR2)
RX2 = np.random.uniform(-(boxsize/2 + pimax), boxsize/2 + pimax, NR2) + D

alpha1R, delta1R, cz1R = _cart_to_sph(RX1,RY1,RZ1)
alpha2R, delta2R, cz2R = _cart_to_sph(RX2,RY2,RZ2)

# DS = 1

D1 = (alpha1, delta1, cz1)
D2 = (alpha2, delta2, cz2)
R1 = (alpha1R, delta1R, cz1R)
R2 = (alpha2R, delta2R, cz2R)
# D1 = (alpha1[::DS], delta1[::DS], cz1[::DS])
# D2 = (alpha2[::DS], delta2[::DS], cz2[::DS])
# R1 = (alpha1R[::DS], delta1R[::DS], cz1R[::DS])
# R2 = (alpha2R[::DS], delta2R[::DS], cz2R[::DS])

pairs_D1D2 = ( D1, D2 )
pairs_D1R2 = ( D1, R2 )
pairs_D2R1 = ( D2, R1 )
pairs_R1R2 = ( R1, R2 )

pair_sets   = (pairs_D1D2, pairs_D1R2, pairs_D2R1, pairs_R1R2)
pair_counts = []

for pair,tag in zip(pair_sets,("D1D2","D1R2","R1D2","R1R2")):
    alpha1, delta1, cz1 = pair[0]
    alpha2, delta2, cz2 = pair[1]

    kwargs = dict(RA2=alpha2, DEC2=delta2, CZ2=cz2, weights2=None, is_comoving_dist=True, verbose=False)
    counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rp_bins, alpha1, delta1, cz1, **kwargs)

    pair_counts.append(counts)
    print(f"{len(pair_counts)} / {len(pair_sets)}: {tag}")
    
#-- sets of pair counts (DD, DR, RD, RR)
count_sets = [ pair_counts[pair_idx]["npairs"] for pair_idx in range(4) ]

D1D2, D1R2, D2R1, R1R2 = count_sets

xi = ( D1D2/(ND1*ND2) - D1R2/(ND1*NR2) - D2R1/(ND2*NR1) ) / ( R1R2/(NR1*NR2) ) + 1

#-- get wp for multiple values of pimax
pimax_array = np.arange(10.,pimax+1.,10.)

wp_from_DDrppi = {}
wp_from_DDrppi["rp_cen"] = rp_mids

for this_pimax in pimax_array:
    this_wp = 2*sum([ xi[int(i)::int(pimax)] for i in range(int(pimax)) ][:int(this_pimax)])
    wp_from_DDrppi[str(this_pimax)] = this_wp

result = Table( wp_from_DDrppi )

np.save(f"{BASEDIR}/clustering/wp_zsnap{zsnap_tag}_{d}_zmaglim{zlim_tag}.npy", result)

