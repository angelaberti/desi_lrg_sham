import os, sys
import numpy as np
from params import get_abs_mag_bins_clust

from datetime import datetime

def now():
    return f"[{str(datetime.now().time())[:-7]}]"


def print_now(bracket=True):
    now = str(datetime.now().time())[:-7]
    if bracket:
        print(f"[{now}]")
    else:
        print(now)


def get_cat_tag(d, zrange):
    """catalog tag: field (north/south) and photo-z range"""
    assert((d=="north") | (d=="south"))
    assert(type(zrange)==tuple)
    zmin, zmax = zrange[0], zrange[1]
    assert(zmin >= 0)
    assert(zmax > zmin)
    cat_tag  = "z{:.2f}-{:.2f}_{}".format(zmin,zmax,d)
    cat_tag  = cat_tag.replace(".","p")
    return cat_tag


def get_zmag_tag(zmag_lim):
    assert((type(zmag_lim)==float) | (type(zmag_lim)==int))
    """apparent z-band magnitude limit for photometric data"""
    zmag_tag = "zmaglim" + str(float(zmag_lim)).replace(".","p")
    return zmag_tag


def get_Mr_tag(Mr_lim):
    assert((type(Mr_lim)==float) | (type(Mr_lim)==int))
    assert(Mr_lim < 0)
    """absolute r-band magnitude limit for photometric data"""
    abs_mag_tag = "Mrlimn" + str(float(Mr_lim))[1:].replace(".","p")
    return abs_mag_tag


def get_Mz_tag(Mz_lim):
    assert((type(Mz_lim)==float) | (type(Mz_lim)==int))
    assert(Mz_lim < 0)
    """absolute z-band magnitude limit for photometric data"""
    abs_mag_tag = "Mzlimn" + str(float(Mz_lim))[1:].replace(".","p")
    return abs_mag_tag


def get_MW1_tag(MW1_lim):
    assert((type(MW1_lim)==float) | (type(MW1_lim)==int))
    assert(MW1_lim < 0)
    """absolute W1-band magnitude limit for photometric data"""
    abs_mag_tag = "MW1limn" + str(float(MW1_lim))[1:].replace(".","p")
    return abs_mag_tag


def get_zsnap_tag(zsnap):
    """redshift snapshot value for simulation"""
    zsnap_tag = "zsnap" + str(float(zsnap)).replace(".","p")
    return zsnap_tag


def get_abs_mag_bin_tags(zmin, band, nbins=None):
    bins = get_abs_mag_bins_clust(zmin, band, nbins=nbins)
    tags = [ f"{band}n{np.abs(bins[i])}-n{np.abs(bins[i+1])}" for i in range(len(bins)-1) ]
    tags = [ i.replace(".","p") for i in tags ]
    return tags


def get_abs_mag_bin_label(tag):
    if "Mr" in tag[:3]:
        band = r"$M_r$"
        tag = tag[2:]
    if "Mz" in tag[:3]:
        band = r"$M_z$"
        tag = tag[2:]
    elif "MW1" in tag[:3]:
        band = r"$M_{W1}$"
        tag = tag[3:]
    else:
        raise Exception()

    mag1, mag2 = tag.replace("p",".").split("-")
    
    if mag1[0]=="n": mag1 = -float(mag1[1:])
    else: mag1 = float(mag1)
    if mag2[0]=="n": mag2 = -float(mag2[1:])
    else: mag2 = float(mag2)
    
    if mag1 > mag1:
        mag_max = mag1
        mag_min = mag2
    else:
        mag_max = mag2
        mag_min = mag1
    
    if (mag_max - mag_min) > 5:
        label = band + f" $< {mag_max}$"
    elif (mag_max - mag_min) < -5:
        label = f"${mag_min} <$ " + band
    else:
        label = f"${mag_min} <$ " + band + f" $< {mag_max}$"

    return label



def sigma_tanh(alpha, delta, sigma0, m0, mag, quiet=True):
    return sigma0 + alpha*np.tanh( (mag - m0)/delta )



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



# def get_rp_use_tag(rp_use_range, return_tag=True, return_label=False):
#     rp_use_min, rp_use_max = rp_use_range
#     if (rp_use_min==None) & (rp_use_max==None):
#         rp_use_tag   = "all-rp-bins"
#         rp_use_label = "all-rp-bins"
        
#         if rp_use_max != None:
#             rp_use_tag   = f"rpmax{int(rp_use_max)}Mpch"
#             rp_use_label = r"$r_{\rm p} <\ $" + f"${int(rp_use_max)}$" + r"\ $h^{-1} {\rm Mpc}$"
#     elif (rp_use_min==None) and not (rp_use_max==None):
#         assert(rp_use_max >= 20)
#         rp_use_tag   = f"rpmax{str(round(rp_use_max,2)).replace('.','p')}Mpch"
#         rp_use_label = r"$r_{\rm p} <\ $" + f"${rp_use_max:.2f}$" + r"\ $h^{-1} {\rm Mpc}$"
#     elif not (rp_use_min==None) and (rp_use_max==None):
#         rp_use_tag   = f"rpmin{str(round(rp_use_min,2)).replace('.','p')}Mpch"
#         rp_use_label = r"$r_{\rm p} >\ $" + f"${rp_use_min:.2f}$" + r"\ $h^{-1} {\rm Mpc}$"
#     else:
#         assert((rp_use_min >= 0) & (rp_use_min < rp_use_max))
#         rp_use_tag   = f"rp{str(round(rp_use_min,2)).replace('.','p')}-{int(rp_use_max)}Mpch"
#         rp_use_label = f"${rp_use_min:.2f}$" + r"$\ < r_{\rm p} <\ $" + f"${int(rp_use_max)}$" + r"$\ h^{-1} {\rm Mpc}$"

#     if (return_tag==True & return_label==False):
#         return rp_use_tag
#     elif (return_label==True & return_tag==False):
#         return rp_use_label
#     elif (return_tag==True & return_label==True):
#         return rp_use_tag, rp_use_label
#     else:
#         return rp_use_tag
    
    


#-- copy/past of halotools.utils.crossmatch()
def crossmatch(x, y, skip_bounds_checking=False):
    # Ensure inputs are Numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Require that the inputs meet the assumptions of the algorithm
    if skip_bounds_checking is True:
        pass
    else:
        try:
            assert len(set(y)) == len(y)
            assert np.all(np.array(y, dtype=np.int64) == y)
            assert np.shape(y) == (len(y), )
        except:
            msg = ("Input array y must be a 1d sequence of unique integers")
            raise ValueError(msg)
        try:
            assert np.all(np.array(x, dtype=np.int64) == x)
            assert np.shape(x) == (len(x), )
        except:
            msg = ("Input array x must be a 1d sequence of integers")
            raise ValueError(msg)

    # Internally, we will work with sorted arrays, and then undo the sorting at the end
    idx_x_sorted = np.argsort(x)
    idx_y_sorted = np.argsort(y)
    x_sorted = np.copy(x[idx_x_sorted])
    y_sorted = np.copy(y[idx_y_sorted])

    # x may have repeated entries, so find the unique values as well as their multiplicity
    unique_xvals, counts = np.unique(x_sorted, return_counts=True)

    # Determine which of the unique x values has a match in y
    unique_xval_has_match = np.in1d(unique_xvals, y_sorted, assume_unique=True)

    # Create a boolean array with True for each value in x with a match, otherwise False
    idx_x = np.repeat(unique_xval_has_match, counts)

    # For each unique value of x with a match in y, identify the index of the match
    matching_indices_in_y = np.searchsorted(y_sorted, unique_xvals[unique_xval_has_match])

    # Repeat each matching index according to the multiplicity in x
    idx_y = np.repeat(matching_indices_in_y, counts[unique_xval_has_match])

    # Undo the original sorting and return the result
    return idx_x_sorted[idx_x], idx_y_sorted[idx_y]
