import numpy as np

import os, sys

from scipy import interpolate

from astropy.table import Table, Column, vstack, hstack, join
from astropy.io import fits, ascii
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c as c_speed
from astropy import units as u

from params import BASEDIR, DATADIR, get_zsnap_data, H0, Om0
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
zkcorrs,_,_ = get_zsnap_data("bolshoip")


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



def sph_to_cart(RA,Dec,redshift):
    """Convert (RA,Dec,redshift) to cartesian with observer at origin
    Positional inputs: RA (deg), Dec (deg), redshift
    Returns: (x,y,z) (3D cartesian coordinates)"""
    r     = cosmo.comoving_distance(redshift).value 
    theta = (90.0 - Dec)*np.pi/180.0
    phi   = RA*np.pi/180.0
    
    x = r*np.sin(theta)*np.cos(phi).data
    y = r*np.sin(theta)*np.sin(phi).data
    z = r*np.cos(theta).data
    
    return x, y, z

    
    
def load_cat(d="north", zrange=(0.3,0.4), zphot_tag="Z_PHOT_MEAN", quiet=False, DATADIR=DATADIR):
    """load north/south data catalog in speficied redshift range"""
    assert(len(zrange)==2)
    zmin, zmax = zrange[0], zrange[1]    
    
    cat_tag = "z{:.2f}-{:.2f}_{}".format(zmin,zmax,d)
    cat_tag = cat_tag.replace(".","p")
    cat_fname = DATADIR + "/cat_" + cat_tag + ".npy"
    pz_fname  = DATADIR + "/cat_" + cat_tag + "_pz.npy"

    if not quiet : print(f"Loading {cat_fname}...")
    cat_mag = Table(np.load(cat_fname))
    if zphot_tag not in cat_mag.colnames:
        if not quiet : print(f"Loading {pz_fname}...")
        cat_pz  = Table(np.load(pz_fname))
        if not quiet : print("Combining catalogs...")
        cat = hstack([cat_pz,cat_mag])
        del cat_pz
    else:
        cat = cat_mag.copy()
    if not quiet : print("Cleaning up...")
    del cat_mag
    zz = cat[zphot_tag]
    zp_mask = (zz >= zmin) & (zz < zmax)

    if not quiet : print(f"+/-150 Mpc/h\t: {len(cat)}\n{zmin} < z < {zmax}\t: {len(cat[zp_mask])}")
    
    return cat



def load_kcorr(d="north", zrange=(0.3,0.4), zkcorr=0.33028, quiet=False):
    """load pre-computed k-corrections for data catalog with specified parameters (north/south, photo-z range, zkcorr)"""
    assert(len(zrange)==2)
    zmin, zmax = zrange[0], zrange[1]
    
    #kcorr_tag = "z{:.2f}-{:.2f}_{}_z{:.2f}".format(zmin,zmax,d,zkcorr)
    kcorr_tag = "z{:.2f}-{:.2f}_{}".format(zmin,zmax,d)
    kcorr_tag = kcorr_tag.replace("-","_")
    kcorr_tag = kcorr_tag.replace(".","p")
    kcorr_fname = f"{DATADIR}/kcorr/kcorr_{kcorr_tag}.fits"

    if not quiet : print("Loading K-corrections...")
    #if d.lower()=="north":
    kcorr = fits.open(kcorr_fname, mode="denywrite", memmap=True)
    kcorr = kcorr[0].data
    [Kg,Kr,Kz,Kw1,Kw2] = kcorr.T
    # elif d.lower()=="south":
    #     kcorr = Table(fits.open(f"{DATADIR}/kcorr/kcorr_z0p30_0p40_south_z0p33.fits")[1].data)
    #     [Kg,Kr,Kz,Kw1,Kw2] = [kcorr[f"col{i}"].data for i in range(5)]
    del kcorr

    return [Kg,Kr,Kz,Kw1,Kw2]




#-- patch number substitutions for healpix assignments
#-- valid ONLY for NSIDE=6
def get_patch_subs(d):
    """patch number substitutions for north/south healpix jackknife regions
    NOTE: valid only for NSIDE=6"""
    if (d.lower()=="north") or (d.lower()=="n"):
        return {115:91, 116:92, 117:93, 118:94, 119:95, 120:96, 121:97, 122:98, 123:99, 124:100, 125:101, 126:102, 103:78, 79:55, 56:36, 37:21, 10:21}
    elif (d.lower()=="south") or (d.lower()=="s"):
        return {106:130, 129:154, 153:177, 272:249, 297:273, 298:274, 299:275, 276:252, 388:405, 257:281, 282:305, 306:330, 185:208, 233:208,
                236:212, 260:237, 261:238, 262:239, 263:240, 264:241, 265:242, 266:243, 244:219, 345:368, 364:369, 347:370, 91:115, 92:116, 
                93:117, 94:118, 95:119, 96:120, 97:121, 98:122, 99:123, 100:124, 101:125, 102:150, 126:150, 107:130, 84:131, 85:108, 86:109,
                135:110, 224:201, 324:300, 346:369, 396:377, 354:377, 410:395, 163:140, 408:393, 409:394, 418:405, 419:406}
    else:
        raise Exception("Arg must be 'north' ('n') or 'south' ('s')")
        
        
        
#-- fractional sky area
def get_survey_area(masked=True, pts_per_degsq=2500, quiet=False):
    if masked : return (0.11153, 0.32040)
    else : return (0.12498, 0.37669)
    

    
#-- percentage by which to pad redshit cut to include extra d Mpc/h (default=150)
def _p_of_z(z, d=150, pp = np.arange(0.01,0.31,0.01)):
    dd = cosmo.comoving_distance(z*(1+pp)) - cosmo.comoving_distance(z)
    f = interpolate.interp1d(dd, pp)
    pnew = f(d)
    return float(pnew)



def _prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    #-- define name of temporary dummy file
    dummy_file = file_name + ".bak"
    #-- open original file in read mode and dummy file in write mode
    with open(file_name, "r") as read_obj, open(dummy_file, "w") as write_obj:
        #-- write given line to the dummy file
        write_obj.write(line + "\n")
        #-- read lines from original file one by one and append them to the dummy file
        for line in read_obj : write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)

    
    
#-- create z-band magnitude limited DR9 catalogs by north/south region with corresponding photo-z catalog
#-- z <= 21 limit for confident photo-zs (Zhu+ 2020)
def make_cat(ddir, zbins, pad_z_Mpc=150, N_files="all", glim=24.0, rlim=23.4, zlim=21.0, zphot_tag="Z_PHOT_MEAN", format_kcorr_idl=True, quiet=True):
    assert(os.path.exists(ddir))
    for d in (DATADIR, f"{DATADIR}/kcorr"):
        if not os.path.exists(d):
            if not quiet : print(f"Creating {d}...")
            os.makedirs(d)
    if "south" in ddir : suffix = "south"
    elif "north" in ddir : suffix = "north"
    else : raise Exception()
    assert(len(zbins) >= 2)
    #-- get list of all files in ddir
    fmg = os.listdir(f"{ddir}/")
    fmg.sort()
    if ".fits" not in fmg[0] : fmg = fmg[1:]    
    
    #-- trim file list to specified length >= 1
    if N_files=="all":
        N_files = len(fmg)
        filelist = fmg
    elif N_files==1 : filelist = [fmg[0]]
    else : filelist = fmg[:N_files]

    if not quiet : print(f"Loading {N_files} of {len(fmg)} file(s) from {ddir}...")
    col_idx = np.concatenate([ np.arange(1,9), np.arange(10,16), np.arange(18,23), np.arange(25,30), np.arange(53,67), [111,112] ])

    chunks_mag = {}
    chunks_pz  = {}
    #-- loop over files
    for i,f in enumerate(filelist):
        if not quiet:
            if N_files < 20 : print(f"{i} / {N_files}")
            elif i%10==0 : print(f"{i} / {N_files}")    
        #-- load this photo-z and this mag file
        
        hdu  = fits.open(f"{ddir}/{f}", mode="denywrite", memmap=True)
        data = hdu[1].data
        hdu.close()
        chunk_mag = Table([data[data.names[i]] for i in col_idx], names=[data.names[i] for i in col_idx])
        
        hdu  = fits.open(f"{ddir}-photo-z/{f[:-5]}-pz.fits", mode="denywrite", memmap=True)
        data = hdu[1].data
        hdu.close()
        chunk_pz = Table([data[data.names[i]] for i in np.arange(3,11)], names=[data.names[i] for i in np.arange(3,11)])
        #-- define and apply magnitude cuts
        
        gAB = 2.5*( 9 - np.log10(chunk_mag["FLUX_G"]/chunk_mag["MW_TRANSMISSION_G"]) )
        rAB = 2.5*( 9 - np.log10(chunk_mag["FLUX_R"]/chunk_mag["MW_TRANSMISSION_R"]) )
        zAB = 2.5*( 9 - np.log10(chunk_mag["FLUX_Z"]/chunk_mag["MW_TRANSMISSION_Z"]) )
        
        if glim is not None : gmask = (gAB <= glim)
        else : gmask = np.ones(len(chunk_mag), dtype=bool)    

        if rlim is not None : rmask = (rAB <= rlim)
        else : rmask = np.ones(len(chunk_mag), dtype=bool)
                                           
        if zlim is not None : zmask = (zAB <= zlim)
        else : zmask = np.ones(len(chunk_mag), dtype=bool)
                                           
        mag_mask = gmask & rmask & zmask

        chunk_pz  = chunk_pz[mag_mask]
        chunk_mag = chunk_mag[mag_mask]
        
        #-- get redshifts
        zphot = chunk_pz[zphot_tag]

        #-- loop over redshift bins
        for j in range(len(zbins)-1):
            zmin, zmax = zbins[j], zbins[j+1]
            #if not quiet : print(f"Redshift range: ({zmin},{zmax})")
            chunky = "{:.2f}-{:.2f}".format(zmin,zmax)
            if chunky not in chunks_mag.keys() : chunks_mag[chunky] = []
            if chunky not in chunks_pz.keys() : chunks_pz[chunky] = []
            
            #-- optionally pad redshift limits for clustering measurements
            if pad_z_Mpc is not None:
                zmin *= (1 - _p_of_z(zmin, d=pad_z_Mpc))
                zmax *= (1 + _p_of_z(zmax, d=pad_z_Mpc))
                if (i%10==0) and not quiet : print(f"Padding redshift range by {pad_z_Mpc} Mpc/h " + "({:.3f},{:.3f})".format(zmin,zmax))

            #-- define redshift cut and apply to current photo-z and mag files
            pz_mask = (zphot >= zmin) & (zphot < zmax)
            this_chunk_pz = chunk_pz[pz_mask]
            if len(this_chunk_pz) > 0:
                chunks_mag[chunky].append( chunk_mag[pz_mask] )
                chunks_pz[chunky].append( this_chunk_pz )
                del this_chunk_pz
                
    for k in chunks_mag.keys():
        if not quiet : print(f"Processing {k}...")
        cat_mag = np.concatenate(chunks_mag[k])
        cat_pz  = np.concatenate(chunks_pz[k])
        cat_mag_fname = "{}/cat_z{}_{}.npy".format(DATADIR, k.replace(".","p"), suffix)
        cat_pz_fname  = "{}/cat_z{}_{}_pz.npy".format(DATADIR, k.replace(".","p"), suffix)
        if not quiet : print(f"Saving compiled catalogs:\n  {cat_mag_fname}\n  {cat_pz_fname}")
        np.save(cat_mag_fname, cat_mag)
        np.save(cat_pz_fname, cat_pz)
        
        if format_kcorr_idl : fmt_kcorr_idl(cat_mag_fname, cat_mag=cat_mag, cat_pz=cat_pz, zphot_tag=zphot_tag, quiet=quiet)
        del cat_mag, cat_pz



#-- format catalog data (maggies, maggies_ivar, photo-zs) for idl kcorrect
def fmt_kcorr_idl(cat_mag_fname, cat_mag=False, cat_pz=False, zphot_tag="Z_PHOT_MEAN", quiet=True, chunk_length=1e8):
    if type(cat_mag) is bool : cat_mag = np.load(cat_mag_fname)
    if type(cat_pz) is bool : cat_pz = np.load(cat_mag_fname[:-4] + "_pz.npy")
    f = cat_mag_fname.split("_")
    k, suffix = f[1].replace("-","_"), f[2][:5]

    maggies_fname = f"maggies_{k}_{suffix}.txt"

    nk = len(cat_mag)
    N_chunks = int(np.ceil(nk/chunk_length))
    for i in np.arange(N_chunks):
        min_idx, max_idx = int(i*chunk_length), int((i+1)*chunk_length)
        if max_idx > nk : max_idx = nk
        nk_chunk = max_idx-min_idx
        chunk_mask = np.arange(min_idx,max_idx)
        if not quiet : print(f"Processing ({min_idx},{max_idx-1})...")
        
        if N_chunks > 1 : maggies_fname = f"{DATADIR}/kcorr/maggies_{k}_{suffix}_{i}.txt"

        pname_out = f"{DATADIR}/kcorr/zphot{maggies_fname[7:]}"
        pname_out = pname_out.replace("-","_")
        np.savetxt(pname_out, cat_pz[zphot_tag][chunk_mask])
        header = "{} {} {}".format(2,1,nk_chunk)
        _prepend_line(pname_out, str(header))
        if not quiet : print(f"Creating {pname_out}...")

        out = np.concatenate([ cat_mag["FLUX_G"][chunk_mask]/cat_mag["MW_TRANSMISSION_G"][chunk_mask],
                               cat_mag["FLUX_R"][chunk_mask]/cat_mag["MW_TRANSMISSION_R"][chunk_mask],
                               cat_mag["FLUX_Z"][chunk_mask]/cat_mag["MW_TRANSMISSION_Z"][chunk_mask],
                               cat_mag["FLUX_W1"][chunk_mask]/cat_mag["MW_TRANSMISSION_W1"][chunk_mask],
                               cat_mag["FLUX_W2"][chunk_mask]/cat_mag["MW_TRANSMISSION_W2"][chunk_mask],
                             ])
        pname_out = f"{DATADIR}/kcorr/maggies_ivar{maggies_fname[7:]}"
        pname_out = pname_out.replace("-","_")
        np.savetxt(pname_out, out)
        del out
        header = "{} {} {}".format(2,5,nk_chunk)
        _prepend_line(pname_out, str(header))
        if not quiet : print(f"Creating {pname_out}...")

        out = np.concatenate([ cat_mag["FLUX_IVAR_G"][chunk_mask],
                               cat_mag["FLUX_IVAR_R"][chunk_mask],
                               cat_mag["FLUX_IVAR_Z"][chunk_mask],
                               cat_mag["FLUX_IVAR_W1"][chunk_mask],
                               cat_mag["FLUX_IVAR_W2"][chunk_mask],
                             ])
        pname_out = f"{DATADIR}/kcorr/{maggies_fname}"
        pname_out = pname_out.replace("-","_")
        np.savetxt(pname_out, out)
        del out
        _prepend_line(pname_out, str(header))
        if not quiet : print(f"Creating {pname_out}...")
