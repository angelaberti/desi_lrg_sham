import os, sys
import numpy as np

from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack, join

from params import BASEDIR, DATADIR, get_zsnap_data, H0, Om0
zkcorrs,_,_ = get_zsnap_data("bolshoip")

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

import healpy as hp

from functions import load_kcorr, get_patch_subs #, sph_to_cart, load_cat, get_survey_area

#-- DESI LRG sample cuts
#-- https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection
def desi_lrg_mask(cat, d="north", zrange=(0.3,0.4), zkcorr=None, quiet=False):
    """return LRG selection mask for a given catalog"""
    g  = 2.5*( 9 - np.log10(cat["FLUX_G"]/cat["MW_TRANSMISSION_G"]) )
    r  = 2.5*( 9 - np.log10(cat["FLUX_R"]/cat["MW_TRANSMISSION_R"]) )
    z  = 2.5*( 9 - np.log10(cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
    W1 = 2.5*( 9 - np.log10(cat["FLUX_W1"]/cat["MW_TRANSMISSION_W1"]) )

    zfiber    = 2.5*( 9 - np.log10(cat["FIBERFLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
    zfibertot = 2.5*( 9 - np.log10(cat["FIBERTOTFLUX_Z"]) )
    
    #-- quality cuts for north and south fields
    qq = [ (cat["FLUX_IVAR_R"] > 0) & (cat["FLUX_R"] > 0), ## r-band quality
           (cat["FLUX_IVAR_Z"] > 0) & (cat["FLUX_Z"] > 0) & (cat["FIBERFLUX_Z"] > 0), ## z-band quality
           (cat["FLUX_IVAR_W1"] > 0) & (cat["FLUX_W1"] > 0), ## W1-band quality
           # (cat["NOBS_G"] > 0) & (cat["NOBS_R"] > 0) & (cat["NOBS_Z"] > 0), ## observed in all bands
           (zfibertot > 17.5), ## remove bright stars
        ]
    #-- NOTE: BRIGHT, GALAXY, CLUSTER, and bright GAIA star masking already applied to full photometric sample
  
    if d=="north":
        cc = [(z - W1) > (0.8*(r - z) - 0.6),
              zfiber < 21.61, 
              (g - W1 > 2.97) | (r - W1 > 1.8),
              ( (r - W1 > 1.83*(W1 - 17.13)) & (r - W1 > (W1 - 16.31)) ) | (r - W1 > 3.4),
              ]

    elif d=="south":
        cc = [(z - W1) > (0.8*(r - z) - 0.6),
              zfiber < 21.6,
              (g - W1 > 2.9) | (r - W1 > 1.8),
              ( (r - W1 > 1.8*(W1 - 17.14)) & (r - W1 > (W1 - 16.33)) ) | (r - W1 > 3.3),
              ]

    return np.array( cc[0] & cc[1] & cc[2] & cc[3] & qq[0] & qq[1] & qq[2] & qq[3] )



def extra_masks(cat, d=None, zrange=None, quiet=False, zkcorr=None, zphot_tag="Z_PHOT_MEAN", mask_name=None):
    assert(mask_name != None)
    snaps = {}
    for i in np.arange(0,10,1): snaps[round(0.1*i,1)] = zkcorrs[int(i)]
    
    zmin, zmax = zrange
    assert( (zmax > zmin) & (zmin in snaps.keys()) )

    if (zkcorr=="zsnap") | (zkcorr==None) : zkcorr = snaps[zmin]
    if not quiet : print(f"K-correction redshift set to {zkcorr}")

    mg  = 2.5*( 9 - np.log10(cat["FLUX_G"]/cat["MW_TRANSMISSION_G"]) )
    mr  = 2.5*( 9 - np.log10(cat["FLUX_R"]/cat["MW_TRANSMISSION_R"]) )
    mz  = 2.5*( 9 - np.log10(cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
    mW1 = 2.5*( 9 - np.log10(cat["FLUX_W1"]/cat["MW_TRANSMISSION_W1"]) )

    mx = np.array( [mg, mr, mz, mW1] )

    Kx_all = load_kcorr(d=d, zrange=zrange, zkcorr=zkcorr, quiet=quiet)
    Kx     = np.array( [k for k in Kx_all] )
    Kg, Kr, Kz, KW1 = Kx[:-1]

    Mx = mx - cosmo.distmod( cat[zphot_tag] ).value - Kx[:-1]

    g_r = (mg - Kg) - (mr - Kr)
    Mr  = Mx[1]
    
    g_W1 = (mg - Kg) - (mW1 - KW1)
    MW1  = Mx[3]
    
    if mask_name=="red_seq_Mr_vs_g-r_z0p4-0p5_box1":
        mask = (-22.4 < Mr) & (Mr < -19.8) & (1.5 < g_r) & (g_r < 2.0)
    elif mask_name=="red_seq_Mr_vs_g-r_z0p4-0p5_line1":
        mask = g_r > -0.148*Mr - 1.63575
        
    elif mask_name=="red_seq_MW1_vs_g-W1_z0p4-0p5_box1":
        mask = (-24 < MW1) & (MW1 < -21) & (2.8 < g_W1) & (g_W1 < 4.0)
    elif mask_name=="red_seq_MW1_vs_g-W1_z0p4-0p5_line1":
        mask = g_W1 > -0.273*MW1 - 3.113095
    
    else:
        raise Exception("Invalid extra mask name!")

    return mask



#-- Rongpu's 2022 DESI target selection paper
def desi_lrg_mask_optical(cat, d="south", zrange=None, zkcorr=None, quiet=False):
    """optical LRG selection mask (Zhu+ 2020) for a given catalog"""
    g  = 2.5*( 9 - np.log10(cat["FLUX_G"]/cat["MW_TRANSMISSION_G"]) )
    r  = 2.5*( 9 - np.log10(cat["FLUX_R"]/cat["MW_TRANSMISSION_R"]) )
    z  = 2.5*( 9 - np.log10(cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
    W1 = 2.5*( 9 - np.log10(cat["FLUX_W1"]/cat["MW_TRANSMISSION_W1"]) )

    zfiber    = 2.5*( 9 - np.log10(cat["FIBERFLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
    zfibertot = 2.5*( 9 - np.log10(cat["FIBERTOTFLUX_Z"]) )
    
    #-- quality cuts for north and south fields
    qq = [ (cat["FLUX_IVAR_R"] > 0) & (cat["FLUX_R"] > 0), ## r-band quality
           (cat["FLUX_IVAR_Z"] > 0) & (cat["FLUX_Z"] > 0) & (cat["FIBERFLUX_Z"] > 0), ## z-band quality
           (cat["FLUX_IVAR_W1"] > 0) & (cat["FLUX_W1"] > 0), ## W1-band quality
           (zfibertot > 17.5), ## remove bright stars
        ]
    #-- NOTE: BRIGHT, GALAXY, CLUSTER, and bright GAIA star masking already applied to full photometric sample

    cc = [((z < 21.0) | (zfiber < 22.0)),
          (z - W1) > (0.8*(r - z) - 0.8),
          ((g - W1 > 2.5) & (g - r > 1.3)) | (r - W1 > 1.7),
          ((z < 20.2) & (r - z > 0.45*(z - 17.20)) & (r - z > 0.19*(z - 14.17))) | ((z >= 20.2) & (((z - 23.18)/1.3)**2 + (r - z + 2.5)**2 > 4.48**2)),
          ]

    return np.array( cc[0] & cc[1] & cc[2] & cc[3] & qq[0] & qq[1] & qq[2] & qq[3] )


# #-- https://arxiv.org/pdf/2001.06018.pdf
# def desi_lrg_mask_optical(cat, d="south", zrange=None, zkcorr=None, quiet=False):
#     """optical LRG selection mask (Zhu+ 2020) for a given catalog"""
#     g  = 2.5*( 9 - np.log10(cat["FLUX_G"]/cat["MW_TRANSMISSION_G"]) )
#     r  = 2.5*( 9 - np.log10(cat["FLUX_R"]/cat["MW_TRANSMISSION_R"]) )
#     z  = 2.5*( 9 - np.log10(cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
#     W1 = 2.5*( 9 - np.log10(cat["FLUX_W1"]/cat["MW_TRANSMISSION_W1"]) )

#     zfiber    = 2.5*( 9 - np.log10(cat["FIBERFLUX_Z"]/cat["MW_TRANSMISSION_Z"]) )
#     zfibertot = 2.5*( 9 - np.log10(cat["FIBERTOTFLUX_Z"]) )
    
#     #-- quality cuts for north and south fields
#     qq = [ (cat["FLUX_IVAR_R"] > 0) & (cat["FLUX_R"] > 0), ## r-band quality
#            (cat["FLUX_IVAR_Z"] > 0) & (cat["FLUX_Z"] > 0) & (cat["FIBERFLUX_Z"] > 0), ## z-band quality
#            (cat["FLUX_IVAR_W1"] > 0) & (cat["FLUX_W1"] > 0), ## W1-band quality
#            (zfibertot > 17.5), ## remove bright stars
#         ]
#     #-- NOTE: BRIGHT, GALAXY, CLUSTER, and bright GAIA star masking already applied to full photometric sample

#     cc = [(z - W1) > (0.8*(r - z) - 0.6),
#           z < 20.41,
#           (r - z) > 0.5*(z - 17.18),
#           (r - z) > 0.9,
#           ((r - z) > 1.15) | ((g - r) > 1.65)]

#     return np.array( cc[0] & cc[1] & cc[2] & cc[3] & cc[4] & qq[0] & qq[1] & qq[2] & qq[3] )



def cat_geo_mask(cat, plot=False, quiet=False):
    """define geometric mask for data and assign jackknife regions (NSIDE=6)"""
    RA  = cat["RA"]
    Dec = cat["DEC"]
    
    d = "north" if (np.max(Dec) > 45.) else "south"

    if not quiet : print(f"Assigning jackknife patches ({d.upper()})...")
    theta = np.radians(90.0 - Dec)
    phi   = np.radians(RA)
    nside = 6

    px_idx = hp.ang2pix(nside, theta, phi)
    pixel_indices = px_idx.copy()
    replace = get_patch_subs(d)

    for i in replace.keys() : pixel_indices[pixel_indices==i] = replace[i]

    if "PATCH" in cat.colnames : cat.remove_column("PATCH")
    cat.add_column(Column(data=pixel_indices, name="PATCH"))

    if not quiet : print("Getting data catalog masks...")
    bit_mask = masks_from_maskbits(cat["MASKBITS"])["COMBINED"]
    if not quiet : print("{:.3f}\tSky contamination".format(len(RA[~bit_mask])/len(RA)))

    star_mask = (masks_from_fitbits(cat["FITBITS"])["STARS"]) | (cat["TYPE"]=="PSF")
    if not quiet : print("{:.3f}\tStars".format(len(RA[~star_mask])/len(RA)))

    coord_mask = get_coord_mask(RA,Dec,field=d)
    if not quiet : print("{:.3f}\tRA/Dec cuts to match randoms".format(len(RA[~coord_mask])/len(RA)))

    geo_mask = ~( coord_mask | bit_mask | star_mask )
    if not quiet : print("{:.3f}\tAll combined".format(len(RA[geo_mask])/len(RA)))

    if plot:
        if d=="north":
            ds = 1
            xlim, ylim = (88,301), (30,85)
            xticks, yticks = np.arange(90,305,5), np.arange(30,90,5)
        elif d=="south":
            ds = 50
            xlim, ylim = (0,360), (-70,36)
            xticks, yticks = np.arange(0,370,10), np.arange(-70,40,5)

        fig, ax = plt.subplots(1,1,figsize=(30,10))

        cm1 = plt.get_cmap("tab20b")
        cm2 = plt.get_cmap("tab20c")
        ax.set_prop_cycle(color=np.concatenate([cm1.colors,cm2.colors]))

        ax.scatter(RA[::ds], Dec[::ds], s=1, color="black")
        for p in np.unique(cat["PATCH"]):
            patch_mask = cat["PATCH"]==p
            ax.scatter(RA[geo_mask & patch_mask][::ds], Dec[geo_mask & patch_mask][::ds], s=1, alpha=0.5)
            #ax.scatter(RA[geo_mask][::ds], Dec[geo_mask][::ds], s=1, color="gray")

        ax.grid(ls=":")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xlabel("RA", fontsize=16)
        ax.set_ylabel("Dec", fontsize=16)

        plt.savefig(f"{BASEDIR}/figures/masked_data_{d}.png", bbox_inches="tight", pad_inches=0.1, dpi=200)
        plt.show()
        
    return cat, geo_mask



def masks_from_maskbits(maskbits, keys=("ALLMASK_G","ALLMASK_R","ALLMASK_Z","WISEM1","WISEM2","MEDIUM","GALAXY","CLUSTER")):
    """Decodes data or random maskbit array and returns dictionary of individual masks"""
    mapping = [x.split("\t") for x in """
	0	NPRIMARY	touches a pixel that is outside the BRICK_PRIMARY region of a brick
	1	BRIGHT	touches a pixel within the locus of a radius-magnitude relation for Tycho-2 stars or one for Gaia DR2 stars to G < 13
	2	SATUR_G	touches a pixel that was saturated in at least one g-band image
	3	SATUR_R	touches a pixel that was saturated in at least one r-band image
	4	SATUR_Z	touches a pixel that was saturated in at least one z-band image
	5	ALLMASK_G	touches a pixel that has any of the ALLMASK_G bits set
	6	ALLMASK_R	touches a pixel that has any of the ALLMASK_R bits set
	7	ALLMASK_Z	touches a pixel that has any of the ALLMASK_Z bits set
	8	WISEM1	touches a pixel in a WISEMASK_W1 bright star mask
	9	WISEM2	touches a pixel in a WISEMASK_W2 bright star mask
	10	BAILOUT	touches a pixel in a blob where we "bailed out" of source fitting
	11	MEDIUM	touches a pixel within the locus of a radius-magnitude relation for Gaia DR2 stars to G < 16
	12	GALAXY	touches a pixel in an SGA large galaxy
	13	CLUSTER	touches a pixel in a globular cluster""".split("\n") if x != ""]
    
    mapping = {int(k): v for _, k, v, _ in mapping}
    lookup_bit_by_name = {v: k for k, v in mapping.items()}
    masks = {k: (maskbits & (1 << lookup_bit_by_name[k]) != 0) for k in lookup_bit_by_name.keys()}
    masks["ALL_MASKS"] = np.any(np.array([masks[k] for k in masks.keys()]),axis=0)
    masks["COMBINED"] = np.any(np.array([masks[k] for k in keys]),axis=0)
    return masks



def masks_from_fitbits(fitbits, keys=("MEDIUM","GAIA","TYCHO2")):
    """Decodes fitbit mask array and returns dictionary of individual masks"""
    mapping = [x.split("\t") for x in """
	0	FORCED_POINTSOURCE	the source was forced to be type PSF
	1	FIT_BACKGROUND	the source had its background levels independently fit
	2	HIT_RADIUS_LIMIT	the source hit the radius limit during fitting (based on the limits for galaxy models in the Tractor code)
	3	HIT_SERSIC_LIMIT	the source hit the Sersic index limit during fitting (e.g., see the Sersic model limits in the Tractor code)
	4	FROZEN	the source was not fit (all parameters for the source were frozen at the reference catalog values)
	5	BRIGHT	the source is a bright star
	6	MEDIUM	the source is a medium-bright star
	7	GAIA	the source is a Gaia source
	8	TYCHO2	the source is a Tycho-2 star
	9	LARGEGALAXY	the source is an SGA large galaxy
	10	WALKER	fitting the source shifted its position by > 1 arcsec
	11	RUNNER	fitting the source shifted its position by > 2.5 arcsec
	12	GAIA_POINTSOURCE	the source was a Gaia source that was treated as a point source
	13	ITERATIVE	the source was found during iterative detection""".split("\n") if x != ""]
    
    mapping = {int(k): v for _, k, v, _ in mapping}
    lookup_bit_by_name = {v: k for k, v in mapping.items()}
    masks = {k: (fitbits & (1 << lookup_bit_by_name[k]) != 0) for k in lookup_bit_by_name.keys()}
    masks["ALL_MASKS"] = np.any(np.array([masks[k] for k in masks.keys()]),axis=0)
    masks["STARS"] = np.any(np.array([masks[k] for k in keys]),axis=0)
    return masks



def get_coord_mask(RA,Dec,field="south"):
    """coordinate masks (RA/Dec) for north/south data and random catalogs;
    trims data and random to same geometric area"""
    if field.lower()=="south":
        s1, b1 = _mb((307,  0),    (320, 20))
        s2, b2 = _mb((320, 20),    (333, 30))
        s3, b3 = _mb((106, 32.375),(127, -9))
        s4, b4 = _mb((268, 20),    (275, 32.375))
        s5, b5 = _mb((  0,-34),    ( 10,-20))    
        m = [( (Dec<-65.8) | (Dec>32.375) | ( (Dec>1) & (RA>64) & (RA<76) ) | ( (RA>74.5) & (RA<80) & (Dec<0) & (Dec>-13) ) ),
             ( (Dec>-40) & (Dec<-14) & (RA>300) ),
             ( (RA>300) & (RA<303) & (Dec<-59) ),
             ( (RA>300) & (RA<305.5) & (Dec>-50) & (Dec<-40) ),
             ( (RA>280) & ( (RA<300) | (Dec-b1>s1*RA) | (Dec-b2>s2*RA) ) ),
             ( (RA>280) & (RA<308.25) & (Dec<5) & (Dec>-11) ),
             ( (Dec-b3<s3*RA) & (RA>101) ),
             ( ( (RA>100) & (RA<280) ) & ( ( (Dec-b4<s4*RA) & (RA>268) ) | ( (Dec<-9) & (Dec>-35) ) ) ),
             ( (Dec>-40) & (Dec<-14.5) & (RA<10.5) & (Dec-b5>s5*RA) ),
            ]
        mask = ( m[0] | m[1] | m[2] | m[3] | m[4] | m[5] | m[6] | m[7] | m[8] )
    elif field.lower()=="north":
        s1, b1 = _mb((283.5,75.5),  (298,  67))
        s2, b2 = _mb((279,  32.375),(287,  47))
        s3, b3 = _mb((287,  47),    (300,  65))
        s4, b4 = _mb(( 93,  55),    ( 99.5,35))
        s5, b5 = _mb(( 95,  80),    (104,  74))
        s6, b6 = _mb(( 95,  80),    (104.5,81))
        s7, b7 = _mb(( 98,  70),    (103,  72.5))
        s8, b8 = _mb((284,  80),    (280,  81))
        m = [( (Dec < 32.375) | (Dec > 82.25) | (RA<90) | (RA>299.5) | ( (RA>260) & (Dec>81) ) | ( (RA>283.5) & ( (Dec>75.5) | (Dec-b1>s1*RA) ) ) ),
             ( ( (Dec-b4<s4*RA) & (Dec<56) ) | ( (Dec-b5<s5*RA) & (Dec>74) ) | ( (RA<104.5) & (Dec-b6>s6*RA) ) | ( (RA<93) & (Dec>62) & (Dec<70) ) ),
             ( ( (Dec-b7>s7*RA) & (Dec>70) & (Dec<74) & (RA<103.5) ) | ( (Dec>61) & (Dec<63) & (RA<95.5) ) | ( (Dec>65.5) & (RA>297.5) ) ),
             ( ( (Dec<47) & (Dec-b2<s2*RA) ) | ( (Dec>47) & (Dec-b3<s3*RA) ) | ( (RA<98) & (Dec>70) & (Dec<74) ) | (Dec-b8>s8*RA) ),
            ]
        mask = ( m[0] | m[1] | m[2] | m[3] )
    else:
        raise Exception("field must be either 'north' or 'south'")
    return mask



#-- slope and y-intercept of line given two input points
#-- used with certain mask functions
def _mb(p0,p1):
    (x0,y0) = p0
    (x1,y1) = p1
    s = (y1-y0)/(x1-x0)
    b = y0 - s*x0
    return s, b
    
    