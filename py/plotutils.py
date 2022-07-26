import numpy as np
import matplotlib.pyplot as plt


plot_settings = {
    "font.size": 16,
    "axes.linewidth": 1.0,
    "xtick.major.size": 6.0,
    "xtick.minor.size": 4.0,
    "xtick.major.width": 1.5,
    "xtick.minor.width": 1.0,
    "xtick.direction": "in", 
    "xtick.minor.visible": True,
    "xtick.top": True,
    "ytick.major.size": 6.0,
    "ytick.minor.size": 4.0,
    "ytick.major.width": 1.5,
    "ytick.minor.width": 1.0,
    "ytick.direction": "in", 
    "ytick.minor.visible": False,
    "ytick.right": True,
}


fig_labels = {"vpeak"     : r"$v_{\rm peak}\ [{\rm km\ s}^{-1}]$",
              "log_vpeak" : r"$\log(v_{\rm peak}/{\rm km\ s}^{-1})$",
              "vmax"      : r"$v_{\rm max}\ [{\rm km\ s}^{-1}]$",
              "log_vmax"  : r"$\log(v_{\rm max}/{\rm km\ s}^{-1})$",
              "n_eff"     : r"$n_{\rm eff} \left(h^3{\rm Mpc}^{-3}\right)$",
              "rp"        : r"$r_{\rm p}\ \left(h^{-1}{\rm Mpc}\right)$",
              "wp"        : r"$\omega_{\rm p}(r_{\rm p})\ \left(h^{-1}{\rm Mpc}\right)$",
              "rpwp"      : r"$r_{\rm p}\times\omega_{\rm p}(r_{\rm p})\ \left(h^{-2}{\rm Mpc}^2\right)$",
             }



def get_colors(N, c1="plasma", c2="viridis_r"):
    cm1 = plt.get_cmap(c1)
    cm2 = plt.get_cmap(c2)
    if N < 5:
        colors = [cm1((i+1/N)/N) for i in range(N)]
    else:
        c = np.concatenate([ [cm1(i/(N-1)) for i in range(N-1)], [cm2(i/(N-1)) for i in range(N)] ])
        if (N==5):
            colors = c[1:6]
        else:
            colors = [ c[int(i%len(c))] for i in range(len(c)) ][::2]
    return colors



def get_corners(ax, margin=0.05, log=False, logx=False, logy=False):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
       
    if log:
        logx = True
        logy = True
        xmin, xmax = np.log10(xmin), np.log10(xmax)
        ymin, ymax = np.log10(ymin), np.log10(ymax)
        xr, yr = (xmax - xmin), (ymax - ymin)
        corners = dict(upper_left   = (10**(xmin + margin*xr), 10**(ymax - margin*yr)),
                       upper_right  = (10**(xmax - margin*xr), 10**(ymax - margin*yr)),
                       upper_center = (10**(xmin + 0.5*xr),    10**(ymax - margin*yr)),
                       lower_right  = (10**(xmax - margin*xr), 10**(ymin + margin*yr)),
                       lower_left   = (10**(xmin + margin*xr), 10**(ymin + margin*yr)),
                       lower_center = (10**(xmin + 0.5*xr),    10**(ymin + margin*yr)),
                      )        

    elif (logx and not logy):
        xmin, xmax = np.log10(xmin), np.log10(xmax)
        xr, yr = (xmax - xmin), (ymax - ymin)
        corners = dict(upper_left   = (10**(xmin + margin*xr), (ymax - margin*yr)),
                       upper_right  = (10**(xmax - margin*xr), (ymax - margin*yr)),
                       upper_center = (10**(xmin + 0.5*xr),    (ymax - margin*yr)),
                       lower_right  = (10**(xmax - margin*xr), (ymin + margin*yr)),
                       lower_left   = (10**(xmin + margin*xr), (ymin + margin*yr)),
                       lower_center = (10**(xmin + 0.5*xr),    (ymin + margin*yr)),
                      )        

    elif (logy and not logx):
        ymin, ymax = np.log10(ymin), np.log10(ymax)
        xr, yr = (xmax - xmin), (ymax - ymin)
        corners = dict(upper_left   = ((xmin + margin*xr), 10**(ymax - margin*yr)),
                       upper_right  = ((xmax - margin*xr), 10**(ymax - margin*yr)),
                       upper_center = ((xmin + 0.5*xr),    10**(ymax - margin*yr)),
                       lower_right  = ((xmax - margin*xr), 10**(ymin + margin*yr)),
                       lower_left   = ((xmin + margin*xr), 10**(ymin + margin*yr)),
                       lower_center = ((xmin + 0.5*xr),    10**(ymin + margin*yr)),
                      )        

    else:
        xr, yr = (xmax - xmin), (ymax - ymin)
        corners = dict(upper_left   = (xmin + margin*xr, ymax - margin*yr),
                       upper_right  = (xmax - margin*xr, ymax - margin*yr),
                       upper_center = (xmin + 0.5*xr,    ymax - margin*yr),
                       lower_right  = (xmax - margin*xr, ymin + margin*yr),
                       lower_left   = (xmin + margin*xr, ymin + margin*yr),
                       lower_center = (xmin + 0.5*xr,    ymin + margin*yr),
                      )
    return corners
