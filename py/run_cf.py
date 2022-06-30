import sys
from clustering import run_cf
import numpy as np

config_file = sys.argv[1]
band        = sys.argv[2]
zmin        = np.round( float(sys.argv[3]), 1 )
zrange      = ( zmin, round(zmin+0.1,1) )

quiet = True if (("-q" in sys.argv[4:]) | ("--quiet" in sys.argv[4:])) else False

cmd_kwargs = dict(zrange=zrange, band=band, quiet=quiet)

run_cf( config_file, **cmd_kwargs )

