# remove "_isnull" columns;
# vstack the catalogs (originally separated to circumvent the SQL size limit)

from __future__ import division, print_function
import sys, os, time, argparse, glob, gc
import numpy as np
from astropy.table import Table, vstack
import fitsio

input_dir = '/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/pdr3_col_cuts'
output_dir = '/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat'

hsc_list = ['hsc-pdr3-wide-cosmos-reduced.fits',
            'hsc-pdr3-wide-xmm-reduced.fits',
            'hsc-pdr3-wide-aegis-reduced.fits',
            'hsc-pdr3-wide-autumn-reduced.fits',
            'hsc-pdr3-wide-hectomap-reduced.fits',
            'hsc-pdr3-wide-spring-reduced.fits'
            ]

# hsc_list = ['hsc-pdr3_prim_cut-wide-hectomap-reduced.fits',
#             'hsc-pdr3_prim_cut-wide-spring-reduced.fits',
#             'hsc-pdr3_prim_cut-wide-autumn-reduced.fits',
#             ]

output_fn = "hsc-pdr3-wide-nocut.fits"
output_path = os.path.join(output_dir, output_fn)
if os.path.isfile(output_path):
    print('File already exists:', output_path)

cat_vstack = []

for fn in hsc_list:
    print(fn)

    cat = fitsio.read(os.path.join(input_dir, fn))
    cat = Table(cat)
    print(len(cat))

    cat_vstack.append(cat)

cat_vstack = vstack(cat_vstack)

print(output_path)
cat_vstack.write(output_path)

gc.collect()