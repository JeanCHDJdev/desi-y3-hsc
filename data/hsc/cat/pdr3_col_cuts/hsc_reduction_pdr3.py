# remove "_isnull" columns;
# vstack the catalogs (originally separated to circumvent the SQL size limit)

from __future__ import division, print_function
import sys, os, time, argparse, glob, gc
import numpy as np
from astropy.table import Table, vstack
import fitsio

input_dir = '/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/pdr3_col_cuts'
output_dir = '/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/pdr3_col_cuts'

hsc_list = [['hsc-pdr3-wide-cosmos.fits'],
            ['hsc-pdr3-wide-xmm.fits'],
            ['hsc-pdr3-wide-aegis.fits'],
            ['hsc-pdr3-wide-autumn-1.fits',
             'hsc-pdr3-wide-autumn-2.fits',
             'hsc-pdr3-wide-autumn-3.fits',
             'hsc-pdr3-wide-autumn-4.fits',
             'hsc-pdr3-wide-autumn-5.fits',
             'hsc-pdr3-wide-autumn-6.fits',
             'hsc-pdr3-wide-autumn-7.fits',
             'hsc-pdr3-wide-autumn-8.fits',
             'hsc-pdr3-wide-autumn-9.fits',
             'hsc-pdr3-wide-autumn-10.fits'],
            ['hsc-pdr3-wide-hectomap-2.fits',
             'hsc-pdr3-wide-hectomap-3.fits',
             'hsc-pdr3-wide-hectomap-4.fits',
             'hsc-pdr3-wide-hectomap-5.fits'],
            ['hsc-pdr3-wide-spring-1.fits',
             'hsc-pdr3-wide-spring-2.fits',
             'hsc-pdr3-wide-spring-3.fits',
             'hsc-pdr3-wide-spring-4.fits',
             'hsc-pdr3-wide-spring-5.fits',
             'hsc-pdr3-wide-spring-6.fits',
             'hsc-pdr3-wide-spring-7.fits',
             'hsc-pdr3-wide-spring-8.fits',
             'hsc-pdr3-wide-spring-9.fits',
             'hsc-pdr3-wide-spring-10.fits'],
            ]

for index1 in range(len(hsc_list)):

    fn = hsc_list[index1][0]
    if len(hsc_list[index1])==1:
        output_fn = fn[:-5]+'-reduced.fits'
    else:
        output_fn = fn[:fn.rfind('-')]+'-reduced.fits'
    output_path = os.path.join(output_dir, output_fn)
    if os.path.isfile(output_path):
        print('File already exists:', output_path)
        continue

    cat_vstack = []

    for fn in hsc_list[index1]:
        print(fn)

        cat = fitsio.read(os.path.join(input_dir, fn))
        cat = Table(cat)
        print(len(cat))

        for colname in cat.colnames:
            if colname.endswith('_isnull'):
                cat.remove_column(colname)

        cat_vstack.append(cat)

    cat_vstack = vstack(cat_vstack)

    print(output_path)
    cat_vstack.write(output_path)

    gc.collect()