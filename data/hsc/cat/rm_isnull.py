import fitsio
import os
from astropy.table import Table
input_dir = "."

cat = fitsio.read('hscs19a.fits')
cat = Table(cat)
print(len(cat))

for colname in cat.colnames:
    if colname.endswith('_isnull'):
        cat.remove_column(colname)

cat.write('/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscs19a_nocut.fits')