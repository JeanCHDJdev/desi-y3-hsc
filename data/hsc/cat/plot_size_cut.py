import numpy as np
import matplotlib.pyplot as plt
import fitsio as fio

cat = fio.FITS("hsc-pdr3-wide-nocut.fits")[1].read()

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(10, 8))

cut = (cat['i_sdssshape_shape11'] + cat['i_sdssshape_shape22'] > 0)

Timage = cat['i_sdssshape_shape11'][cut] + cat['i_sdssshape_shape22'][cut]
TPSF = cat['i_sdssshape_psf_shape11'][cut] + cat['i_sdssshape_psf_shape22'][cut]
R2 = 1 - TPSF/Timage

imag = cat['i_cm_mag'][cut] - cat['a_i'][cut]

imag_withcut=imag[R2>=0]
R2_withcut = R2[R2>=0]

plt.scatter(imag_withcut, R2_withcut, s=0.000004, color='C0')
x = np.arange(10, 25.1, 0.1)
plt.plot(x, np.minimum(0.2, -2/15 * (x - 24)), color='red')

plt.xlabel(r'$\text{i}_{\text{mag}} - \text{a}_\text{i}$')
plt.ylabel(r'$\text{R}_2$')
plt.xlim(10, 25.1)
plt.ylim(0, 1)
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()
plt.savefig(f'/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/src/statistics/scripts_nz/paper_figs/i_vs_R2.png', bbox_inches='tight', dpi=300)
