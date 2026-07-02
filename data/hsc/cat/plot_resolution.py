import numpy as np
import matplotlib.pyplot as plt
import fitsio as fio
from scipy.stats import pearsonr, spearmanr
from astropy.table import Table

cat = fio.FITS("hscs19a_nocut.fits")[1].read()

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(10, 8))

cut = (cat['i_sdssshape_shape11'] + cat['i_sdssshape_shape22'] > 0)

e1 = cat['i_sdssshape_shape11'][cut]
e2 = cat['i_sdssshape_shape22'][cut]
e1_psf = cat['i_sdssshape_psf_shape11'][cut]
e2_psf = cat['i_sdssshape_psf_shape22'][cut]

R2 = 1 - (e1_psf + e2_psf) / (e1 + e2)
resolution = cat['resolution'][cut]

R2_withcut = R2[(R2 >= 0) & (resolution >=0)]
resolution_withcut = resolution[(R2 >= 0) & (resolution >=0)]

plt.scatter(resolution_withcut, R2_withcut, s=0.000004, color='C0')
x = np.arange(0, 1.01, 0.1)
plt.plot(x, x, color='red', label="Identity")

r_coeff_p, _ = pearsonr(resolution_withcut, R2_withcut)
r_coeff_s, _ = spearmanr(resolution_withcut, R2_withcut)
plt.plot([], [], ' ', label=fr'$r_{{Pearson}}^2$ = {r_coeff_p**2:.3g}')
plt.plot([], [], ' ', label=fr'$r_{{Spearman}}^2$ = {r_coeff_s**2:.3g}')

plt.xlabel(r'$\text{resolution}$')
plt.ylabel(r'$\text{R}_2$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc="upper left")
plt.savefig(f'/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/src/statistics/scripts_nz/paper_figs/resolution_vs_R2.png', bbox_inches='tight', dpi=300)
