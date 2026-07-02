import fitsio as fio
import matplotlib.pyplot as plt
import numpy as np

# 1. Chargement des données et des aires (en arcmin²)
area_s19a_nocut = 455 * 3600
label_s19a_nocut = "S19A"
hsc_s19a_nocut = fio.FITS("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscy3_cat_withflags.fits")[1].read()

area_pdr3_cut = 1106 * 3600
label_pdr3_cut = "PDR3-highz"
hsc_pdr3_cut = fio.FITS("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscpdr3_highz_final_maj.fits")[1].read()
hsc_pdr3_cut = hsc_pdr3_cut[(hsc_pdr3_cut["dnnz_photoz_best"] > 0.9) | ((hsc_pdr3_cut["dnnz_photoz_best"] <= 0.9) & (hsc_pdr3_cut["dnnz_photoz_err95_max"] - hsc_pdr3_cut["dnnz_photoz_err95_min"] < 2.7) & (hsc_pdr3_cut["mizuki_photoz_err95_max"] - hsc_pdr3_cut["mizuki_photoz_err95_min"] < 2.7))]

bin_width = 0.1
tomo_bins = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.7], dtype=np.float64)
bins = np.arange(0.0, 4.0, bin_width, dtype=np.float64)

z_s19a_nocut = hsc_s19a_nocut["dnnz_photoz_best"].astype(np.float64)
z_pdr3_cut = hsc_pdr3_cut["dnnz_photoz_best"].astype(np.float64)

plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots(constrained_layout=True, figsize=(11, 8))

cmap = plt.get_cmap("plasma")

label = "PDR3 high-z"
ax.hist(z_pdr3_cut, bins=bins, histtype='step', linewidth=2.5, color=cmap(0.2),
                weights=np.ones_like(z_pdr3_cut) / (area_pdr3_cut), label=label)

label = "S19A & no calibration cut"
ax.hist(z_s19a_nocut, bins=bins, histtype='step', linewidth=2.5, color=cmap(0.4),
                weights=np.ones_like(z_s19a_nocut) / (area_s19a_nocut), label=label)

cuts = [2.7, 2.5]
for i, cut in enumerate(cuts):
        
        z_s19a_cut = z_s19a_nocut[(hsc_s19a_nocut["dnnz_photoz_err95_max"] - hsc_s19a_nocut["dnnz_photoz_err95_min"] < cut) & (hsc_s19a_nocut["mizuki_photoz_err95_max"] - hsc_s19a_nocut["mizuki_photoz_err95_min"] < cut)].astype(np.float64)        
        
        label = fr"S19A & $(z_{{\text{{err,95}}}}^{{\max}} - z_{{\text{{err,95}}}}^{{\min}}) < {cut}$ (DNNz & Mizuki)"
        if cut == 2.7:
                ax.hist(z_s19a_cut, bins=bins, histtype='step', linewidth=2.5, color=cmap(0.6),
                weights=np.ones_like(z_s19a_cut) / (area_s19a_nocut), label=label)
        else:
                ax.hist(z_s19a_cut, bins=bins, histtype='step', linewidth=2.5, color=cmap(0.8),
                weights=np.ones_like(z_s19a_cut) / (area_s19a_nocut), label=label)

ax.axvspan(1.5, 2.0, color="gray", alpha=0.2, label="Bin 5")
ax.axvspan(2.0, 2.7, color="gray", alpha=0.4, label="Bin 6")

ax.set_xlabel(r"$z_{\mathrm{best}}^{\mathtt{DNNz}}$")
ax.set_ylabel(r"Number density $\mathrm{d}N/\mathrm{d}\Omega$ ($\mathrm{arcmin}^{-2}$)") 
ax.set_yscale("log")
ax.set_xlim(0, 3.0)
# ax.set_ylim(2e-3, None)
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, which="both", ls="--", alpha=0.5)

# Saving
import os
os.makedirs("data_analysis_figs", exist_ok=True)
output_file = "paper_figs/density_comparison_all.png"
plt.savefig(output_file, dpi=300)
print(f"Plot saved in {output_file}")