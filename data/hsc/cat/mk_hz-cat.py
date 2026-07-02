from astropy.table import Table
import numpy as np

# 1. Reading catalogs
print("Reading catalogs...")
cat_pdr3_nocut = Table.read("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hsc-pdr3-wide-nocut.fits")

# 2. Apply cuts on PDR3
cut = ((cat_pdr3_nocut['i_detect_isprimary'] == 1) & 
        (cat_pdr3_nocut["i_deblend_skipped"] == 0) &
        (cat_pdr3_nocut["i_sdsscentroid_flag"] == 0) &
        (cat_pdr3_nocut["i_pixelflags_interpolatedcenter"] == 0) &
        (cat_pdr3_nocut["i_pixelflags_saturatedcenter"] == 0) &
        (cat_pdr3_nocut["i_pixelflags_crcenter"] == 0) &
        (cat_pdr3_nocut["i_pixelflags_bad"] == 0) &
        (cat_pdr3_nocut["i_pixelflags_suspectcenter"] == 0) &
        (cat_pdr3_nocut["i_pixelflags_clipped"] == 0) &
        (cat_pdr3_nocut["i_pixelflags_edge"] == 0) &
        (cat_pdr3_nocut["i_extendedness_value"] != 0) &
        (cat_pdr3_nocut['i_cm_flux']/cat_pdr3_nocut['i_cm_fluxerr'] >= 10) &
        (cat_pdr3_nocut["i_apertureflux_10_mag"] <= 25.5) &
        (cat_pdr3_nocut["i_blendedness_abs"] < 0.416869) &
        ~((cat_pdr3_nocut['ra'] >= 132.5) & (cat_pdr3_nocut['ra'] <= 140.0) & 
        (cat_pdr3_nocut['dec'] >= 1.6)  & (cat_pdr3_nocut['dec'] <= 5.0)) &
        (cat_pdr3_nocut['i_sdssshape_shape11'] + cat_pdr3_nocut['i_sdssshape_shape22'] > 0) &
        (1-((cat_pdr3_nocut['i_sdssshape_psf_shape11'] + cat_pdr3_nocut['i_sdssshape_psf_shape22'])/(cat_pdr3_nocut['i_sdssshape_shape11'] + cat_pdr3_nocut['i_sdssshape_shape22'])) >= 0) &
        (1-((cat_pdr3_nocut['i_sdssshape_psf_shape11'] + cat_pdr3_nocut['i_sdssshape_psf_shape22'])/(cat_pdr3_nocut['i_sdssshape_shape11'] + cat_pdr3_nocut['i_sdssshape_shape22'])) >= np.minimum(0.2, -2/15 * (cat_pdr3_nocut['i_cm_mag'] - cat_pdr3_nocut['a_i'] - 24)))
        )

cat_pdr3_withcut = cat_pdr3_nocut[cut]

# 3. Add columns weight and z_bin
print("Calculating tomographic binning columns...")

# Definition of tomographic bins (7 bins in total)
z_best = cat_pdr3_withcut["dnnz_photoz_best"].astype(np.float64)
tomo_bins = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.7], dtype=np.float64)
raw_bins = np.digitize(z_best, tomo_bins, right=True)

# Security to return objects outside the bounds in bin index 0 (final_bins = 1)
# Note : raw_bins >= 8 corresponds to objects with z > 2.7
final_bins = np.where((raw_bins >= 8) | (raw_bins == 0), 1, raw_bins)

# Index of bin starting at 0 (0 to 6)
z_bin = (final_bins - 1).astype(np.int32)

# Add columns directly in the Astropy Table
cat_pdr3_withcut['weight'] = np.ones(len(cat_pdr3_withcut), dtype=np.int32)
cat_pdr3_withcut['z_bin'] = z_bin

# 4. Writing the file
output_path = "/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscpdr3_highz_final_maj.fits"
print(f"Writing the final catalog to {output_path}...")
cat_pdr3_withcut.write(output_path, overwrite=True)
print("Done!")