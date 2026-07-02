import numpy as np
import fitsio as fio
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from mocpy import MOC

# 1. Loading catalog
print("Reading catalog...")
hsc_base = fio.FITS("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscy3_cat_withflags.fits")[1].read()

# --- OPTIONNal : Match with S19A footprint ---
# To activate, uncomment the following lines :
# print("Matching with S19A footprint...")
# coords_pdr3 = SkyCoord(ra=hsc_base["ra"] * u.deg, dec=hsc_base["dec"] * u.deg)
# moc_pdr3 = MOC.from_skycoords(coords_pdr3, max_norder=12) # norder=11 ou max_depth=11 selon la version
# hsc_s19a = fio.FITS("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscy3_cat_withflags.fits")[1].read()
# coords_s19a = SkyCoord(ra=hsc_s19a["ra"] * u.deg, dec=hsc_s19a["dec"] * u.deg)
# moc_s19a = MOC.from_skycoords(coords_s19a, max_norder=12)
# moc_matched = moc_pdr3.intersection(moc_s19a)
# in_matched_footprint = moc_matched.contains(hsc_base["ra"] * u.deg, hsc_base["dec"] * u.deg)
# hsc_base = hsc_base[in_matched_footprint]

# Name of the cuts
cut_names = ["no cut",
    "bimodal 2.9",
    "bimodal 2.7",
    "bimodal 2.5",
    "bimodal 2.3"
]

print("\nStarting sequential cut analysis...")
print(f"Total objects in raw catalog: {len(hsc_base)}")
print("-" * 60)

for idx, name in enumerate(cut_names):
    # 2. Apply each cut individually, then all together
    if idx == 0: cut = np.ones(len(hsc_base), dtype=bool)
    elif idx == 1: cut = ((hsc_base['dnnz_photoz_err95_max'] - hsc_base['dnnz_photoz_err95_min'] < 2.9) &
                          (hsc_base['mizuki_photoz_err95_max'] - hsc_base['mizuki_photoz_err95_min'] < 2.9))
    elif idx == 2: cut = ((hsc_base['dnnz_photoz_err95_max'] - hsc_base['dnnz_photoz_err95_min'] < 2.7) &
                          (hsc_base['mizuki_photoz_err95_max'] - hsc_base['mizuki_photoz_err95_min'] < 2.7))
    elif idx == 3: cut = ((hsc_base['dnnz_photoz_err95_max'] - hsc_base['dnnz_photoz_err95_min'] < 2.5) &
                          (hsc_base['mizuki_photoz_err95_max'] - hsc_base['mizuki_photoz_err95_min'] < 2.5))
    elif idx == 4: cut = ((hsc_base['dnnz_photoz_err95_max'] - hsc_base['dnnz_photoz_err95_min'] < 2.3) &
                          (hsc_base['mizuki_photoz_err95_max'] - hsc_base['mizuki_photoz_err95_min'] < 2.3))

    hsc = hsc_base[cut]

    print(f"\n[CUT {idx+1}/{len(cut_names)}] Applied: {name}")
    print(f"Remaining objects: {len(hsc)}")

    # 3. Computation of the area of the footprint
    ra = np.asarray(hsc["ra"])
    dec = np.asarray(hsc["dec"])
    nside = 4096

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    pix = hp.ang2pix(nside, theta, phi, nest=False)
    
    unique_pix = np.unique(pix)
    num_unique_pix = len(unique_pix)
    total_pix = hp.nside2npix(nside)
    
    # Area of the footprint in square degrees
    area_deg2 = (num_unique_pix / total_pix) * 41252.9612
    print(f"Calculated Area: {area_deg2:.0f} deg²")

    # 4. Computation of the number density in the bins
    area_arcmin2 = area_deg2 * 3600.0  # 1 deg² = 3600 arcmin²
    z = hsc["dnnz_photoz_best"].astype(np.float64)

    tomo_bins = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.7], dtype=np.float64)
    accuracy = 3

    print("Number density per tomographic bin:")
    for i in range(len(tomo_bins) - 1):
        zmin, zmax = tomo_bins[i], tomo_bins[i + 1]
        if i == 0:
            z_mask = (z >= zmin) & (z <= zmax)
        else:
            z_mask = (z > zmin) & (z <= zmax)
        n_galaxies = np.sum(z_mask)
        density = n_galaxies / area_arcmin2
        if i == 0:
            print(f"  {zmin:3.1f} <= z <= {zmax:3.1f} : {density:.{accuracy}g} arcmin^-2 (N={n_galaxies})")
        else:
            print(f"  {zmin:3.1f} < z <= {zmax:3.1f} : {density:.{accuracy}g} arcmin^-2 (N={n_galaxies})")
    
    # Last bin for z > 2.7
    last_mask = z > 2.7
    n_last = np.sum(last_mask)
    density_last = n_last / area_arcmin2
    print(f"  2.7 < z        : {density_last:.{accuracy}g} arcmin^-2 (N={n_last})")
    print("-" * 60)