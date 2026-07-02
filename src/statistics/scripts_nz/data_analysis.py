## Imports
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import fitsio as fio
import skymapper as skm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import re
import healpy as hp

from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.table import Table, vstack, unique
from mocpy import MOC
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

## to look at the distribution of target classes in DESI on the plot
from desitarget.targetmask import desi_mask

## file manager; replace with your own module if needed

import sys
sys.path.append("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/src/statistics")
import corrfiles as cf
import inference as inf
import combination as comb
import cosmotools as ct
import sgp as sgp
import importlib
import scipy.interpolate as interp
sys.path.append("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/src")
import analysis.maps as maps
import analysis.plots as plots

importlib.reload(plots)

DESI_ROOT_DR2 = Path(
    "/global/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/nonKP"
)
PAPER_FIGURES_ROOT = Path(
    "/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/paper/figures/"
)
pm = plots.PlotManager(root=PAPER_FIGURES_ROOT, overwrite=True)

# import catalog
hsc = "highz" # "s19a", "nocut", "highz"
if hsc == "s19a":
    area = 455 * 3600
    label = "S19A"
    hsc = fio.FITS("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscy3_cat_withflags.fits")[1].read()
elif hsc == "nocut":
    area = 1206 * 3600
    label = "PDR3-nocut"
    hsc = fio.FITS("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hsc-pdr3-wide-nocut.fits")[1].read()
elif hsc == "highz":
    area = 1106 * 3600
    label = "PDR3-highz"
    hsc = fio.FITS("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/data/hsc/cat/hscpdr3_highz_final_maj.fits")[1].read()
    
    # # generating moc pdr3
    # coords_pdr3 = SkyCoord(ra=hsc["ra"] * u.deg, dec=hsc["dec"] * u.deg)
    # moc_pdr3 = MOC.from_skycoords(coords_pdr3, max_norder=12)

    # # generating moc s19a
    # coords_s19a = SkyCoord(ra=hsc_s19a["ra"] * u.deg, dec=hsc_s19a["dec"] * u.deg)
    # moc_s19a = MOC.from_skycoords(coords_s19a, max_norder=12)

    # # match footprint
    # moc_matched = moc_pdr3.intersection(moc_s19a)
    
    # in_matched_footprint = moc_matched.contains(
    # hsc["ra"] * u.deg, hsc["dec"] * u.deg
    # )

    # # 2. apply footprint cut
    # hsc = hsc[in_matched_footprint]
    # print(len(hsc))


# plot density
def density():
    ra = np.asarray(hsc["ra"])      # degrés
    dec = np.asarray(hsc["dec"])    # degrés

    nside = 4096

    # coordonnées sphériques
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    # pixels occupés
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    mask = np.zeros(hp.nside2npix(nside))
    mask[pix] = 1
    input_mask = mask
    threshold = 0.1
    if np.issubdtype(input_mask.dtype, np.bool_):
        area = float(np.sum(input_mask)) / len(input_mask) * 4 * np.pi * (180 / np.pi) ** 2
    else:
        area = len(input_mask[input_mask > threshold]) / len(input_mask) * 4 * np.pi * (180 / np.pi) ** 2
    print(area)
    
    area *= 3600 # deg² to arcmin²
    
    bin_width = 0.1
    tomo_bins = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 2, 2.5, 3, 3.5], dtype=np.float64)
    bins = np.arange(0.0, 4.0, bin_width, dtype=np.float64)
    
    z = hsc["dnnz_photoz_best"].astype(np.float64)
    
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(constrained_layout=True, figsize=(11, 8))
    
    ax.hist(z, bins=bins, histtype='step', linewidth=2.5, color="C0", weights=np.ones_like(z) / area, label=label)
    
    ax.axvspan(1.5, 2.0, color="gray", alpha=0.15, label="Bin 5")
    ax.axvspan(2.0, 2.5, color="gray", alpha=0.30, label="Bin 6")

    ax.set_xlabel(r"$z_{\mathrm{photo}}$")
    ax.set_ylabel(r"Number density $\mathrm{d}N/\mathrm{d}\Omega$ ($\mathrm{arcmin}^{-2}$)") 
    ax.set_yscale("log")
    ax.set_xlim(0, 3.5)
    ax.set_ylim(2e-3, None)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Saving
    import os
    os.makedirs("data_analysis_figs", exist_ok=True)
    output_file = f"data_analysis_figs/density_{label}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved in {output_file}")

# plot imag
def imag():
    bin_width = 0.1

    i_mag = hsc["i_cm_mag"][:] - hsc["a_i"][:]
    bins = np.arange(16, 26, bin_width)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        i_mag,
        bins=bins,
        histtype="stepfilled",
        linewidth=1.2,
        label=label,
    )
    ax.legend()
    ax.set_xlabel(r"$i_{cm\_mag}$")
    ax.set_ylabel(r"Number of objects")
    ax.set_ylim(0,1.5e7)
    plt.savefig(f"data_analysis_figs/imag_{label}.png", dpi=300)

# plot imagerr
def imagerr():
    bin_width = 0.005

    i_mag = hsc["i_cm_magerr"][:]
    bins = np.arange(0, 0.12, bin_width)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        i_mag,
        bins=bins,
        histtype="stepfilled",
        linewidth=1.2,
        label=label,
    )
    ax.legend()
    ax.set_xlabel(r"$i_{cm\_magerr}$")
    ax.set_ylabel(r"Number of objects")
    ax.set_ylim(0, 1.5e7)
    plt.savefig(f"data_analysis_figs/imagerr_{label}.png", dpi=300)

# calcul footprint
def footprint_cat():
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
    return area_deg2

# plot footprint
def plot_footprint():
    caps = ["NGC", "SGC"]
    tgts_dr1 = ["BGS_ANY", "LRG", "ELG_LOPnotqso", "QSO"]
    desi_tgts_dr1 = {
        (tgt, cap): cf.fetch_desi_files(tgt=tgt, version="DR1", cap=cap)
        for tgt in tgts_dr1
        for cap in caps
    }
    tgts_dr2 = ["BGS_ANY", "LRG", "ELGnotqso", "QSO"]
    desi_tgts_dr2 = {
        (tgt, cap): cf.fetch_desi_files(tgt=tgt, version="DR2", cap=cap)
        for tgt in tgts_dr2
        for cap in caps
    }

    desi_tbls_dr1 = {}
    print("Reading DR1 catalogs...")
    for tracer in tgts_dr1:
        desi_cat = [desi_tgts_dr1[(tracer, cap)] for cap in caps]
        dt = vstack([Table(fio.read(dc, columns=["RA", "DEC", "Z"])) for dc in desi_cat])
        desi_tbls_dr1[tracer] = dt
    desi_dr1 = vstack(list(desi_tbls_dr1.values()))
    
    desi_tbls_dr2 = {}
    print("Reading DR2 catalogs...")
    for tracer in tgts_dr2:
        desi_cat = [desi_tgts_dr2[(tracer, cap)] for cap in caps]
        dt = vstack([Table(fio.read(dc, columns=["RA", "DEC", "Z"])) for dc in desi_cat])
        desi_tbls_dr2[tracer] = dt
    desi_dr2 = vstack(list(desi_tbls_dr2.values()))
    
    hsc_coords = SkyCoord(ra=hsc["ra"] * u.deg, dec=hsc["dec"] * u.deg)
    desi_coords_dr1 = SkyCoord(ra=desi_dr1["RA"] * u.deg, dec=desi_dr1["DEC"] * u.deg)
    desi_coords_dr2 = SkyCoord(ra=desi_dr2["RA"] * u.deg, dec=desi_dr2["DEC"] * u.deg)
    
    nside = 512
    smoothing = 0.07 * u.deg
    sep = 15
    show_filled = True

    with pm.make_plot(
        "footprint", figsize=(18, 9), tight_layout=False, custom_layout=True, show=True
    ) as fig:

        proj = skm.Hammer()
        footprint = skm.Map(proj, facecolor="white", ax=fig.gca())

        pixels, rap, decp, vertices = skm.healpix.getGrid(nside, return_vertices=True)

        # Galactic plane
        l = np.linspace(0, 360, 4000) * u.deg
        b = np.zeros(4000) * u.deg
        gal = SkyCoord(l=l, b=b, frame="galactic")
        eq = gal.icrs
        ra = eq.ra.wrap_at(180 * u.deg)
        dec = eq.dec
        galactic_plane = SkyCoord(ra=ra, dec=dec, frame="icrs")

        survey_list = [desi_coords_dr2, desi_coords_dr1, hsc_coords]
        survey_names = ["DESI DR2", "DESI DR1", f"HSC {label}"]
        survey_colors = ["midnightblue", "steelblue", "darkred"]
        xpos_list = [-62.5, 0, 3]
        ypos_list = [15, 35, 7.5]
        alphas_list = [1, 1, 1]

        for survey, sname, posx, posy, color, alpha in zip(
            survey_list, survey_names, xpos_list, ypos_list, survey_colors, alphas_list
        ):
            pix_file = (
                f"/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/src/analysis/data/maps/pix_{sname}_nside_{nside}_smooth_{smoothing.value}".replace(
                    " ", ""
                )
                + ".npy"
            )
            vert_file = (
                f"/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/src/analysis/data/maps/vert_{sname}_nside_{nside}_smooth_{smoothing.value}".replace(
                    " ", ""
                )
                + ".npy"
            )

            if os.path.exists(pix_file) and os.path.exists(vert_file):
                pix = np.load(pix_file, allow_pickle=True)
                vert = np.load(vert_file, allow_pickle=True)
            else:
                pix, vert, _ = maps.put_survey_on_grid(
                    survey.ra.deg,
                    survey.dec.deg,
                    rap,
                    decp,
                    pixels,
                    vertices,
                    smoothing=smoothing,
                )
                os.makedirs("/global/cfs/projectdirs/desi/users/qlavier/desi-y3-hsc/src/analysis/data/maps", exist_ok=True)
                np.save(vert_file, vert)
                np.save(pix_file, pix)

            if sname != f"HSC {label}" and show_filled:
                ext_vert = vert
                footprint.vertex(
                    ext_vert,
                    facecolors=color,
                    edgecolors=color,
                    lw=1.2,
                    alpha=1.0,
                    label=sname,
                )
            else:
                ext_vert = vertices[maps.get_boundary_mask(pix, nside, niter=2)]
                footprint.vertex(
                    ext_vert,
                    facecolors=color,
                    edgecolors=None,
                    lw=1.2,
                    alpha=1.0,
                    label=sname,
                )
            txt = footprint.ax.text(
                np.deg2rad(posx),
                np.deg2rad(posy),
                sname,
                size=25,
                color=color,
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        pix, vert, _ = maps.put_survey_on_grid(
            galactic_plane.ra.deg,
            galactic_plane.dec.deg,
            rap,
            decp,
            pixels,
            vertices,
            smoothing=smoothing,
        )
        footprint.vertex(vert, facecolors="gray", alpha=0.6, lw=1)

        handles, labels = footprint.ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="gray", lw=1, ls="-"))
        labels.append("Galactic plane")
        footprint.ax.legend(handles, labels, loc="upper left", fontsize=14)
        
        footprint.grid(sep=sep)
        for artist in footprint.artists("grid-"):
            gid = artist.get_gid()
            if "np.int64(" in gid:
                import re

                match = re.search(r"np\.int64\((-?[0-9]+)\)", gid)
                if match:
                    number = int(match.group(1))
                    number = ((number + 180) % 360) - 180  # wrap into [-180,180]
                    if "parallel" in gid:
                        new_gid = f"grid-parallel-{number}"
                    elif "meridian" in gid:
                        new_gid = f"grid-meridian-{number}"
                    artist.set_gid(new_gid)

        print(artist.get_gid())

        footprint.labelMeridiansAtParallel(0, loc="bottom")
        footprint.labelParallelsAtMeridian(footprint.proj.lon_0)
        for artist in footprint.ax.get_children():
            if hasattr(artist, "get_gid") and artist.get_gid() is not None:
                import re
                if re.match(r"parallel-label-0(\.0+)?", artist.get_gid()):
                    artist.remove()
        
        text = f"HSC area : {footprint_cat():.0f} deg²"

        footprint.ax.text(
            0.5, 0.02,
            text,
            transform=footprint.ax.transAxes,
            fontsize=14,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        
        plt.savefig(f"paper_figs/footprint_{label}.png", dpi=300)