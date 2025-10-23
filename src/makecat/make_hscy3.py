"""
This script modifies the original `make_hscy3.py` to make a HSC Y3 catalog
with dnnz photo-zs.
"""

from astropy.table import Table, join, vstack, hstack
from glob import glob
from tqdm import tqdm
import fitsio as fio
import os
from pathlib import Path


def get_psf_ellip(catalog, return_shear=False):
    """This utility gets the PSF ellipticity (uncalibrated shear) from a data
    or sims catalog. Function taken from
    https://github.com/mr-superonion/utils_shear_ana/blob/c5030baacfb4c2ac3df9969605ffac96f804e35f/utils_shear_ana/catutil.py#L1781
    """
    if "e1_psf" in catalog.dtype.names:
        return catalog["e1_psf"], catalog["e2_psf"]
    elif "base_SdssShape_psf_xx" in catalog.dtype.names:
        psf_mxx = catalog["base_SdssShape_psf_xx"] * 0.168**2.0
        psf_myy = catalog["base_SdssShape_psf_yy"] * 0.168**2.0
        psf_mxy = catalog["base_SdssShape_psf_xy"] * 0.168**2.0
    elif "i_sdssshape_psf_shape11" in catalog.dtype.names:
        psf_mxx = catalog["i_sdssshape_psf_shape11"]
        psf_myy = catalog["i_sdssshape_psf_shape22"]
        psf_mxy = catalog["i_sdssshape_psf_shape12"]
    elif "ishape_sdss_psf_ixx" in catalog.dtype.names:
        psf_mxx = catalog["ishape_sdss_psf_ixx"]
        psf_myy = catalog["ishape_sdss_psf_iyy"]
        psf_mxy = catalog["ishape_sdss_psf_ixy"]
    else:
        raise ValueError("Input catalog does not have required column name")

    if return_shear:
        return (
            (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2.0,
            psf_mxy / (psf_mxx + psf_myy),
        )
    else:
        return (
            (psf_mxx - psf_myy) / (psf_mxx + psf_myy),
            2.0 * psf_mxy / (psf_mxx + psf_myy),
        )


def make_hscy3_cat(
    fpath_cats="/pscratch/sd/x/xiangchl/data/catalog/hsc_year3_shape/",
    fpath_primcats="catalog_obs_reGaus_public/",
    fpath_secondary="/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/prelim_hscy3/gfarm.ipmu.jp/~surhud/S19ACatalogs/catalog_tracts/",
    field_names=["GAMA09H", "GAMA15H", "HECTOMAP", "VVDS", "WIDE12H", "XMM"],
    # 20 deg region excluded by HSC analysis because of B modes appearing in x-corr
    # and 4th moment PSF modelling issues
    use_bmode_mask=True,
    # if we need to add photo-zs
    add_photz=True,
    # which photo-z methods to add, if add_photz is True
    photoz_method=["dnnz", "mizuki"],
    check_all_galaxies=False,
):
    # which HSC bands to use
    MAGNITUDE_BANDS = ["g", "r", "i", "z", "y"]

    base_rename_map = {
        "i_ra": "ra",
        "i_dec": "dec",
        "i_hsmshaperegauss_e1": "e_1",
        "i_hsmshaperegauss_e2": "e_2",
        "i_hsmshaperegauss_derived_weight": "weight",
        "i_hsmshaperegauss_derived_rms_e": "e_rms",
        "i_hsmshaperegauss_derived_shear_bias_m": "m_corr",
        "i_hsmshaperegauss_derived_shear_bias_c1": "c_1",
        "i_hsmshaperegauss_derived_shear_bias_c2": "c_2",
        "i_hsmshaperegauss_resolution": "resolution",
        "i_apertureflux_10_mag": "i_aperture_mag",
        "i_cmodel_mag": "i_cm_mag",
        "i_cmodel_magerr": "i_cm_magerr",
        "hsc_y3_zbin": "z_bin",
    }

    mag_rename_map = {
        f"forced_{mag}_cmodel_mag": f"forced_{mag}_cm_mag" for mag in MAGNITUDE_BANDS
    }
    mag_rename_map.update(
        {
            f"forced_{mag}_cmodel_magerr": f"forced_{mag}_cm_magerr"
            for mag in MAGNITUDE_BANDS
        }
    )
    mag_rename_map.update(
        {
            f"forced_{mag}_cmodel_flag": f"forced_{mag}_cm_flag"
            for mag in MAGNITUDE_BANDS
        }
    )

    rename_map = {**base_rename_map, **mag_rename_map}

    # Pre-define column lists
    if add_photz:
        columns_pz = ["object_id"]
        for method in photoz_method:
            pz_prefix = f"{method}_photoz_"
            columns_pz.extend(
                [
                    pz_prefix + "best",
                    pz_prefix + "err95_min",
                    pz_prefix + "err95_max",
                    pz_prefix + "std_best",
                ]
            )

        mag_columns = ["object_id", "i_cmodel_mag", "i_cmodel_magerr", "a_i"]
        for mag in MAGNITUDE_BANDS:
            mag_columns.extend(
                [
                    f"forced_{mag}_cmodel_mag",
                    f"forced_{mag}_cmodel_magerr",
                    f"forced_{mag}_cmodel_flag",
                ]
            )
    field_tables = []

    for field_name in field_names:
        pth = Path(fpath_cats, fpath_primcats, f"{field_name}.fits")

        if not Path(pth).exists():
            raise FileNotFoundError(f"File {pth} does not exist")

        lenscat = Table.read(pth)

        if use_bmode_mask:
            lenscat = lenscat[lenscat["b_mode_mask"]]

        if add_photz:
            secondary_cats = glob(f"{fpath_secondary}{field_name}_tracts/*_pz.fits")
            if not secondary_cats:
                print(f"Warning: No secondary catalogs found for field {field_name}")
                continue

            field_secondary_tables = []
            for secondary_cat in tqdm(secondary_cats, desc=f"Processing {field_name}"):
                mag_cat = secondary_cat.replace("_pz.fits", "_no_m.fits")
                if not (Path(mag_cat).exists() and Path(secondary_cat).exists()):
                    raise FileNotFoundError(
                        f"Required files {mag_cat} or {secondary_cat} do not exist"
                    )

                try:
                    with fio.FITS(mag_cat) as f:
                        hudl_mag = Table(f[1].read(columns=mag_columns))

                    with fio.FITS(secondary_cat) as f:
                        hdul_nofz = Table(f[1].read(columns=columns_pz))

                    # Join magnitude and photoz data
                    joint_pz_mag = join(
                        hudl_mag, hdul_nofz, keys="object_id", join_type="inner"
                    )
                    field_secondary_tables.append(joint_pz_mag)

                except Exception as e:
                    print(f"Error processing {secondary_cat}: {e}")
                    continue

            if field_secondary_tables:
                field_secondary_combined = vstack(field_secondary_tables)

                joint_tab = join(
                    lenscat,
                    field_secondary_combined,
                    keys="object_id",
                    join_type="inner",
                )
                field_tables.append(joint_tab)

                if check_all_galaxies:
                    if not set(lenscat["object_id"]).issubset(
                        set(joint_tab["object_id"])
                    ):
                        print(
                            f"Warning: Not all galaxies from {field_name} found in joined catalog"
                        )
            else:
                print(
                    f"Warning: No valid secondary catalogs processed for field {field_name}"
                )
        else:
            field_tables.append(lenscat)

    if not field_tables:
        raise ValueError("No valid field tables were processed")

    final_lenscat = vstack(field_tables)
    for old, new in rename_map.items():
        if old in final_lenscat.colnames:
            final_lenscat.rename_column(old, new)

    e1_psf, e2_psf = get_psf_ellip(final_lenscat, return_shear=False)
    final_lenscat["e1_psf"] = e1_psf
    final_lenscat["e2_psf"] = e2_psf

    base_columns = [
        "object_id",
        "ra",
        "dec",
        "e_1",
        "e_2",
        "z_bin",
        "weight",
        "m_corr",
        "c_1",
        "c_2",
        "resolution",
        "e_rms",
        "e1_psf",
        "e2_psf",
        "i_aperture_mag",
        "i_cm_mag",
        "i_cm_magerr",
        "a_i",
    ]

    mag_columns_final = []
    for mag in MAGNITUDE_BANDS:
        mag_columns_final.extend(
            [f"forced_{mag}_cm_mag", f"forced_{mag}_cm_magerr", f"forced_{mag}_cm_flag"]
        )

    all_columns = base_columns + mag_columns_final

    if add_photz:
        photoz_columns = []
        for method in photoz_method:
            photoz_columns.extend(
                [
                    f"{method}_photoz_best",
                    f"{method}_photoz_err95_min",
                    f"{method}_photoz_err95_max",
                    f"{method}_photoz_std_best",
                ]
            )
        all_columns.extend(photoz_columns)

    available_columns = [col for col in all_columns if col in final_lenscat.colnames]
    final_lenscat.keep_columns(available_columns)

    # apply sign corrections
    if "e_2" in final_lenscat.colnames:
        final_lenscat["e_2"] = -final_lenscat["e_2"]
    if "c_2" in final_lenscat.colnames:
        final_lenscat["c_2"] = -final_lenscat["c_2"]
    if "e2_psf" in final_lenscat.colnames:
        final_lenscat["e2_psf"] = -final_lenscat["e2_psf"]

    return final_lenscat


if __name__ == "__main__":
    try:
        print("Starting HSC Y3 catalog creation...")
        final_lenscat = make_hscy3_cat()

        # change this by your output path. will overwrite by default !
        output_path = "/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat.fits"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing catalog with {len(final_lenscat)} objects to {output_path}")
        final_lenscat.write(output_path, overwrite=True)
        print("Catalog creation completed successfully!")

    except Exception as e:
        raise RuntimeError(f"An error occurred during catalog creation: {e}")
