'''
This script modifies the original `make_hscy3.py` to make a HSC Y3 catalog
with dnnz photo-zs. 
'''
from astropy.table import Table, join, vstack, hstack
from glob import glob
from tqdm import tqdm
import fitsio as fio

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
        raise ValueError("Input catalog does not have required coulmn name")

    if return_shear:
        return (
            (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2.0, 
            psf_mxy / (psf_mxx + psf_myy)
        )
    else:
        return (
            (psf_mxx - psf_myy) / (psf_mxx + psf_myy), 
            2.0 * psf_mxy / (psf_mxx + psf_myy)
            )
    

def make_hscy3_cat(
        fpath_cats = "/pscratch/sd/x/xiangchl/data/catalog/hsc_year3_shape/",
        fpath_primcats = "catalog_obs_reGaus_public/",
        fpath_secondary = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/prelim_hscy3/gfarm.ipmu.jp/~surhud/S19ACatalogs/catalog_tracts/",
        field_names = ["GAMA09H", "GAMA15H", "HECTOMAP", "VVDS", "WIDE12H", "XMM"],
        use_bmode_mask = True, #True
        add_photz = True,
        photoz_method = "dnnz",
        check_all_galaxies = False,
    ):
    final_lenscat = Table()
    for field_name in field_names:
        pth = fpath_cats + fpath_primcats + f'{field_name}.fits'
        lenscat = Table.read(pth)

        if use_bmode_mask:
            lenscat = lenscat[lenscat["b_mode_mask"]]
        if add_photz:
            secondary_cats = glob(f"{fpath_secondary}{field_name}_tracts/*_pz.fits")

            pz_prefix = f"{photoz_method}_photoz_"
            columns_pz = [
                'object_id',
                pz_prefix + "best",
                pz_prefix + "err68_min",
                pz_prefix + "err68_max",
                pz_prefix + "err95_min",
                pz_prefix + "err95_max",
                pz_prefix + "risk_best",
                pz_prefix + "std_best",
            ]
            mag_columns = [
                'object_id',
                'i_cmodel_mag',
                'i_cmodel_magerr', 
                ]
            for mag in ['g', 'i', 'r', 'z', 'y']:
                mag_columns.extend([
                    f'forced_{mag}_cmodel_mag', 
                    f'forced_{mag}_cmodel_magerr', 
                    f'forced_{mag}_cmodel_flag'
                ])
            
            for secondary_cat in tqdm(secondary_cats, desc=f"{field_name}"):
                mag_cat = secondary_cat.replace('_pz.fits', '_no_m.fits')

                with fio.FITS(mag_cat) as f:
                    hudl_mag = Table(f[1].read(columns=mag_columns))

                with fio.FITS(secondary_cat) as f:
                    hdul_nofz = Table(f[1].read(columns=columns_pz))
                
                joint_pz_mag = join(hudl_mag, hdul_nofz, keys='object_id', join_type='inner')
                joint_tab = join(lenscat, joint_pz_mag, keys='object_id', join_type='inner')

                final_lenscat = vstack([final_lenscat, joint_tab])

            if check_all_galaxies:
                assert set(lenscat['object_id']).issubset(set(final_lenscat['object_id']))
        else:
            final_lenscat = vstack([final_lenscat,lenscat])
    # can do this later...
    #lens_bin_mask = final_lenscat['hsc_y3_zbin']>0
    #final_lenscat = final_lenscat[lens_bin_mask]
    rename_map = {
        'i_ra': 'ra',
        'i_dec': 'dec',
        'i_hsmshaperegauss_e1': 'e_1',
        'i_hsmshaperegauss_e2': 'e_2',
        'i_hsmshaperegauss_derived_weight': 'weight',
        'i_hsmshaperegauss_derived_rms_e': 'e_rms',
        'i_hsmshaperegauss_derived_shear_bias_m': 'm_corr',
        'i_hsmshaperegauss_derived_shear_bias_c1': 'c_1',
        'i_hsmshaperegauss_derived_shear_bias_c2': 'c_2',
        'i_hsmshaperegauss_resolution': 'resolution',
        'i_apertureflux_10_mag': 'i_aperture_mag',
        'i_cmodel_mag': 'i_cm_mag',
        'i_cmodel_magerr': 'i_cm_magerr',
        'hsc_y3_zbin': 'z_bin',
    }
    rename_map.update(
        {
            f'forced_{mag}_cmodel_mag': f'forced_{mag}_cm_mag',
            f'forced_{mag}_cmodel_magerr': f'forced_{mag}_cm_magerr',
            f'forced_{mag}_cmodel_flag': f'forced_{mag}_cm_flag',
        } for mag in ['g', 'r', 'i', 'z', 'y']
    )
    
    for old, new in rename_map.items():
        final_lenscat.rename_column(old, new)

    e1_psf,e2_psf = get_psf_ellip(final_lenscat,return_shear=False)
    final_lenscat['e1_psf'] = e1_psf
    final_lenscat['e2_psf'] = e2_psf
    all_columns = [
        'object_id', 
        'ra',
        'dec',
        'e_1',
        'e_2',
        'z_bin',
        'weight',
        'm_corr',
        'c_1',
        'c_2',
        'resolution',
        'e_rms',
        'e1_psf',
        'e2_psf',
        'i_aperture_mag',
        'i_cm_mag',
        'i_cm_magerr',
        ]
    for mag in ['g', 'r', 'i', 'z', 'y']:
        all_columns += [
            f'forced_{mag}_cm_mag',
            f'forced_{mag}_cm_magerr',
            f'forced_{mag}_cm_flag'
        ]
    if add_photz:
        all_columns += [
            f"{photoz_method}_photoz_best",
            f"{photoz_method}_photoz_err68_min",
            f"{photoz_method}_photoz_err68_max",
            f"{photoz_method}_photoz_err95_min",
            f"{photoz_method}_photoz_err95_max",
            f"{photoz_method}_photoz_risk_best",
            f"{photoz_method}_photoz_std_best"
        ]
    final_lenscat.keep_columns(all_columns)

    final_lenscat['e_2'] = -final_lenscat['e_2']
    final_lenscat['c_2'] = -final_lenscat['c_2']
    final_lenscat['e2_psf'] = -final_lenscat['e2_psf']

    return final_lenscat

if __name__=="__main__":
    final_lenscat = make_hscy3_cat()
    final_lenscat.write(
        "/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat_with_grizy.fits",
        overwrite=True
        )
