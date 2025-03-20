'''
This script modifies the original `make_hscy3.py` to make a HSC Y3 catalog
with dnnz photo-zs. 
'''
from astropy.table import Table,join,vstack
from glob import glob
from tqdm import tqdm

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
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2.0, psf_mxy / (
            psf_mxx + psf_myy
        )
    else:
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy), 2.0 * psf_mxy / (
            psf_mxx + psf_myy
        )

def make_hscy3_cat(
        fpath_cats = "/pscratch/sd/x/xiangchl/data/catalog/hsc_year3_shape/",
        fpath_primcats = "catalog_obs_reGaus_public/",
        fpath_secondary = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/prelim_hscy3/gfarm.ipmu.jp/~surhud/S19ACatalogs/catalog_tracts/",
        field_names = ["GAMA09H","GAMA15H","HECTOMAP","VVDS","WIDE12H","XMM"],
        use_bmode_mask = True,
        add_photz = True,
        photoz_method = "dnnz",
        check_all_galaxies = False,
    ):
    final_lenscat = Table()
    for field_name in field_names:
        pth = fpath_cats + fpath_primcats + f'{field_name}.fits'
        print(pth)
        lenscat = Table.read(pth)
        if use_bmode_mask:
            lenscat = lenscat[lenscat["b_mode_mask"]]
        if add_photz:
            secondary_cats = glob(fpath_secondary+f"{field_name}_tracts/*_pz.fits")
            for secondary_cat in tqdm(secondary_cats,
                                      desc=f"Processing {field_name}",
                                      total=len(secondary_cats)):
                hdul_nofz = Table.read(secondary_cat,hdu=1)
                hdul_nofz.keep_columns([
                    'object_id',
                    "{}_photoz_best".format(photoz_method),
                    "{}_photoz_err68_min".format(photoz_method),
                    "{}_photoz_err68_max".format(photoz_method),
                    "{}_photoz_err95_min".format(photoz_method),
                    "{}_photoz_err95_max".format(photoz_method),
                    "{}_photoz_risk_best".format(photoz_method),
                    "{}_photoz_std_best".format(photoz_method)
                ])
                
                joint_tab = join(lenscat,hdul_nofz,keys='object_id',join_type='inner')
                final_lenscat = vstack([final_lenscat,joint_tab])
            if check_all_galaxies:
                assert set(lenscat['object_id']).issubset(set(final_lenscat['object_id']))
        else:
            final_lenscat = vstack([final_lenscat,lenscat])
    lens_bin_mask = final_lenscat['hsc_y3_zbin']>0
    final_lenscat = final_lenscat[lens_bin_mask]

    final_lenscat.rename_column('i_ra','ra')
    final_lenscat.rename_column('i_dec','dec')
    final_lenscat.rename_column('i_hsmshaperegauss_e1','e_1')
    final_lenscat.rename_column('i_hsmshaperegauss_e2','e_2')
    final_lenscat.rename_column('i_hsmshaperegauss_derived_weight','weight')
    final_lenscat.rename_column('i_hsmshaperegauss_derived_rms_e','e_rms')
    final_lenscat.rename_column('i_hsmshaperegauss_derived_shear_bias_m','m_corr')
    final_lenscat.rename_column('i_hsmshaperegauss_derived_shear_bias_c1','c_1')
    final_lenscat.rename_column('i_hsmshaperegauss_derived_shear_bias_c2','c_2')
    final_lenscat.rename_column('i_hsmshaperegauss_resolution','resolution')
    final_lenscat.rename_column('i_apertureflux_10_mag','aperture_mag')
    final_lenscat.rename_column('hsc_y3_zbin','z_bin')
    e1_psf,e2_psf = get_psf_ellip(final_lenscat,return_shear=False)
    final_lenscat['e1_psf'] = e1_psf
    final_lenscat['e2_psf'] = e2_psf
    all_columns = ['object_id', 'ra','dec','e_1','e_2','z_bin','weight','m_corr','c_1','c_2','resolution','e_rms','e1_psf','e2_psf','aperture_mag']
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
    final_lenscat.write("/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat.fits",overwrite=True)
