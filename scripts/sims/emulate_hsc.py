import fitsio as fio 
import numpy as np 
import pandas as pd 
import healpy as hp
import cmocean.cm as cmo
import matplotlib.pyplot as plt
from numpy.lib import recfunctions as rfn

from astropy.table import Table
from pathlib import Path

HSC_CATALOG = Path(
    '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat.fits'
)
buzzard_index = 0
ROOT_BUZZARD = Path(
    f'/global/cfs/projectdirs/desi/mocks/buzzard/buzzard_v2.0/buzzard-{buzzard_index}/addgalspostprocess'
    )
TRUTH = ROOT_BUZZARD / 'truth'
DESI_TGTS = ROOT_BUZZARD / 'desi_targets_v1.2'
MAGS = ROOT_BUZZARD / 'surveymags'

OUTPUT_SIM = Path(
    '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/sims/hsc_sim.fits'
)

class BuzzardCatalog():
    def __init__(self, buzzard_index):
        self.nside_buzzard = 8
        self.buzzard_index = buzzard_index
        self.ROOT_ALL = Path(
            '/global/cfs/projectdirs/desi/mocks/buzzard/buzzard_v2.0'
            )
        self.ROOT_BUZZARD = Path(
            self.ROOT_ALL,
            f'buzzard-{self.buzzard_index}/addgalspostprocess'
            )
        self.TRUTH = ROOT_BUZZARD / 'truth'
        self.DESI_TGTS = ROOT_BUZZARD / 'desi_targets_v1.2'
        self.MAGS = ROOT_BUZZARD / 'surveymags'

        self.files_pix = self.TRUTH.glob('Chinchilla-11_cam_rs_scat_shift_lensed.*.fits')
        self.pix_nums = sorted([int(pix.name.split('.')[1]) for pix in self.files_pix])
            
    def fetch_truth(self, pix):
        return Path(
            self.TRUTH, 
            f'Chinchilla-{self.buzzard_index}_cam_rs_scat_shift_lensed.{pix}.fits'
            )
    
    def fetch_surveymag(self, pix):
        return Path(
            self.MAGS,
            f'surveymags-aux.{pix}.fits'
        )

    def fetch_desi_target(self, pix):
        return Path(
            self.DESI_TGTS,
            f'Chinchilla-0_cam_rs_scat_shift_lensed.{pix}.fits'
        )
    
    def transform_pixels(self, sim_pixels, ra_flip=False, dec_flip=False):
        theta, phi = hp.pix2ang(self.nside_buzzard, sim_pixels, nest=True)
        if ra_flip:
            phi = (phi + np.pi) % (2 * np.pi)
        if dec_flip:
            theta = np.pi - theta
        return hp.ang2pix(self.nside_buzzard, theta, phi, nest=True)
    
def main():

    bc = BuzzardCatalog(buzzard_index=0)
    nside_buzzard = bc.nside_buzzard
    nside_hsc = 2048
    filters = np.loadtxt('filters.txt', dtype=str)
    mag_index = list(filters).index('DECAM_i')
    del filters

    ## Get HSC coordinates
    hsc_tbl_all = fio.FITS(HSC_CATALOG)
    # We move HSC coordinates to a location on the sky covered by Buzzard mocks (NW corner)
    # We do field by field and join afterwards the catalogs. Once the field is moved, 
    # sample on where buzzard actually is by limiting ra, dec coordinates
    offset_ra = 80
    offset_dec = 30
    ra_all = (hsc_tbl_all[1]['RA'].read() + offset_ra) % 360
    dec_all = hsc_tbl_all[1]['DEC'].read() + offset_dec
    assert np.sum(dec_all > 90) == 0
    # limiting ourselves to the NW corner
    mask = (0 < ra_all) & (ra_all < 180) & (0 < dec_all)
    hsc_tbl = hsc_tbl_all[1][mask]
    ra_hsc = ra_all[mask]
    dec_hsc = dec_all[mask]

    npix_hsc = hp.nside2npix(nside_hsc)
    theta = np.radians(90.0 - dec_hsc) 
    phi = np.radians(ra_hsc)
    hsc_pix_indices = hp.ang2pix(nside_hsc, theta, phi, nest=True)
    hsc_hp_map = np.bincount(hsc_pix_indices, minlength=npix_hsc)

    tiles_nonnull_hsc = np.flatnonzero(hsc_hp_map > 0)
    theta, phi = hp.pix2ang(nside_hsc, tiles_nonnull_hsc, nest=True)
    tiles_buzzard = hp.ang2pix(nside_buzzard, theta, phi, nest=True)

    tiles_buzzard_to_hsc = {}
    for t_buzzard, t_hsc in zip(tiles_buzzard, tiles_nonnull_hsc):
        try:
            tiles_buzzard_to_hsc[t_buzzard].append(t_hsc)
        except KeyError:
            tiles_buzzard_to_hsc[t_buzzard] = [t_hsc]