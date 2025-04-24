import fitsio as fio 
import numpy as np 
import pandas as pd 
import healpy as hp
import cmocean.cm as cmo
import matplotlib.pyplot as plt
from numpy.lib import recfunctions as rfn

from astropy.table import Table, vstack
from pathlib import Path

HSC_CATALOG = Path(
    '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat.fits'
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
        self.TRUTH = self.ROOT_BUZZARD / 'truth'
        self.DESI_TGTS = self.ROOT_BUZZARD / 'desi_targets_v1.2'
        self.DESI_TGTS_RANDOMS = self.ROOT_BUZZARD / 'desi_targets'
        self.MAGS = self.ROOT_BUZZARD / 'surveymags'

        self.prefix = f'Chinchilla-{self.buzzard_index}_cam_rs_scat_shift_lensed.'
        self.suffix = '.fits'

        self.files_pix = list(
            self.TRUTH.glob(
                f'{self.prefix}*{self.suffix}'
                )
            )
        self.desi_tgt_files_pix = list(
            self.DESI_TGTS.glob(
                f'{self.prefix}*{self.suffix}'
                )
            )
        self.pix_nums = sorted([int(pixfile.name.split('.')[1]) for pixfile in self.files_pix])
            
    def fetch_truth(self, pix):
        return Path(
            self.TRUTH, 
            f'{self.prefix}{pix}{self.suffix}'
            )
    
    def fetch_surveymag(self, pix):
        return Path(
            self.MAGS,
            # does not work if buzzard_index is 0 so else
            f'Chinchilla-{self.buzzard_index}-aux.{pix}{self.suffix}' 
            if self.buzzard_index == 4 else
            f'surveymags-aux.{pix}{self.suffix}'
        )

    def fetch_desi_target(self, pix):
        return Path(
            self.DESI_TGTS,
            f'{self.prefix}{pix}{self.suffix}'
        )
    
    def transform_pixels(self, sim_pixels, ra_flip=False, dec_flip=False):
        theta, phi = hp.pix2ang(self.nside_buzzard, sim_pixels, nest=True)
        if ra_flip:
            phi = (phi + np.pi) % (2 * np.pi)
        if dec_flip:
            theta = np.pi - theta
        return hp.ang2pix(self.nside_buzzard, theta, phi, nest=True)
    
    def fetch_desi_randoms(self, target):
        assert target in ['bgs', 'lrg', 'elg'], "Target must be one of ['BGS', 'LRG', 'ELG']"
        return Path(
            self.DESI_TGTS_RANDOMS,
            f'{target}_rand{self.suffix}'
        )
    

def sample_on_hsc(active_hsc_pixels, nside_hsc, ra, dec):
    '''
    Given ra, dec coordinates, give a mask on ra and dec 
    where these coordinates are in the active hsc pixels
    (with nside provided). nest scheme is true.
    '''

    hsc_pixels = hp.ang2pix(nside_hsc, np.radians(90-dec), np.radians(ra), nest=True)
    return np.isin(hsc_pixels, active_hsc_pixels)