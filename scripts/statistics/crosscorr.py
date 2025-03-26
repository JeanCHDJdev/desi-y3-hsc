import numpy as np 
import time 

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import fitsio as fio
import pandas as pd
import warnings

from mocpy import MOC
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, match_coordinates_sky
from astropy.io import fits
from astropy.table import Table
from argparse import ArgumentParser

from desitarget.targetmask import desi_mask
from pycorr import (
    TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, utils, setup_logging
)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '-o',
        '--output_dir', 
        type=str, 
        default='out/',
        help='Path to the output file (storing the cross-correlation)'
        'Default is out/'
        )
    parser.add_argument(
        '-w',
        '--nproc', 
        type=int, 
        help='Number of threads to use for cross-correlation'
        'Default is 1'
        )
    parser.add_argument(
        '-r',
        '--randoms', 
        type=int,
        default=1,
        help='Number of randoms to use. Defaults to 1'
    )
    parser.add_argument(
        '-l',
        '--log', 
        type=str,
        default='log.txt',
        help='Log file to store run settings'
    )
    
    return parser.parse_args()

class CrossCorrelation():
    def fetch_desi_files(self, randoms=False):

        root = Path(
            '/global/cfs/projectdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v1.1/nonKP'
            )
        path = f'{self.tgt}{f"_{self.cap}_*" if self.cap else ("_[0-9]*_" if randoms else "_")}clustering{".ran" if randoms else ".dat"}.fits'
        return list(root.glob(path))

    def __init__(
            self, 
            tgt, 
            mocfile, 
            output_dir, 
            bin_distances,
            nproc=None, 
            nrandoms=4, 
            cap=None
            ):

        self.moc = Path(mocfile)
        assert self.moc.exists(), f'{self.moc} does not exist'

        avb_cap = ['NGC', 'SGC']
        if cap:
            assert cap in avb_cap, f'{cap} not in {avb_cap}'
        self.cap = cap

        avb_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
        assert tgt in avb_tgt, f'{tgt} not in {avb_tgt}'
        self.tgt = tgt

        self.nrandoms = nrandoms

        if nproc is None:
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        self.bin_distances = bin_distances

        ## Filesystem setup
        # FESI
        fs = {}
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            fs['outdir'] = Path(self.output_dir)
        random_files = np.random.choice(
            self.fetch_desi_files(randoms=True), size=self.nrandoms, replace=False
            )
        fs['randoms1'] = random_files
        fs['catalog1'] = self.fetch_desi_files(randoms=False)
        fs['moc'] = self.moc

        ## todo : initialize randoms for HSC
        fs['catalog2'] = Path('/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cathscy3_cat.fits')
        fs['randoms2'] = ''

        self.fs = fs 

        print(f'Filesystem : {fs}')

    def run(self, bin1, bin2):
        moc = MOC.from_fits(filename=self.fs['moc'])

        tpcf = TwoPointCorrelationFunction(
            edges=self.bin_distances,
            data_positions1=read_coords(d1, 'RA', 'DEC'),
            data_positions2=read_coords(d2, 'RA', 'Dec'),
            randoms_positions1=read_coords(r1, 'RA', 'DEC'),
            randoms_positions2=read_coords(r2, 'RA', 'Dec'),
            data_weights1=read_weights(dw1, 'WEIGHT'),
            data_weights2=read_weights(dw2, 'weight'),
            randoms_weights1=read_weights(rw1, 'WEIGHT'),
            randoms_weights1=read_weights(rw2, 'weight'),
            n_threads=self.nproc,
            mode='theta',
            position_type='rd', # 'rd' for RA/Dec
            engine='corrfunc',
            estimator='landyszalay',
        )
        outfile = Path(
            self.output_dir, 
            f'{self.tgt}_{self.cap}_b1x{bin1}_b2x{bin2}.fits'
            )
        tpcf.write(outfile, overwrite=True)


def collate_randoms(random_files, ra_col, dec_col, w_col):
    assert len(random_files) > 0, f"No random files "
    randoms = []
    size = 0
    for f in random_files:
        with fio.FITS(f) as rand:
            data = rand[1].read(columns=[ra_col, dec_col, w_col])
            size += len(data)
            randoms.append(data)  

    print(f"Collated {size} randoms (from {len(random_files)} files)")
    dtype = [(ra_col, 'f8'), (dec_col, 'f8'), (w_col, 'f8')]
    return np.concatenate(randoms).astype(dtype)

def sample_file_on_mask(file, ra_col, dec_col, weight_col, mask=None):
    with fio.FITS(file) as f:
        if mask is not None:
            rows = np.flatnonzero(mask)
            data = f[1].read(columns=[ra_col, dec_col, weight_col], rows=rows)
        else:
            data = f[1].read(columns=[ra_col, dec_col, weight_col])
    
    return data[ra_col], data[dec_col], data[weight_col]

def read_coords(file, ra_col, dec_col):
    with fio.FITS(file) as f:
        data = f[1].read(columns=[ra_col, dec_col])
    return SkyCoord(data[ra_col]*u.deg, data[dec_col]*u.deg, frame='icrs')

def read_weights(file, weight_col):
    with fio.FITS(file) as f:
        weight = f[1].read(column=weight_col)
    return weight

def main():
    args = parse_args()

    output_dir = args.output_dir
    nr = args.randoms
    nproc = args.nproc
    log = args.log

    setup_logging()

    bin_distances = np.linspace(0., 200., 51)

    bins_bgs = np.arange(0, 0.7, 8) # 0 < z < 0.6
    bins_lrg = np.arange(0.3, 1.1, 9) # 0.4 < z < 1
    bins_elg = np.arange(0.6, 1.7, 12) # 0.6 < z < 1.6
    bins_qso = np.arange(0.9, 2.2, 0.1) # 0.9 < z < 2.1
    bins_hsc = np.linspace(0.3, 1.2, 4) 
    bins_redshift = {
        'BGS_ANY': bins_bgs,
        'LRG': bins_lrg,
        'ELGnotqso': bins_elg,
        'QSO': bins_qso,
        'HSC': bins_hsc,
    }
    with open(log, 'a') as f:
        f.write(f'Number of randoms: {nr}\n')
        f.write(f'Number of threads: {nproc}\n')
        f.write(f'Output directory: {output_dir}\n')
        f.write('Bins:\n')
        f.write(f'{bins_redshift}\n')
        f.write(f'Fiducial bin distances: {bin_distances}\n')
        f.write(f'Log file: {log}\n')
        f.write('\n')

    # for now, we will only consider the following targets, could do more later
    avb_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    assert tgt in avb_tgt, f'{tgt} not in {avb_tgt}'
    
    setup_logging()

    moc_list = Path('/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/mocs/').glob('hsc_moc*.fits')

    ra_hsc_col = 'RA'
    dec_hsc_col = 'Dec'
    w_hsc_col = 'weight'
    ra_desi_col = 'RA'
    dec_desi_col = 'DEC'
    w_desi_col = 'WEIGHT'

    for mocf in moc_list:
        moc = MOC.from_fits(filename=mocf)
        coords_hsc = read_coords(mocf, ra_hsc_col, dec_hsc_col)
        mask = moc.contains_skycoord(coords_hsc)

        print(f'{mocf} contains {mask.sum()} HSC objects')
        if mask.sum() == 0:
            raise ValueError(f'{mocf} contains no HSC objects')
        
        for tgt in avb_tgt:
            cc = CrossCorrelation(
                tgt, 
                mocf, 
                output_dir, 
                bin1, 
                bin2, 
                nproc=nproc, 
                nrandoms=nr, 
                cap=None
            )
            randoms1 = cc.collate_randoms(cc.fs['randoms1'])

        bin1 = bins_redshift[tgt]
        bin2 = bins_redshift['HSC']
        print(f'Running for {tgt}, {bin1}, {bin2}')
        cc = CrossCorrelation(
            tgt, 
            mocf, 
            output_dir, 
            nproc=nproc, 
            nrandoms=nr, 
            cap=None
            )
        
        cc.run()


    desi_tgt = np.array(desi[1]['DESI_TARGET'].read())

    is_bgs  = (desi_tgt & desi_mask.BGS_ANY != 0)   #- instead of 2**60
    is_lrg  = (desi_tgt & desi_mask.LRG != 0)
    is_elg  = (desi_tgt & desi_mask.ELG != 0)
    is_qso  = (desi_tgt & desi_mask.QSO != 0)

    tgt = 'LRG'
    print('Set up bins and targets')


if __name__ == '__main__':
    main()