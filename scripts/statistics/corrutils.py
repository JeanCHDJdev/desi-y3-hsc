'''
This script holds wrapper functions and utilities for 
calculating and plotting correlation matrices.
'''
import time 
import os
import fitsio as fio
import numpy as np
import logging
import psutil
import multiprocessing as mp

from mocpy import MOC
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord

from scripts.statistics import cosmtools as ct
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler

class CorrelationMeta:

    # Attributes that each cross corr code needs to know about
    ra_hsc_col = 'RA'
    dec_hsc_col = 'Dec'
    ra_hsc_randoms_col = 'ra'
    dec_hsc_randoms_col = 'dec'
    w_hsc_col = 'weight'
    z_hsc_col = 'dnnz_photoz_best'

    ra_desi_col = 'RA'
    dec_desi_col = 'DEC'
    w_desi_col = 'WEIGHT'
    z_desi_col = 'Z'

    ## MOC list 
    moc_list = [
            Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/mocs/', 
                f'hsc_moc{i+1}.fits'
            )
            for i in range(0, 4)
        ]

    ## Defining fiducial bins here
    bin_distances = np.linspace(0.01, 2, 41) #np.logspace(np.log10(0.001), np.log10(3), 71)

    bins_bgs = np.arange(0, 0.6, 0.1) # 0 < z < 0.6
    bins_lrg = np.arange(0.4, 1.2, 0.1) # 0.4 < z < 1
    bins_elg = np.arange(0.8, 1.7, 0.1) # 0.6 < z < 1.6 => 0.8 < z < 1.6 in redshift distribution
    bins_qso = np.arange(0.8, 3.4, 0.1) # 0.9 < z < 2.1

    bins_hsc = np.arange(0.3, 1.8, 0.3) # 0.3 < z <= 1.5 

    @staticmethod
    def save_bins(root):
        """
        Save the bins to a file
        """
        bin_dir = Path(root, 'bins')
        if not bin_dir.exists():
            bin_dir.mkdir(parents=True)
            np.savez(
                Path(bin_dir, 'bins_desi.npz'),
                bins_bgs=CorrelationMeta.bins_bgs,
                bins_lrg=CorrelationMeta.bins_lrg,
                bins_elg=CorrelationMeta.bins_elg,
                bins_qso=CorrelationMeta.bins_qso
            )

            np.savez(
                Path(bin_dir, 'bins_hsc.npz'),
                bins_hsc=CorrelationMeta.bins_hsc
            )

    def __init__(self, logger, tgt, moc, output_dir, sims, nproc=None):
        assert logger is not None, 'Logger not provided'
        self.logger = logger

        avb_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
        assert tgt in avb_tgt, f'{tgt} not in {avb_tgt}'
        self.tgt = tgt

        if nproc is None: 
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        ## Filesystem setup
        fs = {}
        self.output_dir = Path(output_dir, tgt)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        fs['outdir'] = Path(self.output_dir)

        # Grabbing the catalogs on initialisation
        fs['catalog1'] = fetch_desi_files(tgt, randoms=False, sims=sims)
        fs['randoms1'] = fetch_desi_files(tgt, randoms=True, sims=sims)
        fs['catalog2'] = fetch_hsc_files(randoms=False, sims=sims)
        fs['randoms2'] = fetch_hsc_files(randoms=True, sims=sims)
        
        ## Loading the MOC footprint
        self.moc = moc

        # Keep the filesystem
        self.fs = fs 


class CrossCorrelation(CorrelationMeta):
    def __init__(
            self, 
            tgt : str, 
            moc : MOC, 
            output_dir :str | Path, 
            bin_distances : np.ndarray,
            bin_redshift1 : np.ndarray,
            bin_redshift2 : np.ndarray,
            nproc:int=None, 
            sample_rate_desi:int=1, 
            sample_rate_hsc:int=1,
            logger:logging.Logger=None
            ):

        assert logger is not None, 'Logger not provided'
        self.logger = logger

        avb_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
        assert tgt in avb_tgt, f'{tgt} not in {avb_tgt}'
        self.tgt = tgt

        if nproc is None: 
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        self.bin_distances = bin_distances

        ## Filesystem setup
        fs = {}
        self.output_dir = Path(output_dir, tgt)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        fs['outdir'] = Path(self.output_dir)

        # Grabbing the catalogs on initialisation
        fs['catalog1'] = fetch_desi_files(tgt, randoms=False)
        fs['randoms1'] = fetch_desi_files(tgt, randoms=True)
        fs['catalog2'] = fetch_hsc_files(randoms=False)
        fs['randoms2'] = fetch_hsc_files(randoms=True)
        
        ## Loading the MOC footprint
        self.moc = moc

        # Keep the filesystem
        self.fs = fs 

        # Sample rate for randoms
        self.sample_rate_desi = sample_rate_desi
        self.sample_rate_hsc = sample_rate_hsc

        trd = time.time()
        logger.info(f'Collating randoms ...')
        self.randoms1 = sample_randoms_on_moc(
            list(self.fs['randoms1'][:5]),
            ra_col=self.ra_desi_col,
            dec_col=self.dec_desi_col,
            w_col=self.w_desi_col,
            z_col=self.z_desi_col,
            moc=self.moc,
            )
        self.randoms1 = self.randoms1[::self.sample_rate_desi]
        
        logger.info(f'Collated DESI randoms in {time.time()-trd:.2f} seconds')
        trh = time.time()
        self.randoms2 = sample_randoms_on_moc(
            list(self.fs['randoms2']), 
            ra_col='ra',
            dec_col='dec',
            w_col=None,
            z_col=None,
            moc=self.moc,
            )
        self.randoms2 = self.randoms2[::self.sample_rate_hsc]
        logger.info(f'Collated HSC randoms in {time.time()-trh:.2f} seconds')

        tid = time.time()
        self.data1 = sample_file_on_moc(
            self.fs['catalog1'], 
            ra_col=self.ra_desi_col, 
            dec_col=self.dec_desi_col, 
            weight_col=self.w_desi_col, 
            z_col=self.z_desi_col,
            moc=self.moc
            )
        self.logger.info(f'Read DESI data in {time.time()-tid:.2f} seconds')
        tih = time.time()
        self.data2 = sample_file_on_moc(
            file=self.fs['catalog2'], 
            ra_col=self.ra_hsc_col, 
            dec_col=self.dec_hsc_col, 
            weight_col=self.w_hsc_col,
            z_col=self.z_hsc_col, 
            moc=self.moc
            )
        self.logger.info(f'Read HSC data in {time.time()-tih:.2f} seconds')

        # Setup redshift masks
        self.zmask_data1 = np.digitize(self.data1[self.z_desi_col], bin_redshift1, right=True)
        self.zmask_data2 = np.digitize(self.data2[self.z_hsc_col], bin_redshift2, right=True)
        self.zmask_randoms1 = np.digitize(self.randoms1[self.z_desi_col], bin_redshift1, right=True)

    def run(self, bin_index1, bin_index2, moc_index):

        outfile = Path(
            self.output_dir, 
            f'{self.tgt}__b1x{bin_index1}_b2x{bin_index2}_moc{moc_index}.npy'
            )
        if outfile.exists():
            self.logger.info(f'File {outfile} already exists, skipping')
            return
        
        # Setup redshift masking
        z_mask_d1 = (self.zmask_data1 == bin_index1)
        z_mask_d2 = (self.zmask_data2 == bin_index2)
        # DESI randoms need to be redshift masked 
        z_mask_r1 = (self.zmask_randoms1 == bin_index1)

        self.logger.info(f'N data1: {np.sum(z_mask_d1)}' + f', N randoms1: {np.sum(z_mask_r1)}')
        self.logger.info(f'N data2: {np.sum(z_mask_d2)}' + f', N randoms2: {len(self.randoms2)}')

        tpcf = TwoPointCorrelationFunction(
            edges=self.bin_distances,
            data_positions1=[
                self.data1[self.ra_desi_col][z_mask_d1], 
                self.data1[self.dec_desi_col][z_mask_d1]
                ],
            data_positions2=[
                self.data2[self.ra_hsc_col][z_mask_d2], 
                self.data2[self.dec_hsc_col][z_mask_d2]
                ],
            randoms_positions1=[
                self.randoms1[self.ra_desi_col][z_mask_r1],
                self.randoms1[self.dec_desi_col][z_mask_r1]
                ],
            # HSC does not need redshift masking on randoms
            randoms_positions2=[
                self.randoms2[self.ra_hsc_randoms_col],
                self.randoms2[self.ra_hsc_randoms_col]
                ],
            data_weights1=self.data1[self.w_desi_col][z_mask_d1],
            data_weights2=self.data2[self.w_hsc_col][z_mask_d2],
            randoms_weights1=self.randoms1[self.w_desi_col][z_mask_r1],
            randoms_weights2=None,
            nthreads=self.nproc,
            mode='theta',
            position_type='rd', # 'rd' for RA/Dec
            engine='corrfunc',
            estimator='landyszalay',
        )
        tpcf.save(outfile)

class JackknifeCrossCorrelation(CorrelationMeta):
    def __init__(
            self, 
            tgt : str, 
            moc : MOC, 
            output_dir :str | Path, 
            bin_distances : np.ndarray,
            bin_redshift1 : np.ndarray,
            bin_redshift2 : np.ndarray,
            nproc:int=None, 
            sample_rate_desi:int=1, 
            sample_rate_hsc:int=1,
            logger:logging.Logger=None
        ):

        assert logger is not None, 'Logger not provided'
        self.logger = logger

        avb_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
        assert tgt in avb_tgt, f'{tgt} not in {avb_tgt}'
        self.tgt = tgt

        if nproc is None: 
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        self.bin_distances = bin_distances

        ## Filesystem setup
        fs = {}
        self.output_dir = Path(output_dir, tgt)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        fs['outdir'] = Path(self.output_dir)

        if not Path(self.output_dir, 'cov').exists():
            Path(self.output_dir, 'cov').mkdir(parents=True)

        # Grabbing the catalogs on initialisation
        fs['catalog1'] = fetch_desi_files(tgt, randoms=False)
        fs['randoms1'] = fetch_desi_files(tgt, randoms=True)
        fs['catalog2'] = fetch_hsc_files(randoms=False)
        fs['randoms2'] = fetch_hsc_files(randoms=True)
        
        ## Loading the MOC footprint
        self.moc = moc

        # Keep the filesystem
        self.fs = fs 

        # Sample rate for randoms
        self.sample_rate_desi = sample_rate_desi
        self.sample_rate_hsc = sample_rate_hsc

        trd = time.time()
        logger.info(f'Collating randoms ...')
        self.randoms1 = sample_randoms_on_moc(
            list(self.fs['randoms1'][:5]),
            ra_col=self.ra_desi_col,
            dec_col=self.dec_desi_col,
            w_col=self.w_desi_col,
            z_col=self.z_desi_col,
            moc=self.moc,
            )
        self.randoms1 = self.randoms1[::self.sample_rate_desi]
        logger.info(
            f'Collated {len(self.randoms1)} DESI randoms in {time.time()-trd:.2f} seconds'
            f' with sample rate {self.sample_rate_desi}'
            )
        trh = time.time()
        self.randoms2 = sample_randoms_on_moc(
            list(self.fs['randoms2']), 
            ra_col='ra',
            dec_col='dec',
            w_col=None,
            z_col=None,
            moc=self.moc,
            )
        self.randoms2 = self.randoms2[::self.sample_rate_hsc]
        logger.info(
            f'Collated {len(self.randoms2)} HSC randoms in {time.time()-trh:.2f} seconds'
            )

        tid = time.time()
        self.data1 = sample_file_on_moc(
            self.fs['catalog1'], 
            ra_col=self.ra_desi_col, 
            dec_col=self.dec_desi_col, 
            weight_col=self.w_desi_col, 
            z_col=self.z_desi_col,
            moc=self.moc
            )
        self.logger.info(f'Read DESI data in {time.time()-tid:.2f} seconds')
        tih = time.time()
        self.data2 = sample_file_on_moc(
            file=self.fs['catalog2'], 
            ra_col=self.ra_hsc_col, 
            dec_col=self.dec_hsc_col, 
            weight_col=self.w_hsc_col,
            z_col=self.z_hsc_col, 
            moc=self.moc
            )
        self.logger.info(f'Read HSC data in {time.time()-tih:.2f} seconds')

        # Setup redshift masks
        self.zmask_data1 = np.digitize(self.data1[self.z_desi_col], bin_redshift1, right=True)
        self.zmask_data2 = np.digitize(self.data2[self.z_hsc_col], bin_redshift2, right=True)
        self.zmask_randoms1 = np.digitize(self.randoms1[self.z_desi_col], bin_redshift1, right=True)

        ## rp2 needs no subsampling on redshift
        rp2 = [
            self.randoms2[self.ra_hsc_randoms_col],
            self.randoms2[self.dec_hsc_randoms_col]
            ]
        logger.info(f'Randoms2 shape {self.randoms2[self.ra_hsc_randoms_col].shape}')
        logger.info('Subsampling randoms2 ...')
        subsampler = KMeansSubsampler(
            mode='angular', 
            # The largest, most complete dataset we have is rp2
            # and no redshift sampling is needed for jackknife
            positions=rp2, 
            nsamples=64, # default was 128
            nside=256,  # default seems to be 512 lets go for lower for now as a test
            random_state=42, 
            position_type='rd'
            )
        labels = subsampler.label(rp2)
        self.subsampler = subsampler
        subsampler.log_info(f'Labels from {labels.min()} to {labels.max()}.')

    def run(self, bin_index1, bin_index2, moc_index):
        
        outfile = Path(
            self.output_dir, 
            'cov',
            f'{self.tgt}__b1x{bin_index1}_b2x{bin_index2}_moc{moc_index}.npy'
            )
        if outfile.exists():
            self.logger.info(f'File {outfile} already exists, skipping')
            return
        # Setup redshift masking
        z_mask_d1 = (self.zmask_data1 == bin_index1)
        z_mask_d2 = (self.zmask_data2 == bin_index2)
        # DESI randoms need to be redshift masked 
        z_mask_r1 = (self.zmask_randoms1 == bin_index1)

        self.logger.info(f'N data1: {np.sum(z_mask_d1)}' + f', N randoms1: {np.sum(z_mask_r1)}')
        self.logger.info(f'N data2: {np.sum(z_mask_d2)}' + f', N randoms2: {len(self.randoms2)}')

        dp1 = [
            self.data1[self.ra_desi_col][z_mask_d1], 
            self.data1[self.dec_desi_col][z_mask_d1]
            ]
        dp2 = [
            self.data2[self.ra_hsc_col][z_mask_d2], 
            self.data2[self.dec_hsc_col][z_mask_d2]
            ]
        rp1 = [
            self.randoms1[self.ra_desi_col][z_mask_r1],
            self.randoms1[self.dec_desi_col][z_mask_r1]
            ]
        rp2 = [
            self.randoms2[self.ra_hsc_randoms_col],
            self.randoms2[self.dec_hsc_randoms_col]
            ]
        dw1 = self.data1[self.w_desi_col][z_mask_d1]
        dw2 = self.data2[self.w_hsc_col][z_mask_d2]

        rw1 = self.randoms1[self.w_desi_col][z_mask_r1]
        rw2 = None

        ds1 = self.subsampler.label(dp1)
        ds2 = self.subsampler.label(dp2)
        rs1 = self.subsampler.label(rp1)
        rs2 = self.subsampler.label(rp2)

        tpcf = TwoPointCorrelationFunction(
            edges=self.bin_distances,

            data_positions1=dp1,
            data_positions2=dp2,

            randoms_positions1=rp1,
            randoms_positions2=rp2,

            data_weights1=dw1,
            data_weights2=dw2,

            randoms_weights1=rw1,
            randoms_weights2=rw2,

            data_samples1=ds1,
            data_samples2=ds2,

            randoms_samples1=rs1,
            randoms_samples2=rs2,

            nthreads=self.nproc,
            mode='theta',
            position_type='rd', # 'rd' for RA/Dec
            engine='corrfunc',
            estimator='landyszalay',
        )
        tpcf.save(outfile)

class DESIAutoCorrelation(CorrelationMeta):
    def __init__(
            self, 
            tgt : str, 
            moc : MOC, 
            output_dir :str | Path, 
            bin_distances : np.ndarray,
            bin_redshift1 : np.ndarray,
            nproc:int=None, 
            sample_rate_desi:int=1, 
            logger:logging.Logger=None
            ):
        
        assert logger is not None, 'Logger not provided'
        self.logger = logger

        self.ra_desi_col = 'RA'
        self.dec_desi_col = 'DEC'
        self.w_desi_col = 'WEIGHT'
        self.z_desi_col = 'Z'

        avb_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
        assert tgt in avb_tgt, f'{tgt} not in {avb_tgt}'
        self.tgt = tgt

        if nproc is None:
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        self.bin_distances = bin_distances

        ## Filesystem setup
        fs = {}
        self.output_dir = Path(output_dir, tgt)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        fs['outdir'] = Path(self.output_dir)

        # Grabbing the catalogs on initialisation
        fs['catalog1'] = fetch_desi_files(tgt, randoms=False)
        fs['randoms1'] = fetch_desi_files(tgt, randoms=True)
        
        ## Loading the MOC footprint
        self.moc = moc

        # Keep the filesystem
        self.fs = fs 

        # Sample rate for randoms
        self.sample_rate_desi = sample_rate_desi

        trd = time.time()
        logger.info(f'Collating randoms ...')
        self.randoms1 = sample_randoms_on_moc(
            list(self.fs['randoms1'][:4]),
            ra_col=self.ra_desi_col,
            dec_col=self.dec_desi_col,
            w_col=self.w_desi_col,
            z_col=self.z_desi_col,
            moc=self.moc,
            )
        self.randoms1 = self.randoms1[::self.sample_rate_desi]
        logger.info(f'Collated DESI randoms in {time.time()-trd:.2f} seconds')

        tid = time.time()
        self.data1 = sample_file_on_moc(
            self.fs['catalog1'], 
            ra_col=self.ra_desi_col, 
            dec_col=self.dec_desi_col, 
            weight_col=self.w_desi_col, 
            z_col=self.z_desi_col,
            moc=self.moc
            )
        self.logger.info(f'Read DESI data in {time.time()-tid:.2f} seconds')

        # Setup redshift masks
        self.zmask_data1 = np.digitize(self.data1[self.z_desi_col], bin_redshift1, right=True)
        self.zmask_randoms1 = np.digitize(self.randoms1[self.z_desi_col], bin_redshift1, right=True)

        logger.info(f'z mask randoms desi{self.zmask_randoms1[:10]}, length {len(self.randoms1[self.z_desi_col])}')
        logger.info(f'z mask data desi {self.zmask_data1[:10]}, length {len(self.randoms1[self.z_desi_col])}') 

    def run(self, bin_index1, moc_index):

        # Setup redshift masking
        z_mask_d1 = (self.zmask_data1 == bin_index1)
        # DESI randoms need to be redshift masked 
        z_mask_r1 = (self.zmask_randoms1 == bin_index1)

        self.logger.info(f'N sources in data1: {np.sum(z_mask_d1)}')
        self.logger.info(f'N sources in randoms1: {np.sum(z_mask_r1)}')

        tpcf = TwoPointCorrelationFunction(
            edges=(np.linspace(0., 200., 51), np.linspace(-40, 40, 81)),#self.bin_distances,
            data_positions1=[
                self.data1[self.ra_desi_col][z_mask_d1], 
                self.data1[self.dec_desi_col][z_mask_d1],
                ct.z2dist(self.data1[self.z_desi_col][z_mask_d1])
                ],
            randoms_positions1=[
                self.randoms1[self.ra_desi_col][z_mask_r1],
                self.randoms1[self.dec_desi_col][z_mask_r1],
                ct.z2dist(self.randoms1[self.z_desi_col][z_mask_r1]),
                ],
            data_weights1=self.data1[self.w_desi_col][z_mask_d1],
            randoms_weights1=self.randoms1[self.w_desi_col][z_mask_r1],
            nthreads=self.nproc,
            mode='rppi', #'theta',
            position_type='rdd', # 'rd' for RA/Dec #'rdd'
            engine='corrfunc',
            estimator='landyszalay',
        )
        outfile = Path(
            self.output_dir, 
            f'{self.tgt}__b1x{bin_index1}_moc{moc_index}.npy'
            )
        tpcf.save(outfile)

class HSCAutoCorrelation():
    def __init__(
            self, 
            moc : MOC, 
            output_dir :str | Path, 
            bin_distances : np.ndarray,
            bin_redshift1 : np.ndarray,
            nproc:int=None, 
            sample_rate_hsc:int=1, 
            logger:logging.Logger=None
            ):
        
        assert logger is not None, 'Logger not provided'
        self.logger = logger

        self.ra_hsc_col = 'RA'
        self.ra_randoms_hsc_col = 'ra'
        self.dec_hsc_col = 'Dec'
        self.dec_randoms_hsc_col = 'dec'
        self.w_hsc_col = 'weight'
        self.z_hsc_col = 'dnnz_photoz_best'

        if nproc is None:
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        self.bin_distances = bin_distances

        ## Filesystem setup
        fs = {}
        self.output_dir = Path(output_dir, 'HSC')
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        fs['outdir'] = Path(self.output_dir)

        # Grabbing the catalogs on initialisation
        fs['catalog1'] = fetch_hsc_files(randoms=False)
        fs['randoms1'] = fetch_hsc_files(randoms=True)
        
        ## Loading the MOC footprint
        self.moc = moc

        # Keep the filesystem
        self.fs = fs 

        # Sample rate for randoms
        self.sample_rate_hsc = sample_rate_hsc

        trd = time.time()
        logger.info(f'Collating randoms ...')
        self.randoms1 = sample_randoms_on_moc(
            list(self.fs['randoms1']),
            ra_col=self.ra_randoms_hsc_col,
            dec_col=self.dec_randoms_hsc_col,
            w_col=None,
            z_col=None,
            moc=self.moc,
            )
        self.randoms1 = self.randoms1[::self.sample_rate_hsc]
        logger.info(
            f'Collated HSC randoms in {time.time()-trd:.2f} seconds'
            f' with sample rate {self.sample_rate_hsc}'
            f' and {len(self.randoms1)} randoms'
            )

        tid = time.time()
        self.data1 = sample_file_on_moc(
            self.fs['catalog1'], 
            ra_col=self.ra_hsc_col, 
            dec_col=self.dec_hsc_col, 
            weight_col=self.w_hsc_col, 
            z_col=self.z_hsc_col,
            moc=self.moc
            )
        self.logger.info(f'Read DESI data in {time.time()-tid:.2f} seconds')

        # Setup redshift masks
        self.zmask_data1 = np.digitize(self.data1[self.z_hsc_col], bin_redshift1, right=True)
        logger.info(f'z mask data desi {self.zmask_data1[:10]}, length {len(self.zmask_data1)}') 

    def run(self, bin_index1, moc_index):

        # Setup redshift masking
        z_mask_d1 = (self.zmask_data1 == bin_index1)

        self.logger.info(f'N sources in data1: {np.sum(z_mask_d1)}')

        tpcf = TwoPointCorrelationFunction(
            edges=self.bin_distances,
            data_positions1=[
                self.data1[self.ra_hsc_col][z_mask_d1], 
                self.data1[self.dec_hsc_col][z_mask_d1]
                ],
            data_positions2=None,
            randoms_positions1=[
                self.randoms1[self.ra_randoms_hsc_col],
                self.randoms1[self.dec_randoms_hsc_col]
                ],
            randoms_positions2=None,
            data_weights1=self.data1[self.w_hsc_col][z_mask_d1],
            data_weights2=None,
            randoms_weights1=None,
            randoms_weights2=None,
            nthreads=self.nproc,
            mode='theta',
            position_type='rd', # 'rd' for RA/Dec
            engine='corrfunc',
            estimator='landyszalay',
        )
        outfile = Path(
            self.output_dir, 
            f'HSC__b1x{bin_index1}_moc{moc_index}.npy'
            )
        tpcf.save(outfile)

## Generic methods for each class
def process_random_file(f, ra_col, dec_col, w_col, z_col, moc):
    if z_col is None or w_col is None:
        cols = [ra_col, dec_col]
    else:
        cols = [ra_col, dec_col, w_col, z_col]
    
    try:
        with fio.FITS(str(f)) as rand:
            tbl = rand[1]

            nrows = tbl.get_nrows()
            data = tbl.read(columns=cols)

            if moc is not None:
                print(f"Filtering {nrows} randoms in {f} using MOC")
                tc = time.time()
                coords = SkyCoord(data[ra_col]*u.deg, data[dec_col]*u.deg, frame='icrs')
                mask = moc.contains_skycoords(coords)
                del coords
                data = data[np.flatnonzero(mask)]
                del mask
                print(f"Filtered in {time.time()-tc:.2f} seconds")
            
            return data if len(data) > 0 else None
    
    except Exception as e:
        print(f"Error processing file {f}: {e}")
        return None

def sample_randoms_on_moc(
        random_files, 
        ra_col, 
        dec_col, 
        w_col=None, 
        z_col=None, 
        moc=None, 
        num_processes=None
        ):
    """
    Multiprocessed random sampling with optional MOC filtering
    """
    assert len(random_files) > 0, f"No random files "
    
    if num_processes is None:
        num_processes = max(mp.cpu_count()-2, 1)
    
    tp = time.time()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_random_file, [
            (f, ra_col, dec_col, w_col, z_col, moc) 
            for f in random_files
        ])
    print(f"Processed {len(random_files)} random files in {time.time()-tp:.2f} seconds")

    randoms = [r for r in results if r is not None]
    
    if len(randoms) == 0:
        raise ValueError("No valid random data found")
    
    size = sum(len(r) for r in randoms)
    print(f"Collated {size} randoms (from {len(random_files)} files)")
    
    return np.concatenate(randoms)

def sample_file_on_moc(file, ra_col, dec_col, weight_col, z_col, moc=None):

    with fio.FITS(str(file)) as f:
        tbl = f[1]
        data = tbl.read(columns=[ra_col, dec_col, weight_col, z_col])
        if moc is not None:
            print(f"Filtering {tbl.get_nrows()} {len(data)} rows in {Path(file).stem} using MOC")
            coords = SkyCoord(data[ra_col] * u.deg, data[dec_col] * u.deg, frame='icrs')
            mask = moc.contains_skycoords(coords)
            del coords
            data = data[mask]
            print(f"Filtered {np.sum(mask)} rows in {Path(file).stem} using MOC")
            del mask

    return np.array(data)


def setup_crosscorr_logging(log_file='logs/output', log_level=logging.INFO):
    """
    Set up logging with both console and file output
    
    Args:
        log_file (str): Path to the log file
        log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_file = str(Path(log_file).with_suffix(''))
    log_file += f'_{time.strftime("%Y%m%d_%H%M%S")}.log'
    print(f"Logging to {log_file}")
    log_dir = Path(log_file).parent
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    def log_memory_usage():
        """
        Log current process memory usage
        """
        try:
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1e6
            logger.info(f"Memory Usage: {memory_usage:.2f} MB")
        except Exception as e:
            logger.error(f"Could not log memory usage: {e}")

    logger.memory_usage = log_memory_usage

    return logger

def fetch_desi_files(tgt, randoms=False, pip_weights=False, sims=False):
    try:
        if sims:
            raise NotImplementedError
        else:
            root = Path(
                '/global/cfs/projectdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v1.1/'
                )
            if pip_weights:
                root = Path(root, 'PIP')
            else:
                root = Path(root, 'nonKP')
            path = f'{tgt}{"_[0-9]*_" if randoms else "_"}clustering{".ran" if randoms else ".dat"}.fits'
            files = list(root.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found for path: {path}")
            if len(files) == 1:
                files = files[0]
            return files
    except PermissionError:
        logging.error(f"Permission denied accessing DESI files and randoms = {randoms}")
        raise

def fetch_hsc_files(randoms=False, include_dud=False, sims=False):
    try:
        if sims and randoms:
            root = Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/sims'
                )
            raise NotImplementedError
        elif sims:
            return Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/sims/hsc_y3_sims.fits'
                )
        elif randoms:
            #WARNING : this path root currently does not contain D/UD randoms as they
            #were deemed unnecessary for the clustering analysis
            root = Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/randoms'
                )
            return list(root.glob(f'edge_sc_cr_hscr{"*" if include_dud else "[0-9]"}.fits'))
        elif not sims and not randoms:
            return Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat.fits'
                )
    except PermissionError:
        logging.error(f"Permission denied accessing HSC files and randoms = {randoms}")
        raise
    except FileNotFoundError:
        logging.error(f"HSC catalog file not found and randoms = {randoms}") 
        raise

class CorrFileReader():
    '''
    Utility class to grab correctly formatted file names for the cross-correlation
    analysis. Provide a ROOT and the directory has to be with the expected shape.
    '''
    def __init__(self, ROOT):
        self.ROOT = Path(ROOT)
        assert self.ROOT.exists(), f"Path {self.ROOT} does not exist"

    def get_file(self, b1, b2, moc, tgt):
        """
        Get the file name for given redshift bins and MOC.
        """
        DIR = self.ROOT / tgt 
        return f'{DIR}/{tgt}__b1x{b1}_b2x{b2}_moc{moc}.npy'
    
    def get_auto_file(self, b1, moc, tgt):
        """
        Get the file name for given redshift bins and MOC.
        """
        DIR = self.ROOT / tgt 
        return f'{DIR}/{tgt}__b1x{b1}_moc{moc}.npy'

    def get_bins(self, name):
        return np.loadtxt(f'{self.ROOT}/bins/bins_{name}.txt', dtype=float)
    
    def get_cov_result(self, tgt):
        """
        Get the covariance result for given redshift bins and MOC.
        """
        covdir = Path(self.ROOT, tgt, 'cov')
        if not covdir.exists():
            raise FileNotFoundError(f"Covariance directory {covdir} does not exist")
        else:
            files = covdir.glob(f'*.npy')
            return list(files)
        
    def get_cov_file(self, b1, b2, moc, tgt):
        """
        Get the covariance file name for given redshift bins and MOC.
        """
        covdir = Path(self.ROOT, tgt, 'cov')
        file = covdir / f'{tgt}__b1x{b1}_b2x{b2}_moc{moc}.npy'
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist")
        else:
            return file
            
        