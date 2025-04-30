'''
This script holds wrapper functions and utilities for 
calculating and plotting correlation matrices.
'''
import time 
import math
import os
import fitsio as fio
import numpy as np
import logging
import psutil
import multiprocessing as mp

from abc import ABC, abstractmethod
from mocpy import MOC
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler

import src.statistics.cosmotools as ct
from src.statistics.corrfiles import fetch_desi_files, fetch_hsc_files, setup_crosscorr_logging

class CorrelationMeta(ABC):
    '''
    Meta class for the cross-correlation analysis, ruling out columns, 
    redshift bins and other parameters, as well as a few setup and utilities.
    '''

    # Attributes that each cross corr code needs to know about
    ra_hsc_col = 'ra'
    dec_hsc_col = 'dec'
    ra_hsc_randoms_col = 'ra'
    dec_hsc_randoms_col = 'dec'
    w_hsc_col = 'weight'
    z_hsc_col = 'dnnz_photoz_best'
    z_hsc_randoms_col = 'redshift'
    z_bin_hsc_col = 'z_bin'

    ra_desi_col = 'RA'
    dec_desi_col = 'DEC'
    w_desi_col = 'WEIGHT'
    z_desi_col = 'Z'
    z_desi_randoms_col = 'Z'

    ## MOC list 
    moc_list = [
            Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/mocs/', 
                f'hsc_moc{i+1}.fits'
            )
            for i in range(0, 4)
        ]

    ## Defining fiducial bins here 

    # from 1 arcmin to 100 arcmin
    bins_theta = np.logspace(math.log(0.2, 10), math.log(100, 10), 33, base=10)/60

    #np.linspace(0.0001, 2, 41)
    bins_rp = np.linspace(0.1, 100, 26)
    bins_rppi_s = np.linspace(0., 200., 51)
    bins_rppi_mu = np.linspace(-100, 100, 21)

    bins_bgs = np.arange(0, 0.6, 0.1) # 0 < z < 0.6
    bins_lrg = np.arange(0.4, 1.2, 0.1) # 0.4 < z < 1
    bins_elg = np.arange(0.8, 1.7, 0.1) # 0.6 < z < 1.6 => 0.8 < z < 1.6 in redshift distribution
    bins_qso = np.arange(0.9, 3.1, 0.3) # 0.9 < z < 2.1

    bins_hsc = np.arange(0.3, 1.8, 0.3) # 0.3 < z <= 1.5 

    bins_tracers = {
        'LRG': bins_lrg,
        'ELGnotqso': bins_elg,
        'QSO': bins_qso,
        'BGS_ANY': bins_bgs,
        'HSC': bins_hsc,
    }
    bins_mode = {
        'theta': bins_theta,
        'rp' : bins_rp,
        'rppi_s' : bins_rppi_s,
        'rppi_mu' : bins_rppi_mu
    }
    bins_all = {**bins_tracers, **bins_mode}

    @staticmethod
    def save_bins(root):
        """
        Save the bins to a file
        """
        bin_dir = Path(root, 'bins')
        if not bin_dir.exists():
            bin_dir.mkdir(parents=True)
        np.savez(
            Path(bin_dir, 'bins_all.npz'),
            **{k: v for k, v in CorrelationMeta.bins_all.items()}
        )

    def __init__(
            self, 
            logger : logging.Logger, 
            moc : MOC, 
            tgt1=None, 
            tgt2=None,
            output_dir=None, 
            sims_version=0,
            use_zbin=False,
            weight_type='nonKP',
            sample_rate_desi=1,
            sample_rate_hsc=1, 
            corr_type='theta',
            nproc=None
            ):
        assert logger is not None, 'Logger not provided'
        self.logger = logger
        
        # use_zbin is used for HSC only
        self.use_zbin = use_zbin
        
        # rename the class attributes if using simulations bc not the same class names
        self.sims = sims_version > 0
        self.sims_version = sims_version
        # override the columns if using simulations because they have different names and it's annoying
        if self.sims:
            self.ra_hsc_col = 'RA'
            self.dec_hsc_col = 'DEC'
            self.ra_hsc_randoms_col = 'ra'
            self.dec_hsc_randoms_col = 'dec'
            self.w_hsc_col = None
            self.z_hsc_col = 'Z'
            self.z_hsc_randoms_col = 'redshift'

            self.ra_desi_col = 'ra'
            self.dec_desi_col = 'dec'
            self.w_desi_col = None
            self.z_desi_col = 'z'
            self.z_desi_randoms_col = 'redshift'

        self.autocorr = False
        self.use_hsc = False
        self.use_desi = False

        desi_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
        hsc_tgt = ['HSC']
        # Idea : 
        # get the couple targets. If one is not specified, assume both are equal
        assert tgt1 is not None and tgt2 is not None, 'tgt1 and tgt2 cannot be None simultaneously'
        if tgt2 is None:
            tgt2 = tgt1
        if tgt1 is None:
            tgt1 = tgt2

        if tgt1 == tgt2:
            self.autocorr = True
            if tgt1 in desi_tgt:
                self.use_desi = True
            elif tgt1 in hsc_tgt:
                self.use_hsc = True
            else:
                raise ValueError(f'Unknown target {tgt1}')
        
        # two different targets are set : this is cross-correlation
        else:
            self.autocorr = False
            if tgt1 in desi_tgt or tgt2 in desi_tgt:
                self.use_desi = True
            if tgt1 in hsc_tgt or tgt2 in hsc_tgt:
                self.use_hsc = True
            if tgt1 in hsc_tgt and tgt2 in desi_tgt:
                # switch targets; useful for later
                tgt1, tgt2 = tgt2, tgt1

        self.tgt1 = tgt1
        self.tgt2 = tgt2

        if not self.use_desi and not self.use_hsc:
            raise ValueError(f'Unknown targets {tgt1} and {tgt2}')
    
        # once targets are figured out we can call the bins
        bin_redshift1 = self.bins_tracers[tgt1]
        bin_redshift2 = self.bins_tracers[tgt2]

        # which edges and correlation type to use :
        self.corr_type = corr_type
        self.pos_type = 'rd' if corr_type == 'theta' else 'rdd'

        if self.corr_type == 'rppi':
            self.edges = (self.bins_mode['rppi_s'], self.bins_mode['rppi_mu'])
        else:
            self.edges = self.bins_mode[self.corr_type]

        self.logger.info(f'mode : {self.corr_type}')
        self.logger.info(f'edges : {self.edges}') 
        self.logger.info(f'edges : {self.pos_type}')

        # Setup multiprocessing; can do mpi4py later on
        if nproc is None: 
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        # Filesystem setup
        fs = {}
        self.output_dir = Path(output_dir, f'{tgt1}x{tgt2}')
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        fs['outdir'] = Path(self.output_dir)

        # Grabbing the catalogs on initialisation
        logger.info(
            f'Grabbing catalogs for {tgt1} and {tgt2} ... ' 
            f'weight_type={weight_type}, sims={self.sims}, '
            )
        if self.use_desi:
            fs['catalog1'] = fetch_desi_files(
                tgt1, randoms=False, weight_type=weight_type, sims=self.sims, sims_version=self.sims_version
                )
            fs['randoms1'] = fetch_desi_files(
                tgt1, randoms=True, weight_type=weight_type, sims=self.sims, sims_version=self.sims_version
                )

        if self.use_hsc:
            fs['catalog2'] = fetch_hsc_files(
                randoms=False, sims=self.sims, include_dud=False, sims_version=self.sims_version
                )
            fs['randoms2'] = fetch_hsc_files(
                randoms=True, sims=self.sims, include_dud=False, sims_version=self.sims_version
                )
        logger.info(fs)

        # Loading the MOC footprint
        self.moc = moc

        # Keep the filesystem
        self.fs = fs 

        # Sample rate for randoms
        self.sample_rate_desi = sample_rate_desi
        self.sample_rate_hsc = sample_rate_hsc

        logger.info(f'Collating randoms ...')
        if 'randoms1' in self.fs:
            trd = time.time()
            self.randoms1 = sample_randoms_on_moc(
                # use onle the first 5 randoms for now, it's fine... 
                # could do more but not really worth the hassle
                self.fs['randoms1'][:5] if not self.sims else self.fs['randoms1'],
                ra_col=self.ra_desi_col,
                dec_col=self.dec_desi_col,
                w_col=self.w_desi_col,
                z_col=self.z_desi_randoms_col,
                moc=self.moc,
                )
            all_r_length = len(self.randoms1)
            self.randoms1 = self.randoms1[::self.sample_rate_desi]
            samp_r_length = len(self.randoms1)
        
            logger.info(
                f'Collated DESI randoms in {time.time()-trd:.2f}s. ' 
                f'Reduction : {samp_r_length/all_r_length*100:.2f}% ({all_r_length} -> {samp_r_length})'
                )

        if 'randoms2' in self.fs:
            trh = time.time()
            self.randoms2 = sample_randoms_on_moc(
                self.fs['randoms2'], 
                ra_col=self.ra_hsc_randoms_col,
                dec_col=self.dec_hsc_randoms_col,
                w_col=None,
                z_col=None if not self.sims else self.z_hsc_randoms_col,
                moc=self.moc,
                )
            all_r_length = len(self.randoms2)
            self.randoms2 = self.randoms2[::self.sample_rate_hsc]
            samp_r_length = len(self.randoms2)
            logger.info(
                f'Collated HSC randoms in {time.time()-trh:.2f}s. ' 
                f'Reduction : {samp_r_length/all_r_length*100:.2f}% ({all_r_length} -> {samp_r_length})'
                )

        if self.use_desi:
            tid = time.time()
            self.data1 = sample_file_on_moc(
                self.fs['catalog1'], 
                ra_col=self.ra_desi_col, 
                dec_col=self.dec_desi_col, 
                # no weights when cross correlating simulations
                weight_col=self.w_desi_col if not self.sims else None, 
                z_col=self.z_desi_col,
                moc=self.moc
                )
            logger.info(f'Read DESI data in {time.time()-tid:.2f} seconds ({len(self.data1)} rows)')
        if self.use_hsc:
            tih = time.time()
            self.data2 = sample_file_on_moc(
                self.fs['catalog2'], 
                ra_col=self.ra_hsc_col, 
                dec_col=self.dec_hsc_col, 
                weight_col=self.w_hsc_col if not self.sims else None,
                z_col=self.z_hsc_col if not self.use_zbin else self.z_bin_hsc_col, 
                moc=self.moc
                )
            logger.info(f'Read HSC data in {time.time()-tih:.2f} seconds ({len(self.data2)} rows)')

        # Setup redshift masks
        if self.use_desi:
            self.zmask_data1 = np.digitize(
                self.data1[self.z_desi_col], 
                bin_redshift1, 
                right=True
                )
            self.zmask_randoms1 = np.digitize(
                self.randoms1[self.z_desi_randoms_col], 
                bin_redshift1, 
                right=True
                )
        if self.use_hsc:
            if self.use_zbin:
                self.zmask_data2 = self.data2[self.z_bin_hsc_col]
            else:
                self.zmask_data2 = np.digitize(
                    self.data2[self.z_hsc_col], 
                    bin_redshift2, 
                    right=True
                    )
            if self.sims:
                self.zmask_randoms2 = np.digitize(
                    self.randoms2[self.z_hsc_randoms_col], 
                    bin_redshift2, 
                    right=True
                    )

        if self.autocorr:
            # specifying the attributes to None for consistency
            self.randoms2 = None 
            self.zmask_randoms2 = None
            self.zmask_data2 = None
            self.data2 = None
            
            if self.use_hsc:
                # in the case of autocorrelation with hsc, we move the 2nd dataset to be the first
                self.randoms1 = self.randoms2
                del self.randoms2
                self.data1 = self.data2
                del self.data2
                self.zmask_data1 = self.zmask_data2
                del self.zmask_data2

    
    @abstractmethod
    def run_corr(self):
        ''' 
        Abstract base method for running the correlation function, has to be overridden
        by inheriting classes.
        '''
        raise NotImplementedError(
            'run_corr() not implemented in the derived class. '
        )
    
    def run(self, bin_index1, bin_index2, moc_index):
        '''
        Base method to call when running cross corr.
        '''
        desccorr = f'{self.tgt1}x{self.tgt2}'
        outfile = Path(
            self.output_dir, 
            f'{desccorr}_b1x{bin_index1}_b2x{bin_index2}_moc{moc_index}.npy'
            )
        if outfile.exists():
            self.logger.info(f'File {outfile} already exists, skipping')
            return
        self.outfile = outfile
        
        # Setup redshift masking
        self.z_mask_d1 = (self.zmask_data1 == bin_index1)
        if self.use_desi:
            # DESI randoms need to be redshift masked 
            self.z_mask_r1 = (self.zmask_randoms1 == bin_index1)
        if not self.autocorr:
            # in the case of crosscorr, we apply a mask to the randoms
            self.z_mask_d2 = (self.zmask_data2 == bin_index2)

        if self.use_desi:
            self.logger.info(
                f'N data {self.tgt1}: {np.sum(self.z_mask_d1)}' + 
                f', N randoms {self.tgt1}: {np.sum(self.z_mask_r1)}'
                )
        if self.use_hsc and not self.autocorr:
            self.logger.info(
                f'N data {self.tgt2}: {np.sum(self.z_mask_d2)}' + 
                f', N randoms {self.tgt2}: {len(self.randoms2)}'
                )
        if self.use_hsc and self.autocorr:
            self.logger.info(
                f'N data {self.tgt1}: {np.sum(self.z_mask_d1)}' + 
                f', N randoms {self.tgt1}: {len(self.randoms1)}'
                )

        ## assertion safeties :
        assert len(self.data1) == len(self.z_mask_d1)
        assert len(self.randoms1) == len(self.z_mask_r1)

        self.run_corr()


class CrossCorrelation(CorrelationMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_corr(self):

        tpcf = TwoPointCorrelationFunction(
            edges=self.edges,
            data_positions1=[
                self.data1[self.ra_desi_col][self.z_mask_d1], 
                self.data1[self.dec_desi_col][self.z_mask_d1]
                ],
            data_positions2=[
                self.data2[self.ra_hsc_col][self.z_mask_d2], 
                self.data2[self.dec_hsc_col][self.z_mask_d2]
                ],
            randoms_positions1=[
                self.randoms1[self.ra_desi_col][self.z_mask_r1],
                self.randoms1[self.dec_desi_col][self.z_mask_r1]
                ],
            # HSC does not need redshift masking on randoms
            randoms_positions2=[
                self.randoms2[self.ra_hsc_randoms_col],
                self.randoms2[self.dec_hsc_randoms_col]
                ],
            data_weights1=self.data1[self.w_desi_col][self.z_mask_d1],
            data_weights2=self.data2[self.w_hsc_col][self.z_mask_d2],
            randoms_weights1=self.randoms1[self.w_desi_col][self.z_mask_r1] if not self.sims else None,
            randoms_weights2=None,
            nthreads=self.nproc,
            mode='theta',
            position_type='rd', # 'rd' for RA/Dec
            engine='corrfunc',
            estimator='landyszalay',
        )
        tpcf.save(self.outfile)

class JackknifeCrossCorrelation(CorrelationMeta):
    def __init__(
        self,
        nside: int = 256,
        nsamples: int = 64,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nside = nside
        self.nsamples = nsamples
        self.seed = seed

        if self.autocorr:
            if self.use_hsc:
                rpsamp = [
                    self.randoms1[self.ra_hsc_col], 
                    self.randoms1[self.dec_hsc_col]
                    ]
            if self.use_desi:
                rpsamp = [
                    self.randoms1[self.ra_desi_col], 
                    self.randoms1[self.dec_desi_col]
                    ]
        else:
            rpsamp = [
                self.randoms2[self.ra_hsc_randoms_col],
                self.randoms2[self.dec_hsc_randoms_col]
                ]
        if self.corr_type == 'rp':
            rpsamp.append(
                ct.z2dist(
                    self.randoms2[self.z_hsc_randoms_col]
                    )
                )
        if self.data2 is not None and self.randoms2 is not None:
            self.logger.info(f'Data2 length {len(self.data2)} and randoms2 length {len(self.randoms2)}')

        self.logger.info('Subsampling data2 with KMeansSubsampler...')
        subsampler = KMeansSubsampler(
            mode='angular' if self.corr_type == 'theta' else '3d', 
            # The largest, most complete dataset we have is rp2
            # and no redshift sampling is needed for jackknife
            positions=rpsamp, 
            nsamples=nsamples,
            nside=nside if self.corr_type == 'theta' else None,
            random_state=seed, 
            position_type='rd' if self.corr_type == 'theta' else 'rdd'
            )
        labels = subsampler.label(rpsamp)

        self.subsampler = subsampler
        self.logger.info(f'Labels from {labels.min()} to {labels.max()}.')

    def run_corr(self):
        
        #self.logger.info(
        #    f'N data1: {np.sum(self.z_mask_d1)}' + 
        #    f', N randoms1: {np.sum(self.z_mask_r1)}'
        #    )
        #self.logger.info(
        #    f'N data2: {np.sum(self.z_mask_d2)}' + 
        #    f', N randoms2: {len(self.randoms2)}'
        #    )

        dp1 = [
            self.data1[self.ra_desi_col][self.z_mask_d1], 
            self.data1[self.dec_desi_col][self.z_mask_d1]
            ]
        rp1 = [
            self.randoms1[self.ra_desi_col][self.z_mask_r1],
            self.randoms1[self.dec_desi_col][self.z_mask_r1]
            ]
        # if not doing autocorrelation, we need to add the second dataset
        if not self.autocorr:
            dp2 = [
                self.data2[self.ra_hsc_col][self.z_mask_d2], 
                self.data2[self.dec_hsc_col][self.z_mask_d2]
                ]
            rp2 = [
                self.randoms2[self.ra_hsc_randoms_col],
                self.randoms2[self.dec_hsc_randoms_col]
                ]
        else:
            rp2 = None
            dp2 = None
        
        if self.corr_type == 'rp':
            self.logger.info('Using redshift for distance calculation')
            dp1.append(
                ct.z2dist(self.data1[self.z_desi_col][self.z_mask_d1])
                )
            rp1.append(
                ct.z2dist(self.randoms1[self.z_desi_randoms_col][self.z_mask_r1])
                )
            if self.autocorr:
                rp2.append(
                    ct.z2dist(self.randoms2[self.z_hsc_randoms_col])
                    )
                dp2.append(
                    ct.z2dist(self.data2[self.z_hsc_col][self.z_mask_d2])
                    )
            
        if self.sims:
            # no weights for simulations
            dw1 = None
            dw2 = None
            rw1 = None
            rw2 = None
        else :
            dw1 = self.data1[self.w_desi_col][self.z_mask_d1]
            if self.autocorr:
                dw2 = None
            else:
                dw2 = self.data2[self.w_hsc_col][self.z_mask_d2]

            rw1 = self.randoms1[self.w_desi_col][self.z_mask_r1]
            rw2 = None

        # casting to float in case of weird dtypes
        dp1 = np.array(dp1, dtype=float)
        rp1 = np.array(rp1, dtype=float)
        if not self.autocorr:
            dp2 = np.array(dp2, dtype=float)
            rp2 = np.array(rp2, dtype=float)
        else:
            dp2 = None
            rp2 = None

        # subsampling with KMeans for jackknife
        ds1 = self.subsampler.label(dp1)
        rs1 = self.subsampler.label(rp1)
        if not self.autocorr:
            rs2 = self.subsampler.label(rp2)
            ds2 = self.subsampler.label(dp2)
        else:
            rs2 = None
            ds2 = None

        tpcf = TwoPointCorrelationFunction(
            edges=self.edges,

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
            mode=self.corr_type,
            position_type=self.pos_type, # 'rd' for RA/Dec, 'rdd' for RA/Dec/Dist
            engine='corrfunc',
            estimator='landyszalay',
        )
        tpcf.save(self.outfile)

class DESIAutoCorrelation(CorrelationMeta):
    def __init__(
            self, 
            tgt1 : str,
            tgt2 : str, 
            moc : MOC, 
            output_dir :str | Path, 
            nproc:int=None, 
            sample_rate_desi:int=1, 
            logger:logging.Logger=None
        ):
        assert tgt1==tgt2, f'Auto correlation only for {tgt1} and {tgt2} equal'
        super().__init__(
            tgt1=tgt1,
            tgt2=tgt2,
            moc=moc,
            sims=False,
            pip=False,
            output_dir=output_dir,
            nproc=nproc,
            sample_rate_desi=sample_rate_desi,
            logger=logger
        )
        
    def run_corr(self):

        tpcf = TwoPointCorrelationFunction(
            edges=(np.linspace(0., 200., 51), np.linspace(-40, 40, 81)),
            data_positions1=[
                self.data1[self.ra_desi_col][self.z_mask_d1], 
                self.data1[self.dec_desi_col][self.z_mask_d1],
                ct.z2dist(self.data1[self.z_desi_col][self.z_mask_d1])
                ],
            randoms_positions1=[
                self.randoms1[self.ra_desi_col][self.z_mask_r1],
                self.randoms1[self.dec_desi_col][self.z_mask_r1],
                ct.z2dist(self.randoms1[self.z_desi_randoms_col][self.z_mask_r1]),
                ],
            data_weights1=self.data1[self.w_desi_col][self.z_mask_d1],
            randoms_weights1=self.randoms1[self.w_desi_col][self.z_mask_r1],
            nthreads=self.nproc,
            mode='rppi',
            position_type='rdd', 
            engine='corrfunc',
            estimator='landyszalay',
        )
        tpcf.save(self.outfile)


## Generic methods for each class
def process_random_file(f, ra_col, dec_col, w_col, z_col, moc):

    cols = [
        c for c in [ra_col, dec_col, w_col, z_col] 
        if c is not None
        ]
    
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
    if isinstance(random_files, (str, Path)):
        random_files = [random_files]
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

def sample_file_on_moc(file, ra_col, dec_col, weight_col=None, z_col=None, moc=None):

    with fio.FITS(str(file)) as f:
        tbl = f[1]
        cols_to_read = [ra_col, dec_col, weight_col, z_col]
        if weight_col is None:
            cols_to_read = [ra_col, dec_col, z_col]
        if z_col is None:
            cols_to_read = [ra_col, dec_col, weight_col]
        if weight_col is None and z_col is None:
            cols_to_read = [ra_col, dec_col]
        data = tbl.read(columns=cols_to_read)
        if moc is not None:
            print(f"Filtering {tbl.get_nrows()} {len(data)} rows in {Path(file).stem} using MOC")
            coords = SkyCoord(data[ra_col] * u.deg, data[dec_col] * u.deg, frame='icrs')
            mask = moc.contains_skycoords(coords)
            del coords
            data = data[mask]
            print(f"Filtered {np.sum(mask)} rows in {Path(file).stem} using MOC")
            del mask

    return np.array(data)

def figure_out_class(tgt1, tgt2=None, jackknife=False):
    desi_avb = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    hsc_avb = ['HSC']
    avb = desi_avb + hsc_avb
    if tgt1 is None and tgt2 is None:
        raise ValueError('No target provided')
    if tgt2 is None:
        tgt2 = tgt1
    if tgt1 is None:
        tgt1 = tgt2
    
    assert tgt1 in avb and tgt2 in avb, f'Unknown target {tgt1} or {tgt2}'

    # Jackknife cross correlation now rules them all 
    if jackknife:
        return JackknifeCrossCorrelation
    else:
        # could be ruled out later maybe idk
        return CrossCorrelation