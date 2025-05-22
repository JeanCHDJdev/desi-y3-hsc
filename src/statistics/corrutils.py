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
import multiprocessing as mp
import pandas as pd

from astropy.table import Table
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
    w_fkp_desi_col = 'WEIGHT_FKP'
    w_comp_desi_col = 'WEIGHT_COMP'
    z_desi_col = 'Z'
    z_desi_randoms_col = 'Z'

    # use redshift column to go to the h-1Mpc distance
    distance_col = 'dist'

    ## MOC list 
    moc_list = sorted([
            Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/mocs/', 
                f'hsc_moc{i+1}.fits'
            )
            for i in range(0, 4)
        ])

    # ----------Defining fiducial bins here---------- 
    # We define the required bins in Mpc/h units (comoving)
    bins_rp = np.logspace(math.log(0.08, 10), math.log(20, 10), 21, base=10)

    bins_rppi_s = np.linspace(0., 200., 51)
    bins_rppi_mu = np.linspace(-100, 100, 21)

    bins_bgs = np.arange(0, 0.55, 0.05) # 0 < z < 0.5
    bins_lrg = np.arange(0.4, 1.15, 0.05) # 0.4 < z < 1.1
    bins_elg = np.arange(0.8, 1.68, 0.08) # 0.8 < z < 1.6 in redshift distribution
    #bins_elg = np.array([0.8, 0.9, 1.0, 1.1]) # for now reduce bin for compute power
    bins_qso = np.arange(0.9, 2.95, 0.15) # 0.9 < z < 2.8

    # use_zbin will override this choice
    bins_hsc = np.arange(0.3, 1.8, 0.3) # 0.3 < z <= 1.5 (tomographic binning has .3 bins)
    #bins_hsc = np.arange(0.3, 1.8, 0.3) # 0.3 < z <= 1.5 (tomographic binning has .3 bins)
    # if mini_bins : 
    #bins_hsc = np.arange(0, 2.825, 0.025)

    bins_tracers = {
        'LRG': bins_lrg,
        'ELGnotqso': bins_elg,
        'QSO': bins_qso,
        'BGS_ANY': bins_bgs,
        'HSC': bins_hsc,
    }
    bins_mode = {
        'rp' : bins_rp,
        'rppi_s' : bins_rppi_s,
        'rppi_mu' : bins_rppi_mu
    }
    bins_all = {**bins_tracers, **bins_mode}

    estimator_type = 'davispeebles' #'landyszalay' or 'davispeebles'

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

    # map MOC to caps
    capdict = {
        0 : 'NGC',
        1 : 'NGC',
        2 : 'SGC',
        3 : 'SGC',
    }

    def __init__(
            self, 
            logger : logging.Logger, 
            moc : MOC, 
            moc_index : int,
            skip_moc : bool = False,
            tgt1=None, 
            tgt2=None,
            output_dir=None, 
            sims_version=0,
            use_zbin=False,
            weight_type='nonKP',
            sample_rate_1=1,
            sample_rate_2=1, 
            corr_type='theta',
            nproc=None
            ):
        assert logger is not None, 'Logger not provided'
        self.logger = logger
        
        # use_zbin is used for HSC only
        self.use_zbin = use_zbin
        
        self.set_simulation_status(sims_version=sims_version)

        # figure out in which cap we are
        self.cap = self.capdict[moc_index]
        self.skip_moc = skip_moc

        self.autocorr = False
        self.double_desi = False
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

        # same target for both : this is autocorrelation
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
            if tgt1 in desi_tgt and tgt2 in desi_tgt:
                # case of cross correlation between two different DESI tracers
                self.double_desi = True

        self.tgt1 = tgt1
        self.tgt2 = tgt2

        self.logger.info(
            f'Using targets {self.tgt1} and {self.tgt2} '
            f'for cross correlation. '
            f'Autocorr={self.autocorr}, double_desi={self.double_desi}'
            f'using_hsc={self.use_hsc}, using_desi={self.use_desi}'
            )

        if not self.use_desi and not self.use_hsc:
            raise ValueError(f'Unknown targets {tgt1} and {tgt2}')
    
        # once targets are figured out we can call the bins
        self.bin_redshift1 = self.bins_tracers[tgt1]
        self.bin_redshift2 = self.bins_tracers[tgt2]

        # which edges and correlation type to use :
        self.corr_type = corr_type
        if self.corr_type not in ['theta', 'rp', 'rppi']:
            raise ValueError(
                f'corr_type {self.corr_type} not in {self.bins_mode.keys()}'
                )
        if self.corr_type == 'theta':
            self.distance_col = None
        self.pos_type = 'rd' if corr_type == 'theta' else 'rdd'

        # weights : here base (nonKP or PIP) + FKP + ...
        self.w_cols_to_operate = [
            self.w_desi_col, 
            self.w_fkp_desi_col,
            #self.w_comp_desi_col
            ]
        self.w_operator = '*'
        # usually PIP, nonKP...
        self.weight_type = weight_type

        if self.corr_type == 'rppi':
            self.edges = (self.bins_mode['rppi_s'], self.bins_mode['rppi_mu'])
        else:
            self.edges = self.bins_mode['rp' if self.corr_type == 'theta' else self.corr_type]

        self.logger.info(f'mode : {self.corr_type}')
        self.logger.info(f'edges : {self.edges} in Mpc/h')
        self.logger.info(f'pos type : {self.pos_type}')

        # Setup multiprocessing; can do mpi4py later on
        if nproc is None: 
            nproc = max(os.cpu_count()-2, 1)
        self.nproc = nproc

        # Filesystem setup
        self.output_dir = Path(output_dir, f'{tgt1}x{tgt2}')
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.logger.info('Estimator type : ' + self.estimator_type)

        # Grabbing the catalogs on initialisation
        logger.info(
            f'Grabbing catalogs for {tgt1} and {tgt2} ... ' 
            f'weight_type={weight_type}, sims={self.sims}, '
            )

        # Loading the MOC footprint
        self.moc = moc

        # Sample rate for randoms
        self.sample_rate_1 = sample_rate_1
        self.sample_rate_2 = sample_rate_2

        self.make_cats(
            bin_redshift1=self.bin_redshift1, 
            bin_redshift2=self.bin_redshift2
        )

    def set_simulation_status(self, sims_version=0):
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
            self.w_desi_fkp_col = None
            self.w_comp_desi_col = None
            self.z_desi_col = 'z'
            self.z_desi_randoms_col = 'redshift'

    def set_desi_tracer(self, tgt, bin_redshift):
        '''
        Returns cat, ran, zmask_cat, zmask_ran
        for a specific DESI tracer (tgt).

        Parameters
        ----------
        tgt : str
            Target name (e.g. LRG, ELGnotqso, QSO, BGS_ANY)
        bin_redshift : array
            Redshift binning for the target. Will digitize the redshift column
            and make the masks.
        '''
        catf = fetch_desi_files(
            tgt, 
            randoms=False, 
            weight_type=self.weight_type, 
            sims=self.sims, 
            sims_version=self.sims_version,
            cap=self.cap
            )
        ranf = fetch_desi_files(
            tgt, 
            randoms=True, 
            weight_type=self.weight_type, 
            sims=self.sims, 
            sims_version=self.sims_version,
            cap=self.cap,
            )
        if not self.sims:
            # use onle the first 6 random files for now, it's fine... 
            # could do more but not really worth the hassle
            ranf = ranf[:5]

        tid = time.time()
        cat = sample_file_on_moc(
            catf, 
            ra_col=self.ra_desi_col, 
            dec_col=self.dec_desi_col, 
            # no weights when cross correlating simulations
            main_weight_col=self.w_desi_col if not self.sims else None,
            weight_cols_to_operate=self.w_cols_to_operate if not self.sims else None, 
            z_col=self.z_desi_col,
            moc=self.moc if not self.skip_moc else None,
            distance_col=self.distance_col,
            operator=self.w_operator,
            )
        self.logger.info(f'Read DESI data in {time.time()-tid:.2f} seconds ({len(cat)} rows)')
        
        trd = time.time()
        ran = sample_randoms_on_moc(
            ranf,
            ra_col=self.ra_desi_col,
            dec_col=self.dec_desi_col,
            main_w_col=self.w_desi_col if not self.sims else None,
            weight_cols_to_operate=self.w_cols_to_operate if not self.sims else None,
            z_col=self.z_desi_randoms_col,
            moc=self.moc if not self.skip_moc else None,
            distance_col=self.distance_col,
            operator=self.w_operator,
            )
        all_r_length = len(ran)
        ran = ran[::self.sample_rate_1]
        samp_r_length = len(ran)
    
        self.logger.info(
            f'Collated DESI randoms in {time.time()-trd:.2f}s. ' 
            f'Reduction : {samp_r_length/all_r_length*100:.2f}% ({all_r_length} -> {samp_r_length})'
            )
        zmask_dat = np.digitize(
            cat[self.z_desi_col], 
            bin_redshift, 
            right=True
            )
        zmask_ran = np.digitize(
            ran[self.z_desi_randoms_col], 
            bin_redshift, 
            right=True
            )
        assert len(cat) == len(zmask_dat), 'cat and zmask_dat have different lengths'
        assert len(ran) == len(zmask_ran), 'ran and zmask_ran have different lengths'
        
        return cat, ran, zmask_dat, zmask_ran
    
    def set_hsc_tracer(self, bin_redshift):
        '''
        Returns cat, ran, zmask_cat, zmask_ran for HSC.
        '''
        catf = fetch_hsc_files(
            randoms=False,
            include_dud=False,
            sims=self.sims,
            sims_version=self.sims_version
        )
        ranf = fetch_hsc_files(
            randoms=True,
            include_dud=False,
            sims=self.sims,
            sims_version=self.sims_version
        )
        trh = time.time()
        ran = sample_randoms_on_moc(
            ranf, 
            ra_col=self.ra_hsc_randoms_col,
            dec_col=self.dec_hsc_randoms_col,
            main_w_col=None,
            z_col=None if not self.sims else self.z_hsc_randoms_col,
            moc=self.moc if not self.skip_moc else None,
            distance_col=self.distance_col,
            operator=self.w_operator,
            )
        all_r_length = len(ran)
        ran = ran[::self.sample_rate_2]
        samp_r_length = len(ran)
        self.logger.info(
            f'Collated HSC randoms in {time.time()-trh:.2f}s. ' 
            f'Reduction : {samp_r_length/all_r_length*100:.2f}% ({all_r_length} -> {samp_r_length})'
            )
        
        tih = time.time()
        # we can't use bins on HSC sims
        if self.sims:
            self.use_zbin = False
        cat = sample_file_on_moc(
            catf, 
            ra_col=self.ra_hsc_col, 
            dec_col=self.dec_hsc_col, 
            main_weight_col=self.w_hsc_col if not self.sims else None,
            # we go with both z col and z bin col in case we want to use the calibration cut
            # that only the bins know about (and it's important for HSC)
            z_col=self.z_hsc_col if not self.use_zbin else [self.z_hsc_col, self.z_bin_hsc_col], 
            moc=self.moc if not self.skip_moc else None,
            distance_col=self.distance_col,
            operator=self.w_operator
            )
        self.logger.info(f'Read HSC data in {time.time()-tih:.2f} seconds ({len(cat)} rows)')

        if self.sims:
            zmask_ran = np.digitize(
                ran[self.z_hsc_randoms_col], 
                bin_redshift, 
                right=True
                )
        else:
            # no z masking on HSC randoms for real HSC data
            zmask_ran = None
        if self.use_zbin:
            # zbins in HSC are 1-indexed. 0 = outside of the binning scheme
            zmask_bins = cat[self.z_bin_hsc_col]
            # here we do a second digitize to get the binning scheme complete
            zvalues = cat[self.z_hsc_col]
            zmask_data = np.digitize(
                zvalues, 
                bin_redshift, 
                right=True
                )
            # tomographic range
            ztomographic = [0.3, 1.5]
            # which bins are in the tomographic range ?
            ztomographic = [0.3, 1.5]
            # Mask redshifts inside tomographic range but with bad quality (tldr : calibration cut)
            inside_tomo_range = (zvalues > ztomographic[0]) & (zvalues < ztomographic[1])
            bad_quality = zmask_bins == 0

            # Zero out these values
            zmask_data[inside_tomo_range & bad_quality] = 0

        else:
            zmask_data = np.digitize(
                cat[self.z_hsc_col], 
                bin_redshift, 
                right=True
                )
        
        return cat, ran, zmask_data, zmask_ran
    
    def make_cats(self, bin_redshift1=None, bin_redshift2=None):
        '''
        Based on settings, makes self.data1, self.data2, self.randoms1, self.randoms2
        and self.z_bool_d1, self.z_bool_d2, self.z_bool_r1, self.z_bool_r2
        '''
        assert bin_redshift1 is not None, 'bin_redshift1 cannot be None'
        assert bin_redshift2 is not None, 'bin_redshift2 cannot be None'

        self.bin_redshift1 = bin_redshift1
        self.bin_redshift2 = bin_redshift2

        if self.use_desi:
            cat, ran, zmask_cat, zmask_ran = self.set_desi_tracer(self.tgt1, bin_redshift1)
        if self.double_desi:
            cat2, ran2, zmask_cat2, zmask_ran2 = self.set_desi_tracer(self.tgt2, bin_redshift2)
        if self.use_hsc and not self.double_desi: # flags should be incompatible, but just in case
            if self.autocorr:
                cat, ran, zmask_cat, zmask_ran = self.set_hsc_tracer(bin_redshift1)
            else:
                cat2, ran2, zmask_cat2, zmask_ran2 = self.set_hsc_tracer(bin_redshift2)
        if self.autocorr:
            cat2 = None
            ran2 = None
            zmask_cat2 = None
            zmask_ran2 = None
        
        self.data1 = cat
        self.randoms1 = ran
        self.zmask_data1 = zmask_cat
        self.zmask_randoms1 = zmask_ran
        self.data2 = cat2
        self.randoms2 = ran2
        self.zmask_data2 = zmask_cat2
        self.zmask_randoms2 = zmask_ran2
        #import ipdb; ipdb.set_trace()
        logging.info(
            'MAKE CATS : ' +
            f'N data {self.tgt1}: {len(self.data1)}' +
            f', N randoms {self.tgt1}: {len(self.randoms1)}' +
            f', N data {self.tgt2}: {len(self.data2) if self.data2 is not None else "None"}' +
            f', N randoms {self.tgt2}: {len(self.randoms2) if self.randoms2 is not None else "None"}'
            )
    
    def set_current_redshift_masks(self, bin1, bin2):
        self.z_bool_d1 = None
        self.z_bool_r1 = None
        self.z_bool_d2 = None
        self.z_bool_r2 = None
        
        if self.zmask_data1 is not None:
            self.z_bool_d1 = self.zmask_data1 == bin1
        if self.zmask_randoms1 is not None:
            self.z_bool_r1 = self.zmask_randoms1 == bin1
        if self.zmask_data2 is not None:
            self.z_bool_d2 = self.zmask_data2 == bin2
        if self.zmask_randoms2 is not None:
            self.z_bool_r2 = self.zmask_randoms2 == bin2

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
        
        self.set_current_redshift_masks(
            bin1=bin_index1, 
            bin2=bin_index2
            )
        zloc = (self.bin_redshift1[bin_index1-1] + self.bin_redshift1[bin_index1]) / 2
        self.theta_edges = ct.hMpc2arcsec(self.edges, zloc)/3600 #convert to degrees
        self.logger.info(f'Execution redshift = bins :{bin_index1}, {bin_index2}, zloc= {zloc}')
        self.logger.info('Theta edges : ' + str(self.theta_edges))

        self.logger.info(
            f'N data {self.tgt1}: {np.sum(self.z_bool_d1) if self.z_bool_d1 is not None else "None"}' + 
            f', N randoms {self.tgt1}: {np.sum(self.z_bool_r1) if self.z_bool_r1 is not None else "None"}'
            f', N data {self.tgt2}: {np.sum(self.z_bool_d2) if self.z_bool_d2 is not None else "None"}' +
            f', N randoms {self.tgt2}: {np.sum(self.z_bool_r2) if self.z_bool_r2 is not None else "None"}'
            )

        ## assertion safeties :
        assert len(self.data1) == len(self.z_bool_d1)
        if not self.use_hsc:
            assert len(self.randoms1) == len(self.z_bool_r1)

        # assert we don't have empty data because what the heck
        #import ipdb; ipdb.set_trace()
        assert len(self.randoms1[self.z_bool_r1]) > 0
        assert len(self.data1[self.z_bool_d1]) > 0
        if not self.autocorr:
            if self.z_bool_r1 is not None:
                assert len(self.randoms2[self.z_bool_r2]) > 0
            else:
                assert len(self.randoms2) > 0
            if self.z_bool_d2 is not None:
                assert len(self.data2[self.z_bool_d2]) > 0
        self.run_corr()

    def make_corr_data(self):
        '''
        Once data1, data2, randoms1, randoms2 are set up, we can make the data
        with the ra, dec, z columns and the weights. make_corr_data does exactly that.
        '''
        if self.use_desi:
            dp1 = [
                self.data1[self.ra_desi_col][self.z_bool_d1], 
                self.data1[self.dec_desi_col][self.z_bool_d1]
                ]
            rp1 = [
                self.randoms1[self.ra_desi_col][self.z_bool_r1],
                self.randoms1[self.dec_desi_col][self.z_bool_r1]
                ]
            if self.corr_type == 'rp' or self.corr_type == 'rppi':
                # autocorrelation case : rp1, dp1 are not used
                dp1.append(
                    self.data1[self.z_desi_col][self.z_bool_d1]
                    )
                rp1.append(
                    self.randoms1[self.z_desi_randoms_col]
                    )
        else:
            dp1 = [
                self.data1[self.ra_hsc_col][self.z_bool_d1], 
                self.data1[self.dec_hsc_col][self.z_bool_d1]
                ]
            rp1 = [
                self.randoms1[self.ra_hsc_randoms_col],
                self.randoms1[self.dec_hsc_randoms_col]
                ]   
            if self.corr_type == 'rp' or self.corr_type == 'rppi':
                # autocorrelation case : rp1, dp1 are not used
                dp1.append(
                    self.data1[self.z_hsc_col][self.z_bool_d1]
                    )
                rp1.append(
                    self.randoms1[self.z_hsc_randoms_col]
                    )
        # if not doing autocorrelation, we need to add the second dataset
        if not self.autocorr:
            if self.double_desi:
                # cross correlation between two different DESI tracers
                dp2 = [
                    self.data2[self.ra_desi_col][self.z_bool_d2], 
                    self.data2[self.dec_desi_col][self.z_bool_d2]
                    ]
                rp2 = [
                    self.randoms2[self.ra_desi_col][self.z_bool_r2],
                    self.randoms2[self.dec_desi_col][self.z_bool_r2]
                    ]
                if self.corr_type == 'rp' or self.corr_type == 'rppi':
                    # autocorrelation case : rp2, dp2 are not used
                    dp2.append(
                        self.data2[self.z_desi_col][self.z_bool_d2]
                        )
                    rp2.append(
                        self.randoms2[self.z_desi_randoms_col]
                        )
            else:
                # cross correlation with HSC
                dp2 = [
                    self.data2[self.ra_hsc_col][self.z_bool_d2], 
                    self.data2[self.dec_hsc_col][self.z_bool_d2]
                    ]
                rp2 = [
                    self.randoms2[self.ra_hsc_randoms_col],
                    self.randoms2[self.dec_hsc_randoms_col]
                    ]
                if self.corr_type == 'rp' or self.corr_type == 'rppi':
                    # autocorrelation case : rp2, dp2 are not used
                    dp2.append(
                        self.data2[self.z_hsc_col][self.z_bool_d2]
                        )
                    rp2.append(
                        self.randoms2[self.z_hsc_randoms_col]
                        )
        else:
            # autocorrelation case : rp2, dp2 are not used
            rp2 = None
            dp2 = None 
            
        dw1 = None
        dw2 = None
        rw1 = None
        rw2 = None

        # no weights for simulations
        if not self.sims:
            if self.use_desi:
                rw1 = self.randoms1[self.w_desi_col][self.z_bool_r1]
            else:
                rw1 = None
            rw2 = None

            if self.use_hsc and self.autocorr:
                colw = self.w_hsc_col
            else: 
                colw = self.w_desi_col
                dw1 = self.data1[colw][self.z_bool_d1]
            
            if self.double_desi:
                dw2 = self.data2[self.w_desi_col][self.z_bool_d2]
                rw2 = self.randoms2[self.w_desi_col][self.z_bool_r2]

        # casting to float in case of weird dtypes
        dp1 = np.array(dp1, dtype=float)
        rp1 = np.array(rp1, dtype=float)
        if not self.autocorr:
            dp2 = np.array(dp2, dtype=float)
            rp2 = np.array(rp2, dtype=float)
        else:
            dp2 = None
            rp2 = None

        return dp1, dp2, rp1, rp2, dw1, dw2, rw1, rw2

class CrossCorrelation(CorrelationMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_corr(self):

        dp1, dp2, rp1, rp2, dw1, dw2, rw1, rw2 = self.make_corr_data()

        tpcf = TwoPointCorrelationFunction(
            edges=self.theta_edges,

            # davis peebles has a weird, non symetric ordering 
            data_positions1=dp1 if self.estimator_type == 'landyszalay' else (
                dp2 if not self.autocorr else dp1
                ),
            data_positions2=dp2 if self.estimator_type == 'landyszalay' else dp1,

            randoms_positions1=rp1 if self.estimator_type == 'landyszalay' else None,
            randoms_positions2=rp2 if self.estimator_type == 'landyszalay' else rp1,

            data_weights1=dw1 if self.estimator_type == 'landyszalay' else dw2,
            data_weights2=dw2 if self.estimator_type == 'landyszalay' else dw1,

            randoms_weights1=rw1 if self.estimator_type == 'landyszalay' else None,
            randoms_weights2=rw2 if self.estimator_type == 'landyszalay' else rw1,

            # other settings n things
            nthreads=self.nproc,
            mode=self.corr_type,
            position_type=self.pos_type, # 'rd' for RA/Dec
            engine='corrfunc',
            estimator=self.estimator_type,
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
                    self.randoms1[self.ra_hsc_randoms_col], 
                    self.randoms1[self.dec_hsc_randoms_col]
                    ]
            if self.use_desi:
                rpsamp = [
                    self.randoms1[self.ra_desi_col], 
                    self.randoms1[self.dec_desi_col]
                    ]
        else:
            if self.double_desi:
                rpsamp = [
                    self.randoms1[self.ra_desi_col], 
                    self.randoms1[self.dec_desi_col]
                    ]
            else:
                rpsamp = [
                    self.randoms1[self.ra_desi_col],
                    self.randoms1[self.dec_desi_col]
                    ]
        
        if self.corr_type == 'rp':
            if self.double_desi:
                rpsamp.append(
                    ct.z2dist(
                        self.randoms2[self.z_desi_randoms_col]
                        )
                    )
            else:
                if self.autocorr:
                    rpsamp.append(
                        ct.z2dist(
                            self.randoms1[self.z_desi_randoms_col]
                            )
                        )
                else:
                    rpsamp.append(
                        ct.z2dist(
                            self.randoms2[self.z_hsc_randoms_col]
                            )
                        )
        #if self.data2 is not None and self.randoms2 is not None:
        #   self.logger.info(f'Data2 length {len(self.data2)} and randoms2 length {len(self.randoms2)}')

        self.logger.info('Subsampling randoms with KMeansSubsampler...')
        subsampler = KMeansSubsampler(
            mode='angular' if self.corr_type == 'theta' else '3d', 
            # The largest, most complete dataset we have is rp2
            # and no redshift sampling is needed for jackknife
            positions=rpsamp, 
            nsamples=nsamples,
            nside=nside if self.corr_type == 'theta' else None,
            random_state=seed, 
            position_type=self.pos_type,
            )
        labels = subsampler.label(rpsamp)

        self.subsampler = subsampler
        self.logger.info(f'Labels from {labels.min()} to {labels.max()}.')

    def run_corr(self):
        dp1, dp2, rp1, rp2, dw1, dw2, rw1, rw2 = self.make_corr_data()

        # subsampling with KMeans for jackknife
        ds1 = self.subsampler.label(dp1)
        rs1 = self.subsampler.label(rp1)
        if not self.autocorr:
            if self.estimator_type == 'landyszalay':
                rs2 = self.subsampler.label(rp2)
            else:
                rs2 = None
            ds2 = self.subsampler.label(dp2)
        else:
            rs2 = None
            ds2 = None

        tpcf = TwoPointCorrelationFunction(
            edges=self.theta_edges,

            # davis peebles has a weird, non symetric ordering 
            data_positions1=dp1 if self.estimator_type == 'landyszalay' else (
                dp2 if not self.autocorr else dp1
                ),
            data_positions2=dp2 if self.estimator_type == 'landyszalay' else dp1,

            randoms_positions1=rp1 if self.estimator_type == 'landyszalay' else None,
            randoms_positions2=rp2 if self.estimator_type == 'landyszalay' else rp1,

            data_weights1=dw1 if self.estimator_type == 'landyszalay' else dw2,
            data_weights2=dw2 if self.estimator_type == 'landyszalay' else dw1,

            randoms_weights1=rw1 if self.estimator_type == 'landyszalay' else None,
            randoms_weights2=rw2 if self.estimator_type == 'landyszalay' else rw1,

            data_samples1=ds1 if self.estimator_type == 'landyszalay' else ds2,
            data_samples2=ds2 if self.estimator_type == 'landyszalay' else ds1,

            randoms_samples1=rs1 if self.estimator_type == 'landyszalay' else None,
            randoms_samples2=rs2 if self.estimator_type == 'landyszalay' else rs1,

            nthreads=self.nproc,
            mode=self.corr_type,
            position_type=self.pos_type, # 'rd' for RA/Dec, 'rdd' for RA/Dec/Dist
            engine='corrfunc',
            estimator=self.estimator_type,
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
            sample_rate_1:int=1, 
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
            sample_rate_1=sample_rate_1,
            logger=logger
        )
        
    def run_corr(self):

        tpcf = TwoPointCorrelationFunction(
            edges=(np.linspace(0., 200., 51), np.linspace(-40, 40, 81)),
            data_positions1=[
                self.data1[self.ra_desi_col][self.z_bool_d1], 
                self.data1[self.dec_desi_col][self.z_bool_d1],
                ct.z2dist(self.data1[self.z_desi_col][self.z_bool_d1])
                ],
            randoms_positions1=[
                self.randoms1[self.ra_desi_col][self.z_bool_r1],
                self.randoms1[self.dec_desi_col][self.z_bool_r1],
                ct.z2dist(self.randoms1[self.z_desi_randoms_col][self.z_bool_r1]),
                ],
            data_weights1=self.data1[self.w_desi_col][self.z_bool_d1],
            randoms_weights1=self.randoms1[self.w_desi_col][self.z_bool_r1],
            nthreads=self.nproc,
            mode='rppi',
            position_type='rdd', 
            engine='corrfunc',
            estimator='landyszalay',
        )
        tpcf.save(self.outfile)


## Generic methods for each class
def _process_random_file(
        f, 
        ra_col, 
        dec_col, 
        main_weight_col, 
        weight_cols_to_operate, 
        z_col, 
        moc, 
        distance_col=None,
        operator=None
    ):
    
    try:
        with fio.FITS(str(f)) as rand:
            tbl = rand[1]

            data = _get_data_to_read(
                tbl=tbl, 
                ra_col=ra_col, 
                dec_col=dec_col, 
                main_weight_col=main_weight_col, 
                weight_cols_to_operate=weight_cols_to_operate, 
                z_col=z_col, 
                operator=operator
                )
            nrows = tbl.get_nrows()

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
        main_w_col=None, 
        weight_cols_to_operate=None,
        z_col=None, 
        moc=None, 
        operator=None,
        distance_col=None,
        num_processes=None
    ):
    """
    Multiprocessed random sampling with optional MOC filtering
    """
    assert operator is not None, f"Operator not provided for weight columns"

    if isinstance(random_files, (str, Path)):
        random_files = [random_files]
    assert len(random_files) > 0, f"No random files "
    
    if num_processes is None:
        num_processes = max(mp.cpu_count()-2, 1)
    
    tp = time.time()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            _process_random_file, [
            (f, ra_col, dec_col, main_w_col, weight_cols_to_operate, z_col, moc, distance_col, operator) 
            for f in random_files
        ])
    print(f"Processed {len(random_files)} random files in {time.time()-tp:.2f} seconds")

    randoms = [r for r in results if r is not None]
    
    if len(randoms) == 0:
        raise ValueError("No valid random data found")
    
    size = sum(len(r) for r in randoms)
    print(f"Collated {size} randoms (from {len(random_files)} files)")
    
    return np.concatenate(randoms)

def sample_file_on_moc(
        file, 
        ra_col, 
        dec_col, 
        main_weight_col, 
        weight_cols_to_operate=None, 
        z_col=None, 
        moc=None, 
        operator=None,
        distance_col=None
        ):
    '''
    Read a file and filter it using a MOC. Multiply the weights if needed.
    '''
    # this function is messy but it works (also kind of does multiple things at once)
    with fio.FITS(str(file)) as f:
        tbl = f[1]
        data = _get_data_to_read(
            tbl=tbl, 
            ra_col=ra_col, 
            dec_col=dec_col, 
            main_weight_col=main_weight_col, 
            weight_cols_to_operate=weight_cols_to_operate, 
            z_col=z_col, 
            operator=operator,
            distance_col=distance_col
        )

        if moc is not None:
            print(f"Filtering {tbl.get_nrows()} {len(data)} rows in {Path(file).stem} using MOC")
            coords = SkyCoord(data[ra_col] * u.deg, data[dec_col] * u.deg, frame='icrs')
            mask = moc.contains_skycoords(coords)
            del coords
            data = data[mask]
            print(f"Filtered {np.sum(mask)} rows in {Path(file).stem} using MOC")
            del mask

    return np.array(data)

def _get_data_to_read(
        tbl, 
        ra_col, 
        dec_col, 
        main_weight_col, 
        weight_cols_to_operate, 
        z_col, 
        operator=None, 
        distance_col=None
    ):
    # if HSC + zbin we can have an edge case where the z_col is two values, so flatten and unpack first
    requested_cols = [ra_col, dec_col, main_weight_col]
    if isinstance(z_col, list):
        if len(z_col) == 2:
            # we want both knowledge on zbin and z
            requested_cols.append(z_col[0])
            requested_cols.append(z_col[1])
        else:
            raise ValueError(f"z_col should be a list of length 2, got {len(z_col)}")
    else:
        # if not a list we're good to go
        requested_cols.append(z_col)
    # None values are not valid columns so we exclude them
    base_cols = [col for col in requested_cols if col is not None]
    cols_to_read = base_cols.copy()

    if len(base_cols) == 0:
        raise ValueError(f"No columns to read in {tbl}")
    # we unpack the weights we want to operate on here...
    if main_weight_col in base_cols:
        if weight_cols_to_operate is not None:
            cols_to_read.remove(main_weight_col)
            cols_to_read += weight_cols_to_operate

    assert all(c in tbl.get_colnames() for c in cols_to_read), f"Columns {cols_to_read} not in {tbl}"
    data = Table(tbl.read(columns=cols_to_read))

    if main_weight_col is not None:
        if weight_cols_to_operate is not None:
            if operator is None:
                raise ValueError("Operator not provided for weight columns")
            if operator in ['*', 'multiply', 'times', 'product']:
                w_col = np.ones_like(data[ra_col])
                for col in weight_cols_to_operate:
                    w_col *= data[col]
                    logging.info(
                        f"Multiplying {col} to {main_weight_col} : {data[col][:3]} * {w_col[:3]}"
                        )
            elif operator in ['+', 'add', 'plus', 'sum']:
                w_col = np.zeros_like(data[ra_col])
                for col in weight_cols_to_operate:
                    w_col += data[col]
        else:
            w_col = data[main_weight_col]
        data[main_weight_col] = w_col

    if distance_col is not None:
        if z_col is not None:
            data[distance_col] = ct.z2dist(data[z_col])
        else:
            raise ValueError("Distance column provided but no redshift column")

    logging.info(
        (
            f"Weight column {main_weight_col} set to {data[main_weight_col][:3]}\n"
            f" after applying operator {operator} to "
            f"columns {weight_cols_to_operate} " 
            if main_weight_col is not None else ""
            ) +
        (
            f"Distance column {distance_col} set to {data[distance_col][:3]}" 
            if distance_col is not None else ""
            )
        )
    if weight_cols_to_operate is not None:
        logging.info(f'{data[:3]}, {Table(tbl.read(columns=weight_cols_to_operate))[:3]}')
    
    return data
    
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
    
def get_target_couple(tgt1, tgt2=None):
    avb = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY', 'HSC']
    assert not((tgt1 is None) and (tgt2 is None)), 'tgt1 and tgt2 cannot be None simultaneously'
    if tgt2 is None and tgt1 is not None:
        tgt2 = tgt1
    elif tgt1 is None and tgt2 is not None:
        tgt1 = tgt2

    if isinstance(tgt1, str):
        tgt1 = [tgt1]
    if isinstance(tgt2, str):
        tgt2 = [tgt2]

    assert len(tgt1) == len(tgt2), 'tgt1 and tgt2 must have the same length'
    assert all(t1 in avb for t1 in tgt1), f'Unknown targets {tgt1}'
    assert all(t2 in avb for t2 in tgt2), f'Unknown targets {tgt2}'
    
    return tgt1, tgt2