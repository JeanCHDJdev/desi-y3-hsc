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

from abc import ABC, abstractmethod
from mocpy import MOC
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord

from src.statistics import cosmotools as ct
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler

class CorrelationMeta(ABC):

    # Attributes that each cross corr code needs to know about
    ra_hsc_col = 'ra'
    dec_hsc_col = 'dec'
    ra_hsc_randoms_col = 'ra'
    dec_hsc_randoms_col = 'dec'
    w_hsc_col = 'weight'
    z_hsc_col = 'dnnz_photoz_best'
    z_hsc_randoms_col = 'redshift'

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
    bin_distances = np.linspace(0.001, 2, 41)
    #np.logspace(
    #    np.log(2), 
    #    np.log(0.001), 
    #    51,
    #    base=np.e
    #)

    bins_bgs = np.arange(0, 0.6, 0.1) # 0 < z < 0.6
    bins_lrg = np.arange(0.4, 1.2, 0.1) # 0.4 < z < 1
    bins_elg = np.arange(0.8, 1.7, 0.1) # 0.6 < z < 1.6 => 0.8 < z < 1.6 in redshift distribution
    bins_qso = np.arange(0.8, 3.4, 0.3) # 0.9 < z < 2.1

    bins_hsc = np.arange(0.3, 1.8, 0.3) # 0.3 < z <= 1.5 

    bins_redshift = {
        'LRG': bins_lrg,
        'ELGnotqso': bins_elg,
        'QSO': bins_qso,
        'BGS_ANY': bins_bgs,
        'HSC': bins_hsc,
        'distances': bin_distances
    }

    @staticmethod
    def save_bins(root):
        """
        Save the bins to a file
        """
        bin_dir = Path(root, 'bins')
        if not bin_dir.exists():
            bin_dir.mkdir(parents=True)
        np.savez(
            Path(bin_dir, 'bins_redshift.npz'),
            **{k: v for k, v in CorrelationMeta.bins_redshift.items()}
        )

    def __init__(
            self, 
            logger : logging.Logger, 
            moc : MOC, 
            tgt1=None, 
            tgt2=None,
            output_dir=None, 
            sims=False,
            weight_type='nonKP',
            sample_rate_desi=1,
            sample_rate_hsc=1, 
            nproc=None
            ):
        assert logger is not None, 'Logger not provided'
        self.logger = logger
        
        # rename the class attributes if using simulations bc not the same class names
        self.sims = sims
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
        if tgt1 is None or tgt2 is None:
            if tgt1 is None and tgt2 is None:
                raise ValueError('No target provided')
            else:
                if tgt1 is None:
                    tgt1 = tgt2
                else:
                    tgt2 = tgt1

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
        bin_redshift1 = self.bins_redshift[tgt1]
        bin_redshift2 = self.bins_redshift[tgt2]

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
            f'weight_type={weight_type}, sims={sims}, '
            )
        if self.use_desi:
            fs['catalog1'] = fetch_desi_files(tgt1, randoms=False, weight_type=weight_type, sims=sims)
            fs['randoms1'] = fetch_desi_files(tgt1, randoms=True, weight_type=weight_type, sims=sims)

        if self.use_hsc:
            fs['catalog2'] = fetch_hsc_files(randoms=False, sims=sims, include_dud=False)
            fs['randoms2'] = fetch_hsc_files(randoms=True, sims=sims, include_dud=False)
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
                weight_col=self.w_desi_col if not sims else None, 
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
                z_col=self.z_hsc_col, 
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

        if self.use_hsc and self.autocorr:
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
        if self.autocorr:
            desccorr = self.tgt1
        else:
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
            # in the case of crosscorr, we wapply a mask to the randoms
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
            edges=self.bin_distances,
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

        rp2 = [
            self.randoms2[self.ra_hsc_randoms_col],
            self.randoms2[self.dec_hsc_randoms_col]
            ]
        self.logger.info(f'Data2 length {len(self.data2)} and randoms2 length {len(self.randoms2)}')
        self.logger.info('Subsampling data2 with KMeansSubsampler...')
        subsampler = KMeansSubsampler(
            mode='angular', 
            # The largest, most complete dataset we have is rp2
            # and no redshift sampling is needed for jackknife
            positions=rp2, 
            nsamples=nsamples,
            nside=nside,
            random_state=seed, 
            position_type='rd'
            )
        labels = subsampler.label(rp2)
        self.subsampler = subsampler
        self.subsampler.log_info(f'Labels from {labels.min()} to {labels.max()}.')

    def run_corr(self):
        
        self.logger.info(
            f'N data1: {np.sum(self.z_mask_d1)}' + 
            f', N randoms1: {np.sum(self.z_mask_r1)}'
            )
        self.logger.info(
            f'N data2: {np.sum(self.z_mask_d2)}' + 
            f', N randoms2: {len(self.randoms2)}'
            )

        dp1 = [
            self.data1[self.ra_desi_col][self.z_mask_d1], 
            self.data1[self.dec_desi_col][self.z_mask_d1]
            ]
        dp2 = [
            self.data2[self.ra_hsc_col][self.z_mask_d2], 
            self.data2[self.dec_hsc_col][self.z_mask_d2]
            ]
        rp1 = [
            self.randoms1[self.ra_desi_col][self.z_mask_r1],
            self.randoms1[self.dec_desi_col][self.z_mask_r1]
            ]
        rp2 = [
            self.randoms2[self.ra_hsc_randoms_col],
            self.randoms2[self.dec_hsc_randoms_col]
            ]
        
        if self.sims:
            # no weights for simulations
            dw1 = None
            dw2 = None
            rw1 = None
            rw2 = None
        else :
            dw1 = self.data1[self.w_desi_col][self.z_mask_d1]
            dw2 = self.data2[self.w_hsc_col][self.z_mask_d2]

            rw1 = self.randoms1[self.w_desi_col][self.z_mask_r1]
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
            position_type='rd',
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
            sims : bool,
            pip : bool,
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

class HSCAutoCorrelation(CorrelationMeta):
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
        
        super().__init__(
            moc=moc,
            output_dir=output_dir,
            bin_distances=bin_distances,
            bin_redshift1=bin_redshift1,
            nproc=nproc,
            sample_rate_hsc=sample_rate_hsc,
            logger=logger
        )
    def run_corr(self):
        tpcf = TwoPointCorrelationFunction(
            edges=self.bin_distances,
            data_positions1=[
                self.data1[self.ra_hsc_col][self.z_mask_d1], 
                self.data1[self.dec_hsc_col][self.z_mask_d1]
                ],
            data_positions2=None,
            randoms_positions1=[
                self.randoms1[self.ra_hsc_randoms_col],
                self.randoms1[self.dec_hsc_randoms_col]
                ],
            randoms_positions2=None,
            data_weights1=self.data1[self.w_hsc_col][self.z_mask_d1],
            data_weights2=None,
            randoms_weights1=None,
            randoms_weights2=None,
            nthreads=self.nproc,
            mode='theta',
            position_type='rd', # 'rd' for RA/Dec
            engine='corrfunc',
            estimator='landyszalay',
        )
        tpcf.save(self.outfile)


## Generic methods for each class
def process_random_file(f, ra_col, dec_col, w_col, z_col, moc):
    if z_col is None:
        cols = [ra_col, dec_col, w_col]
    elif w_col is None:
        cols = [ra_col, dec_col, z_col]
    elif w_col is not None and z_col is None:
        cols = [ra_col, dec_col, z_col]
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

def fetch_desi_files(tgt, randoms=False, weight_type='nonKP', sims=False, sims_version=2):
    try:
        if sims:
            sims_root = '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/sims/'
            if randoms:
                return Path(
                    sims_root,
                    'randoms',
                    f'{tgt}_ran_hsc_zcorr.fits'
                )
            return Path(
                sims_root,
                f'v{sims_version}',
                f'desi_targets_sim_{tgt}_v{sims_version}.fits'
                )
        else:
            root = Path(
                '/global/cfs/projectdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v1.1/'
                )
            if weight_type == 'PIP':
                root = Path(root, 'PIP')
                path = f'{tgt}{"_[0-9]*_" if randoms else "_"}clustering{".ran" if randoms else ".dat"}.fits'
            elif weight_type == 'nonKP':
                root = Path(root, 'nonKP')
                path = f'{tgt}{"_[0-9]*_" if randoms else "_"}clustering{".ran" if randoms else ".dat"}.fits'
            elif weight_type == 'base':
                root = Path(root)
                path = f'{tgt}{"_[0-9]*_" if randoms else "_"}_full_HPmapcut{".ran" if randoms else ".dat"}.fits'
            else:
                raise ValueError(f"Unknown weight type {weight_type}")

            files = list(root.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found for path: {path}")
            if len(files) == 1:
                files = files[0]
            return files
    except PermissionError:
        logging.error(f"Permission denied accessing DESI files and randoms = {randoms}")
        raise

def fetch_hsc_files(randoms=False, include_dud=False, sims=False, sims_version=2):
    try:
        if sims:
            sims_root = '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/sims/'
            if randoms:
                return Path(
                    sims_root,
                    'randoms',
                    f'HSC_randoms_zcorr_v{sims_version}.fits'
                )
            return Path(
                sims_root,
                f'v{sims_version}',
                f'hscy3_sim_v{sims_version}.fits'
                )
        elif randoms:
            #WARNING : this path root currently does not contain D/UD randoms as they
            #were deemed unnecessary for the clustering analysis
            root = Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/randoms'
                )
            return list(
                root.glob(f'edge_sc_cr_hscr{"*" if include_dud else "[0-9]"}.fits')
                )
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

def figure_out_class(tgt1, tgt2=None, jackknife=False):
    desi_avb = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    hsc_avb = ['HSC']
    if tgt1 is None and tgt2 is None:
        raise ValueError('No target provided')
    if tgt2 is None:
        tgt2 = tgt1
    if tgt1 is None:
        tgt1 = tgt2

    if tgt1 != tgt2:
        # cross-correlation case
        if tgt1 in desi_avb:
            if tgt2 in hsc_avb:
                if jackknife:
                    return JackknifeCrossCorrelation
                else:
                    return CrossCorrelation
    else:
        # autocorrelation case
        if tgt1 in hsc_avb and tgt2 in hsc_avb:
            return HSCAutoCorrelation
        if tgt2 in desi_avb and tgt1 in desi_avb:
            return DESIAutoCorrelation
    raise ValueError('Unknown target combination')

class CorrFileReader():
    '''
    Utility class to grab correctly formatted file names for the cross-correlation
    analysis. Provide a ROOT and the directory has to be with the expected shape.
    '''
    def __init__(self, ROOT):
        self.ROOT = Path(ROOT)
        assert self.ROOT.exists(), f"Path {self.ROOT} does not exist"

    def get_file(self, b1, b2, tgt1, tgt2, moc):
        """
        Get the file name for given redshift bins and MOC.
        """
        DIR = self.ROOT / f'{tgt1}x{tgt2}'
        return f'{DIR}/{tgt1}x{tgt2}_b1x{b1}_b2x{b2}_moc{moc}.npy'
    
    def get_auto_file(self, b1, tgt, moc):
        """
        Get the file name for given redshift bins and MOC.
        """
        return self.get_file(b1, b1, tgt, tgt, moc)

    def get_bins(self, name):
        bins = np.load(f'{self.ROOT}/bins/bins_redshift.npz')
        if name not in bins:
            raise ValueError(f"Unknown bin name {name}. Available bins are {bins.files}")
        else:
            return bins[name]
    
    def get_cov_results(self, tgt1, tgt2='HSC'):
        """
        Get the covariance result for given redshift bins and MOC.
        """
        covdir = Path(self.ROOT, f'{tgt1}x{tgt2}', 'cov')
        if not covdir.exists():
            raise FileNotFoundError(f"Covariance directory {covdir} does not exist")
        else:
            files = covdir.glob(f'*.npy')
            return list(files)
        
    def get_cov_file(self, b1, b2, tgt1, tgt2='HSC', moc=0):
        """
        Get the covariance file name for given redshift bins and MOC.
        """
        covdir = Path(self.ROOT, f'{tgt1}x{tgt2}', 'cov')
        file = covdir / f'{tgt1}x{tgt2}_b1x{b1}_b2x{b2}_moc{moc}.npy'
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist")
        else:
            return file
            
        