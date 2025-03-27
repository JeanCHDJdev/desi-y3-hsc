import time 
import matplotlib.pyplot as plt
import os
import fitsio as fio
import numpy as np
import logging
import psutil
import memory_profiler

from mocpy import MOC
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from argparse import ArgumentParser

from pycorr import (
    TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, utils, setup_logging
)

def setup_crosscorr_logging(log_file='logs/output', log_level=logging.INFO):
    """
    Set up logging with both console and file output
    
    Args:
        log_file (str): Path to the log file
        log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_file += f'_{time.strftime("%Y%m%d_%H%M%S")}.log'
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
            memory_usage = process.memory_info().rss / 1e6  # Convert to MB
            logger.info(f"Memory Usage: {memory_usage:.2f} MB")
        except Exception as e:
            logger.error(f"Could not log memory usage: {e}")

    logger.memory_usage = log_memory_usage

    return logger

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '-o',
        '--output_dir', 
        type=str, 
        default='out/',
        help='Path to the output file (storing the cross-correlation). '
        'Default is out/'
        )
    parser.add_argument(
        '-t',
        '--tgt',
        type=str,
        default=None,
        help='Target to cross-correlate with HSC. '
        'Default is all targets'
        )
    parser.add_argument(
        '-rd',
        '--sample_rate_desi', 
        type=int,
        default=1,
        help='Sampling rate for the randoms of DESI. '
        'Defaults to 1.'
    )
    parser.add_argument(
        '-rh',
        '--sample_rate_hsc', 
        type=int,
        default=1,
        help='Sampling rate for the randoms of hsc. '
        'Defaults to 1.'
    )
    parser.add_argument(
        '-l',
        '--log', 
        type=str,
        default=None,
        help='Log file to store run settings'
    )
    parser.add_argument(
        '-w',
        '--nproc', 
        type=int, 
        help='Number of threads to use for cross-correlation. '
        'Default is all-2.'
        )
    
    return parser.parse_args()

class CrossCorrelation():
    def fetch_desi_files(self, randoms=False):
        try:
            root = Path(
                '/global/cfs/projectdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v1.1/nonKP'
                )
            path = f'{self.tgt}{f"_{self.cap}_*" if self.cap else ("_[0-9]*_" if randoms else "_")}clustering{".ran" if randoms else ".dat"}.fits'
            files = list(root.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found for path: {path}")
            return files
        except PermissionError:
            logging.error("Permission denied accessing catalog files")
            raise
    
    def fetch_hsc_files(self, randoms=False):
        try:
            if randoms:
                root = Path(
                    '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/rand'
                    )
                return root.glob('hscran*.fits')
            else:
                return [Path(
                    '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat.fits'
                    )]
        except PermissionError:
            logging.error(f"Permission denied accessing HSC files and randoms = {randoms}")
            raise
        except FileNotFoundError:
            logging.error(f"HSC catalog file not found and randoms = {randoms}") 
            raise

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
            cap:str=None,  
            logger:logging.Logger=None
            ):
        ti = time.time()

        assert logger is not None, 'Logger not provided'
        self.logger = logger
        
        self.ra_hsc_col = 'RA'
        self.dec_hsc_col = 'Dec'
        self.ra_hsc_randoms_col = 'ra'
        self.dec_hsc_randoms_col = 'dec'
        self.w_hsc_col = 'weight'
        self.z_hsc_col = 'dnnz_photoz_best'

        self.ra_desi_col = 'RA'
        self.dec_desi_col = 'DEC'
        self.w_desi_col = 'WEIGHT'
        self.z_desi_col = 'Z'

        avb_cap = ['NGC', 'SGC']
        if cap:
            assert cap in avb_cap, f'{cap} not in {avb_cap}'
        self.cap = cap

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
        fs['catalog1'] = self.fetch_desi_files(randoms=False)
        fs['randoms1'] = self.fetch_desi_files(randoms=True)
        fs['catalog2'] = self.fetch_hsc_files(randoms=False)
        fs['randoms2'] = self.fetch_hsc_files(randoms=True)
        
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
            list(self.fs['randoms1']),
            ra_col=self.ra_desi_col,
            dec_col=self.dec_desi_col,
            w_col=self.w_desi_col,
            z_col=self.z_desi_col,
            moc=self.moc,
            sample_rate=self.sample_rate_desi
            )
        logger.info(f'Collated HSC randoms in {time.time()-trd:.2f} seconds')
        trh = time.time()
        self.randoms2 = sample_randoms_on_moc(
            list(self.fs['randoms2']), 
            ra_col='ra',
            dec_col='dec',
            w_col=None,
            z_col=None,
            moc=self.moc,
            sample_rate=self.sample_rate_hsc
            )
        logger.info(f'Collated HSC randoms in {time.time()-trh:.2f} seconds')

        ti = time.time()
        self.data1 = sample_file_on_moc(
            self.fs['catalog1'][0], 
            self.ra_desi_col, 
            self.dec_desi_col, 
            self.w_desi_col, 
            self.z_desi_col,
            moc=self.moc
            )
        self.data2 = sample_file_on_moc(
            self.fs['catalog2'][0], 
            self.ra_hsc_col, 
            self.dec_hsc_col, 
            self.w_hsc_col,
            self.z_hsc_col, 
            moc=self.moc
            )
        self.logger.info(f'Read catalogs {time.time()-ti:.2f} seconds')

        # Setup redshift masks
        self.zmask_data1 = np.digitize(self.data1[self.z_desi_col], bin_redshift1)
        self.zmask_data2 = np.digitize(self.data2[self.z_hsc_col], bin_redshift2)
        self.zmask_randoms1 = np.digitize(self.randoms1[self.z_desi_col], bin_redshift1)

    def run(self, bin_index1, bin_index2, moc_index):

        # Setup redshift masking
        z_mask_d1 = (self.zmask_data1 == bin_index1)
        z_mask_d2 = (self.zmask_data2 == bin_index2)
        # DESI randoms need to be redshift masked 
        z_mask_r1 = (self.zmask_randoms1 == bin_index1)

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
        outfile = Path(
            self.output_dir, 
            f'{self.tgt}_{self.cap if self.cap else ""}_b1x{bin_index1}_b2x{bin_index2}_moc{moc_index}.npy'
            )
        tpcf.save(outfile)

def sample_randoms_on_moc(random_files, ra_col, dec_col, w_col, z_col, moc=None, sample_rate=1):

    assert len(random_files) > 0, f"No random files "
    randoms = []
    size = 0
    if z_col is None or w_col is None:
        cols = [ra_col, dec_col]
    else:
        cols = [ra_col, dec_col, w_col, z_col]

    for f in random_files:
        with fio.FITS(str(f)) as rand:
            data = rand[1].read(
                columns=cols, 
                rows=np.arange(0, rand[1].get_nrows(), sample_rate)
                )
            if moc is not None:
                coords = SkyCoord(data[ra_col]*u.deg, data[dec_col]*u.deg, frame='icrs')
                mask = moc.contains_skycoords(coords)
                data = data[np.flatnonzero(mask)]
            if len(data) == 0:
                continue
            size += len(data)
            randoms.append(data)  

    print(f"Collated {size} randoms (from {len(random_files)} files)")
    if z_col is None:
        dtype = [(ra_col, 'f8'), (dec_col, 'f8')]
    else:
        dtype = [(ra_col, 'f8'), (dec_col, 'f8'), (w_col, 'f8'), (z_col, 'f8')]

    return np.concatenate(randoms).astype(dtype)

def sample_file_on_moc(file, ra_col, dec_col, weight_col, z_col, moc=None):

    with fio.FITS(str(file)) as f:
        if moc is not None:
            data = f[1].read(columns=[ra_col, dec_col])
            coords = SkyCoord(data[ra_col] * u.deg, data[dec_col] * u.deg, frame='icrs')
            mask = moc.contains_skycoords(coords)
            rows = np.flatnonzero(mask)
        else:
            rows = None

        data = f[1].read(columns=[ra_col, dec_col, weight_col, z_col], rows=rows)

    dtype = [(ra_col, 'f8'), (dec_col, 'f8'), (weight_col, 'f8'), (z_col, 'f8')]
    return np.array(data).astype(dtype)

def main():
    args = parse_args()

    tgt = args.tgt
    output_dir = args.output_dir
    sample_rate_desi = args.sample_rate_desi
    sample_rate_hsc = args.sample_rate_hsc
    nproc = args.nproc
    log = args.log

    if log is None:
        logger = setup_crosscorr_logging()
    else:
        logger = setup_crosscorr_logging(log_file=log)
    setup_logging()

    if nproc is None:
        nproc = max(os.cpu_count()-2, 1)

    # for now, we will only consider the following targets, could do more later
    avb_tgt = ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    if tgt:
        assert tgt in avb_tgt, f'{tgt} not in {avb_tgt}'
        if isinstance(tgt, str):
            tgts = [tgt]
    else:
        tgts = avb_tgt
    
    #bin_distances = np.linspace(0., 200., 51)
    bin_distances = np.linspace(0.03, 3, 51)

    bins_bgs = np.arange(0, 0.7, 0.1) # 0 < z < 0.6
    bins_lrg = np.arange(0.3, 1.1, 0.1) # 0.4 < z < 1
    bins_elg = np.arange(0.6, 1.7, 0.1) # 0.6 < z < 1.6
    bins_qso = np.arange(0.9, 2.2, 0.1) # 0.9 < z < 2.1

    bins_hsc = np.linspace(0.3, 1.2, 4) 

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    np.savetxt(Path(output_dir, 'bin_distances.txt'), bin_distances)
    np.savetxt(Path(output_dir, 'bins_bgs.txt'), bins_bgs)
    np.savetxt(Path(output_dir, 'bins_lrg.txt'), bins_lrg)
    np.savetxt(Path(output_dir, 'bins_elg.txt'), bins_elg)
    np.savetxt(Path(output_dir, 'bins_qso.txt'), bins_qso)
    np.savetxt(Path(output_dir, 'bins_hsc.txt'), bins_hsc)

    bins_redshift = {
        'BGS_ANY': bins_bgs,
        'LRG': bins_lrg,
        'ELGnotqso': bins_elg,
        'QSO': bins_qso,
        'HSC': bins_hsc,
    }

    logger.info(
        f'Sample rate on DESI randoms: {sample_rate_desi} and on HSC randoms: {sample_rate_hsc}\n'
        )
    logger.info(f'Number of threads: {nproc}\n')
    logger.info(f'Output directory: {output_dir}\n')
    logger.info(f'Bins redshift :{bins_redshift}\n')
    logger.info(f'Fiducial bin distances: {bin_distances}\n')
    logger.info('\n')

    moc_list = list(Path(
        '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/mocs/'
        ).glob(
            'hsc_moc*.fits'
            ))

    for m, mocf in enumerate(moc_list):
        logger.info(f'Running for MOC : {mocf}\n')
        moc = MOC.from_fits(mocf)

        for t in tgts:
            bin1 = bins_redshift[tgt]
            bin2 = bins_redshift['HSC']

            logger.memory_usage()
            cc = CrossCorrelation(
                t, 
                moc, 
                output_dir,
                bin_distances=bin_distances,
                bin_redshift1=bin1,
                bin_redshift2=bin2, 
                nproc=nproc, 
                sample_rate_desi=sample_rate_desi,
                sample_rate_hsc=sample_rate_hsc, 
                cap=None,
                logger=logger,
            )
            logger.memory_usage()

            logger.info(f'Running for {tgt}, bin1 {bin1}, bin2 {bin2}, moc {m}\n' + "=" * 80)

            for b in range(len(bin1)-1):
                for c in range(len(bin2)-1):
                    tbc = time.time()
                    cc.run(b, c, m)
                    txt = f'Finished {tgt}, {b} (desi) : {bin1[b]}, {c} (hsc) : {bin2[c]} in {time.time()-tbc:.2f}s\n'
                    logger.info(txt)

if __name__ == '__main__':
    main()