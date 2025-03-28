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
from argparse import ArgumentParser

import corrutils as cu

from pycorr import (
    TwoPointCorrelationFunction, setup_logging
)

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

def main():
    args = parse_args()

    tgt = args.tgt
    output_dir = args.output_dir
    sample_rate_desi = args.sample_rate_desi
    nproc = args.nproc
    log = args.log

    if log is None:
        logger = cu.setup_crosscorr_logging()
    else:
        logger = cu.setup_crosscorr_logging(log_file=log)
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
    bin_distances = np.linspace(0.01, 3, 31) #np.logspace(np.log10(0.001), np.log10(3), 71)

    bins_bgs = np.arange(0, 0.6, 0.1) # 0 < z < 0.6
    bins_lrg = np.arange(0.4, 1.2, 0.1) # 0.4 < z < 1
    bins_elg = np.arange(0.8, 1.7, 0.1) # 0.6 < z < 1.6 => 0.8 < z < 1.6 in redshift distribution
    bins_qso = np.arange(0.8, 3.4, 0.1) # 0.9 < z < 2.1

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not Path(output_dir, 'bins').exists():
        Path(output_dir, 'bins').mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    np.savetxt(Path(output_dir, 'bins', 'bin_distances.txt'), bin_distances)
    np.savetxt(Path(output_dir, 'bins', 'bins_bgs.txt'), bins_bgs)
    np.savetxt(Path(output_dir, 'bins', 'bins_lrg.txt'), bins_lrg)
    np.savetxt(Path(output_dir, 'bins', 'bins_elg.txt'), bins_elg)
    np.savetxt(Path(output_dir, 'bins', 'bins_qso.txt'), bins_qso)

    bins_redshift = {
        'BGS_ANY': bins_bgs,
        'LRG': bins_lrg,
        'ELGnotqso': bins_elg,
        'QSO': bins_qso,
    }

    logger.info(
        f'Sample rate on DESI randoms: {sample_rate_desi}\n'
        )
    logger.info(f'Number of threads: {nproc}\n')
    logger.info(f'Output directory: {output_dir}\n')
    logger.info(f'Bins redshift :{bins_redshift}\n')
    logger.info(f'Fiducial bin distances: {bin_distances}\n')

    moc_list = [
        Path(
            '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/mocs/', 
            f'hsc_moc{i+1}.fits'
        )
        for i in range(0, 4)
        ]

    for m, mocf in enumerate(moc_list):
        moc = MOC.from_fits(mocf)

        for t in tgts:
            bin1 = bins_redshift[tgt]

            logger.memory_usage()
            cc = cu.DESIAutoCorrelation(
                t, 
                moc, 
                output_dir,
                bin_distances=bin_distances,
                bin_redshift1=bin1,
                nproc=nproc, 
                sample_rate_desi=sample_rate_desi, 
                logger=logger,
            )
            logger.memory_usage()

            for b in range(1, len(bin1)):
                tbc = time.time()
                cc.run(b, m)
                txt = f'Finished {tgt}, {b} (desi) : {bin1[b]} in {time.time()-tbc:.2f}s\n'
                logger.info(txt)

if __name__ == '__main__':
    print('Starting cross-correlation script ...')
    ti = time.time()
    main()
    print(f'Finished cross-correlation script in {time.time()-ti:.2f}s')