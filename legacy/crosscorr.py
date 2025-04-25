import time 
import matplotlib.pyplot as plt
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

from pycorr import (
    TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, utils, setup_logging
)

import corrutils as cu

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '-o',
        '--output_dir', 
        type=str, 
        default='crosscorr/new/',
        help='Path to the output file (storing the cross-correlation). '
        'Default is crosscorr/new/'
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
        'Default is `os.cpu_count()-2`.'
        )
    parser.add_argument(
        '-p',
        '--patches', 
        type=int, 
        nargs='+',
        default=None,
        help='Patches of the sky to cross-correlate. '
        )
    
    return parser.parse_args()

def main():
    args = parse_args()

    tgt = args.tgt
    output_dir = args.output_dir
    sample_rate_desi = args.sample_rate_desi
    sample_rate_hsc = args.sample_rate_hsc
    nproc = args.nproc
    log = args.log
    patches = args.patches
    
    moc_list = cu.CorrelationMeta.moc_list
    if patches is not None:
        patches = np.array([int(p) for p in patches])
        assert len(patches) > 0, 'Patches should be a list of integers'
        assert np.all(
            (0 < patches) & (patches < len())
            ), (
                f'Patches should be less than {len(moc_list)} '
                'and greater than 0'
                )
    else:
        patches = np.arange(len(moc_list))

    if log is None:
        logger = cu.setup_crosscorr_logging(log_file=str(Path(output_dir, 'autolog')))
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

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not Path(output_dir, 'bins').exists():
        Path(output_dir, 'bins').mkdir(parents=True, exist_ok=True)

    print('=' * 80)

    bins_redshift = {
        'BGS_ANY': cu.bins_bgs,
        'LRG': cu.bins_lrg,
        'ELGnotqso': cu.bins_elg,
        'QSO': cu.bins_qso,
        'HSC': cu.bins_hsc,
        'distances': cu.bin_distances,
    }
    for k, v in bins_redshift.items():
        np.savetxt(Path(output_dir, 'bins', f'bins_{k}.txt'), v)

    logger.info(
        f'Sample rate on DESI randoms: {sample_rate_desi} and on HSC randoms: {sample_rate_hsc}\n'
        )
    logger.info(f'Number of threads: {nproc}\n')
    logger.info(f'Output directory: {output_dir}\n')
    logger.info(f'Bins redshift :{bins_redshift}\n')
    logger.info(f'Fiducial bin distances: {cu.bin_distances}\n')

    moc_list = cu.moc_list

    for m in range(len(moc_list)):
        mocf = moc_list[m]
        moc = MOC.from_fits(mocf)
        logger.info(f'MOC {m} : {mocf} ...\n')

        for t in tgts:
            bin1 = bins_redshift[t]
            bin2 = bins_redshift['HSC']

            logger.memory_usage()
            cc = cu.CrossCorrelation(
                t, 
                moc, 
                output_dir,
                bin_distances=cu.bin_distances,
                bin_redshift1=bin1,
                bin_redshift2=bin2, 
                nproc=nproc, 
                sample_rate_desi=sample_rate_desi,
                sample_rate_hsc=sample_rate_hsc, 
                logger=logger,
            )
            logger.memory_usage()

            logger.info(f'Running for {tgt}, bin1 {bin1}, bin2 {bin2}, moc {m}\n' + "=" * 80)

            for b in range(1, len(bin1)):
                for c in range(1, len(bin2)):
                    tbc = time.time()
                    cc.run(b, c, m)
                    txt = f'Finished {t}, {b} (desi) : {bin1[b-1]}-{bin1[b]}, {c} (hsc) : {bin2[c-1]}-{bin2[c]} in {time.time()-tbc:.2f}s\n'
                    logger.info(txt)

if __name__ == '__main__':
    print('Starting cross-correlation script ...')
    ti = time.time()
    main()
    print(f'Finished cross-correlation script in {time.time()-ti:.2f}s')