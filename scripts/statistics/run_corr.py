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
    TwoPointCorrelationFunction, setup_logging
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
        '-j',
        '--jackknife',
        action='store_true',
        default=False,
        help='Wether to perform cross correlation with jackknife estimates. '
    )
    parser.add_argument(
        '-ns',
        '--nsamples',
        type=int,
        default=64,
        help='Number of jackknife samples. '
        'Default is 64. '
    )
    parser.add_argument(
        '-r',
        '--resolution',
        type=int,
        default=256,
        help='Healpix pixel resolution for jackknife subsampling. '
    )
    parser.add_argument(
        '-t1',
        '--tgt1',
        type=str,
        nargs='+',
        default=None,
        choices=['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY', 'HSC'],
        help='Target(s) 1. '
        'Default is None. If None, will use all DESI targets.'
        )
    parser.add_argument(
        '-t2',
        '--tgt2',
        type=str,
        choices=['HSC'],
        default=None,
        help='Target 2. '
        'Default is None. Only current option is HSC.'
        )
    parser.add_argument(
        '-rd',
        '--sample_rate_desi', 
        type=int,
        default=1,
        help='Sampling rate for DESI randoms. '
        'Defaults to 1.'
    )
    parser.add_argument(
        '-rh',
        '--sample_rate_hsc', 
        type=int,
        default=1,
        help='Sampling rate for HSC randoms catalog. '
        'Defaults to 1.'
    )
    parser.add_argument(
        '-s',
        '--sims',
        type=bool,
        default=False,
        help='Whether to use the simulated data. '
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
        choices=range(1, 4),
        help='Patches of the sky to cross-correlate. '
        )
    
    return parser.parse_args()

def main():
    args = parse_args()

    tgt1 = args.tgt1
    tgt2 = args.tgt2
    jackknife = args.jackknife
    
    sims = args.sims
    sample_rate_desi = args.sample_rate_desi
    sample_rate_hsc = args.sample_rate_hsc

    output_dir = args.output_dir
    nproc = args.nproc
    log = args.log
    patches = args.patches
    
    print(f'Running cross-correlation for the following targets: {tgt1}x{tgt2}')

    if patches is not None:
        patches = np.array([int(p) for p in patches])
        assert len(patches) > 0, 'Patches should be a list of integers'
        assert np.all(
            (0 < patches) & (patches < len(cu.CorrelationMeta.moc_list))
            ), (
                f'Patches should be less than {len(cu.CorrelationMeta.moc_list)} '
                'and greater than 0'
                )
    else:
        patches = np.arange(len(cu.CorrelationMeta.moc_list))

    ## logging infrastructure
    if log is None:
        logger = cu.setup_crosscorr_logging(log_file=str(Path(output_dir, 'autolog')))
    else:
        logger = cu.setup_crosscorr_logging(log_file=log)  
    setup_logging()

    if nproc is None:
        nproc = max(os.cpu_count()-2, 1)
    
    corrargs = {
        'sims': sims,
        'sample_rate_desi': sample_rate_desi,
        'sample_rate_hsc': sample_rate_hsc,
        'nproc': nproc,
        'logger': logger,
        'output_dir': output_dir,
    }
    if jackknife:
        corrargs.update({
            'nsamples': args.nsamples,
            'nside': args.resolution,
        })

    print('Running with the following settings:')
    print(f'Output directory: {output_dir}')
    print(f'Sample rate on DESI randoms: {sample_rate_desi}')
    print(f'Sample rate on HSC randoms: {sample_rate_hsc}')
    print(f'Number of threads: {nproc}')
    print(f'Log file: {log}')

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info('Saving bins ...')
    cu.CorrelationMeta.save_bins(output_dir) 
    print('=' * 80)
    
    logger.info(
        f'Sample rate on DESI randoms: {sample_rate_desi} and on HSC randoms: {sample_rate_hsc}\n'
        )
    logger.info(f'Number of threads: {nproc}\n')
    logger.info(f'Output directory: {output_dir}\n')
    strbins = '\n'.join(f"{k} : {v}" for k, v in cu.CorrelationMeta.bins_redshift.items())
    logger.info(f'Bins :{strbins}\n')

    moc_list = sorted(cu.CorrelationMeta.moc_list)
    if patches is not None:
        moc_list = [moc_list[p] for p in patches]
        logger.info(f'Using patches {patches} ... {[Path(moc_list[p]).stem for p in patches]}\n')

    # some more checks on targets
    if isinstance(tgt2, str): 
        tgt2 = [tgt2]
    if isinstance(tgt1, str):
        tgt1 = [tgt1]
    if len(tgt1) == 1:
        tgt1 = [tgt1[0]] * len(tgt2)
    if len(tgt2) == 1:
        tgt2 = [tgt2[0]] * len(tgt1)

    for m in range(len(moc_list)):
        mocf = moc_list[m]
        moc = MOC.from_fits(mocf)
        logger.info(f'MOC {m} : {mocf} ...\n')

        # tgt1 is always a list of DESI type targets
        for t1, t2 in zip(tgt1, tgt2):
            # for logging purposes, no real use here
            bin1 = cu.CorrelationMeta.bins_redshift[t1] 
            bin2 = cu.CorrelationMeta.bins_redshift[t2]

            logger.memory_usage()
            corrclass = cu.figure_out_class(t1, t2, jackknife)
            logger.info(f'{corrclass} will be used for {t1}x{t2} ...')
            cc = corrclass(
                tgt1=t1,
                tgt2=t2,
                moc=moc,
                **corrargs
            )
            logger.memory_usage()

            logger.info(
                f'Running for {t1}x{t2}, bin1 {bin1}, bin2 {bin2}, moc {m}\n' + "=" * 80
                )

            for b1 in range(1, len(bin1)):
                for b2 in range(1, len(bin2)):
                    tb1b2 = time.time()
                    cc.run(b1, b2, m)
                    txt = f'Finished {t1}x{t2}, {b1} : {bin1[b1-1]}-{bin1[b1]}, {b2} : {bin2[b2-1]}-{bin2[b2]} in {time.time()-tb1b2:.2f}s'
                    logger.info(txt)

if __name__ == '__main__':
    print('Starting cross-correlation script ...')
    ti = time.time()
    main()
    print(f'Finished cross-correlation script in {time.time()-ti:.2f}s')