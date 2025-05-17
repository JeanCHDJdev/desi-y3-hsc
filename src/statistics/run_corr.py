'''
Convenience script to run cross-correlation between DESI and HSC.
Can also do autocorrelation and jackknife estimates.
'''
import time 
import os
import numpy as np

from mocpy import MOC
from pathlib import Path
from argparse import ArgumentParser
from pycorr import setup_logging

import src.statistics.corrutils as cu

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
        '-e',
        '--estimator',
        type=str,
        default='davispeebles',
        choices=['davispeebles', 'landyszalay', 'peebleshauser'],
        help='Correlation estimator to use. '
    )
    parser.add_argument(
        '-re',
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
        choices=['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY', 'HSC'],
        default=None,
        help='Target 2. '
        'Default is None. Only current option is HSC.'
        )
    parser.add_argument(
        '-r1',
        '--sample_rate_1', 
        type=int,
        default=1,
        help='Sampling rate for DESI randoms. '
        'Defaults to 1.'
    )
    parser.add_argument(
        '-r2',
        '--sample_rate_2', 
        type=int,
        default=1,
        help='Sampling rate for HSC randoms catalog. '
        'Defaults to 1.'
    )
    parser.add_argument(
        '-w',
        '--weight',
        type=str,
        default='nonKP',
        choices=['nonKP', 'PIP', 'base'],
        help='Weighting scheme to use. '
        'Default is nonKP. '
    )
    parser.add_argument(
        '-k',
        '--skip_moc',
        action='store_true',
        default=False,
        help='Wether to skip the MOC masking. '
        'Default is False. '
    )
    parser.add_argument(
        '-z',
        '--z_bin',
        default=False,
        action='store_true',
        help='Wether to bin by the redshift column or the z_bin column. '
    )
    parser.add_argument(
        '-s',
        '--sims',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help='Whether to use the simulated data, and which version. '
        '0: no simulation'
    )
    parser.add_argument(
        '-d',
        '--corr_type',
        type=str,
        default='theta',
        choices=['theta', 'rp'],
        help='Type of correlation to run. '
        'Default is theta, can also be rp. '
    )
    parser.add_argument(
        '-l',
        '--log', 
        type=str,
        default=None,
        help='Log file to store run settings'
    )
    parser.add_argument(
        '-c',
        '--nproc', 
        type=int, 
        help='Number of threads to use for cross-correlation. '
        'Default is `os.cpu_count()-2`.'
        )
    parser.add_argument(
        '-m',
        '--use_mpi',
        action='store_true',
        help='Use MPI for parallelization. '
        'Default is False. Currently not implemented.'
    )
    parser.add_argument(
        '-a',
        '--areas', 
        type=int, 
        nargs='+',
        default=None,
        choices=range(1, 5),
        help='areas of the sky to cross-correlate. '
        )
    
    return parser.parse_args()

def main():
    args = parse_args()

    tgt1 = args.tgt1
    tgt2 = args.tgt2
    jackknife = args.jackknife
    skip_moc = args.skip_moc
    
    sims_version = args.sims
    zbin = args.z_bin
    weight_type = args.weight
    sample_rate_1 = args.sample_rate_1
    sample_rate_2 = args.sample_rate_2
    corr_type = args.corr_type

    resolution = args.resolution
    nsamples = args.nsamples

    output_dir = args.output_dir
    nproc = args.nproc
    log = args.log
    areas = args.areas

    estimator = args.estimator

    if areas is not None:
        areas = np.array([int(p) for p in areas])
        assert len(areas) > 0, 'areas should be a list of integers'
        assert np.all(
            (0 < areas) & (areas < len(cu.CorrelationMeta.moc_list))
            ), (
                f'areas should be less than {len(cu.CorrelationMeta.moc_list)} '
                'and greater than 0'
                )
    else:
        areas = np.arange(len(cu.CorrelationMeta.moc_list))

    ## logging infrastructure
    if log is None:
        logger = cu.setup_crosscorr_logging(log_file=str(Path(output_dir, 'autolog')))
    else:
        logger = cu.setup_crosscorr_logging(log_file=log)  
    setup_logging()

    if nproc is None:
        nproc = max(os.cpu_count()-2, 1)
    
    corrargs = {
        'use_zbin': zbin,
        'sims_version': sims_version,
        'weight_type': weight_type,
        'sample_rate_1': sample_rate_1,
        'sample_rate_2': sample_rate_2,
        'corr_type': corr_type,
        'nproc': nproc,
        'logger': logger,
        'output_dir': output_dir,
        'skip_moc': skip_moc,
        #'estimator': estimator,
    }
    if jackknife:
        corrargs.update({
            'nsamples': nsamples,
            'nside': resolution,
        })

    logger.info(f'Running cross-correlation for the following targets: {tgt1}x{tgt2}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(
        f'Sample rate on HSC randoms: {sample_rate_2}, '
        f'on DESI randoms: {sample_rate_1}'
        )
    logger.info(f'Number of threads: {nproc}')
    if not sims_version > 0:
        logger.info(f'Weighting scheme: {weight_type}')
    else:
        logger.info('Using simulated data ...')
    logger.info(f'Log file: {log}')
    logger.info(f'\nCorrargs :\n{corrargs}\n')

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info('Saving bins ...')
    cu.CorrelationMeta.save_bins(output_dir) 
    logger.info('=' * 80)
    
    logger.info(
        f'Sample rate on DESI randoms: {sample_rate_1} and on HSC randoms: {sample_rate_2}\n'
        )
    logger.info(f'Number of threads: {nproc}\n')
    logger.info(f'Output directory: {output_dir}\n')
    strbins = '\n'.join(f"{k} : {v}" for k, v in cu.CorrelationMeta.bins_all.items())
    logger.info(f'Bins :{strbins}\n')

    moc_list = sorted(cu.CorrelationMeta.moc_list)
    if areas is not None:
        logger.info(f'Using areas {areas} ... {[Path(moc_list[p-1]).stem for p in areas]}\n')
    else:
        areas = np.arange(1, len(moc_list))

    # some more checks on targets
    tgt1, tgt2 = cu.get_target_couple(tgt1, tgt2)

    for m in areas:
        mocf = moc_list[m-1]
        moc = MOC.from_fits(mocf)
        logger.info(f'MOC {m} : {mocf} ...\n')

        # tgt1 is always a list of DESI type targets
        for t1, t2 in zip(tgt1, tgt2):
            # for logging purposes, no real use here
            bin1 = cu.CorrelationMeta.bins_tracers[t1] 
            bin2 = cu.CorrelationMeta.bins_tracers[t2]

            logger.memory_usage()
            corrclass = cu.figure_out_class(t1, t2, jackknife)
            logger.info(f'{corrclass} will be used for {t1}x{t2} ...')
            cc = corrclass(
                tgt1=t1,
                tgt2=t2,
                moc=moc,
                moc_index=m,
                **corrargs
            )
            logger.memory_usage()

            logger.info(
                f'Running for {t1}x{t2}, bin1 {bin1}, bin2 {bin2}, moc {m}\n' + "=" * 80
                )

            for b1 in range(1, len(bin1)):
                for b2 in range(1, len(bin2)):
                    # autocorr skip
                    if t1 == t2 and b1 != b2:
                        continue

                    tb1b2 = time.time()
                    cc.run(b1, b2, m)
                    txt = f'Finished {t1}x{t2}, {b1} : {bin1[b1-1]}-{bin1[b1]}, {b2} : {bin2[b2-1]}-{bin2[b2]} in {time.time()-tb1b2:.2f}s'
                    logger.info(txt)

if __name__ == '__main__':
    print('Starting cross-correlation script ...')
    ti = time.time()
    main()
    print(f'Finished cross-correlation script in {time.time()-ti:.2f}s')