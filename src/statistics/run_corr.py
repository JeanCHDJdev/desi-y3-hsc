"""
Convenience script to run cross-correlation between DESI and HSC.
Can also do autocorrelation and jackknife estimates.
"""

import time
import os
import numpy as np
import logging

from mocpy import MOC
from pathlib import Path
from argparse import ArgumentParser
from pycorr import setup_logging

import src.statistics.corrutils as cu


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="crosscorr/new/",
        help="Path to the output file (storing the cross-correlation). "
        "Default is crosscorr/new/",
    )
    parser.add_argument(
        "-j",
        "--jackknife",
        action="store_true",
        default=False,
        help="Wether to perform cross correlation with jackknife estimates. ",
    )
    parser.add_argument(
        "-ns",
        "--nsamples",
        type=int,
        default=64,
        help="Number of jackknife samples. " "Default is 64. ",
    )
    parser.add_argument(
        "-e",
        "--estimator",
        type=str,
        default="davispeebles",
        choices=["davispeebles", "landyszalay", "peebleshauser"],
        help="Correlation estimator to use. ",
    )
    parser.add_argument(
        "-re",
        "--resolution",
        type=int,
        default=256,
        help="Healpix pixel resolution for jackknife subsampling. ",
    )
    parser.add_argument(
        "-t1",
        "--tgt1",
        type=str,
        nargs="+",
        default=None,
        choices=["LRG", "ELGnotqso", "QSO", "BGS_ANY", "HSC"],
        help="Target(s) 1. " "Default is None. If None, will use all DESI targets.",
    )
    parser.add_argument(
        "-t2",
        "--tgt2",
        type=str,
        choices=["LRG", "ELGnotqso", "QSO", "BGS_ANY", "HSC"],
        default=None,
        help="Target 2. " "Default is None. Only current option is HSC.",
    )
    parser.add_argument(
        "-r1",
        "--sample_rate_1",
        type=int,
        default=1,
        help="Sampling rate for DESI randoms. " "Defaults to 1.",
    )
    parser.add_argument(
        "-r2",
        "--sample_rate_2",
        type=int,
        default=1,
        help="Sampling rate for HSC randoms catalog. " "Defaults to 1.",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default="PIP",  # changed default to be PIP
        choices=["nonKP", "PIP", "base"],
        help="Weighting scheme to use. " "Default is nonKP. ",
    )
    parser.add_argument(
        "-k",
        "--skip_moc",
        action="store_true",
        default=False,
        help="Wether to skip the MOC masking. " "Default is False. ",
    )
    parser.add_argument(
        "-z",
        "--z_bin",
        default=False,
        action="store_true",
        help="Wether to bin by the redshift column or the z_bin column. ",
    )
    parser.add_argument(
        "-s",
        "--sims",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Whether to use the simulated data, and which version. "
        "0: no simulation",
    )
    parser.add_argument(
        "-d",
        "--corr_type",
        type=str,
        default="theta",
        choices=["theta", "rp"],
        help="Type of correlation to run. " "Default is theta, can also be rp. ",
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="Log file to store run settings"
    )
    parser.add_argument(
        "-c",
        "--nproc",
        type=int,
        help="Number of threads to use for cross-correlation. "
        "Default is `os.cpu_count()-2`.",
    )
    parser.add_argument(
        "-a",
        "--areas",
        type=int,
        nargs="+",
        default=None,
        choices=range(1, 5),
        help="areas of the sky to cross-correlate. ",
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

    # currently not implemented; meant to switch between Landy-Szalay and Davis-Peebles estimators
    if args.estimator not in ["davispeebles", "landyszalay"]:
        raise ValueError(
            f"Invalid estimator {args.estimator}. "
            "Choose from 'davispeebles', 'landyszalay'"
        )
    estimator = args.estimator

    if areas is not None:
        areas = np.array([int(p) for p in areas])
        assert len(areas) > 0, "areas should be a list of integers"
        assert np.all((0 < areas) & (areas < len(cu.CorrelationMeta.moc_list))), (
            f"areas should be less than {len(cu.CorrelationMeta.moc_list)} "
            "and greater than 0"
        )
    else:
        areas = np.arange(1, len(cu.CorrelationMeta.moc_list) + 1)

    ## logging infrastructure
    if log is None:
        logger = cu.setup_crosscorr_logging(log_file=str(Path(output_dir, "autolog")))
    else:
        logger = cu.setup_crosscorr_logging(log_file=log)
    setup_logging()

    if nproc is None:
        nproc = max(os.cpu_count() - 2, 1)

    corrargs = {
        "use_zbin": zbin,
        "sims_version": sims_version,
        "weight_type": weight_type,
        "sample_rate_1": sample_rate_1,
        "sample_rate_2": sample_rate_2,
        "corr_type": corr_type,
        "nproc": nproc,
        "logger": logger,
        "output_dir": output_dir,
        "skip_moc": skip_moc,
    }
    if jackknife:
        corrargs.update(
            {
                "nsamples": nsamples,
                "nside": resolution,
            }
        )

    logger.info(f"Running cross-correlation for the following targets: {tgt1}x{tgt2}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Sample rate on HSC randoms: {sample_rate_2}, "
        f"on DESI randoms: {sample_rate_1}"
    )
    logger.info(f"Number of threads: {nproc}")
    if not sims_version > 0:
        logger.info(f"Weighting scheme: {weight_type}")
    else:
        logger.info("Using simulated data ...")
    logger.info(f"Log file: {log}")
    logger.info(f"\nCorrargs :\n{corrargs}\n")

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Saving bins ...")
    cu.CorrelationMeta.save_bins(output_dir)
    logger.info("=" * 80)

    logger.info(
        f"Sample rate on DESI randoms: {sample_rate_1} and on HSC randoms: {sample_rate_2}\n"
    )
    logger.info(f"Number of threads: {nproc}\n")
    logger.info(f"Output directory: {output_dir}\n")
    strbins = "\n".join(f"{k} : {v}" for k, v in cu.CorrelationMeta.bins_all.items())
    logger.info(f"Bins : {strbins}\n")

    moc_list = sorted(cu.CorrelationMeta.moc_list)
    logger.info(
        f"Using areas {areas} ... {[Path(moc_list[p-1]).stem for p in areas]}\n"
    )

    # some more checks on targets
    tgt1, tgt2 = cu.get_target_couple(tgt1, tgt2)
    assert len(areas) > 0, "areas should be a list of integers"

    for m in areas:
        mocf = moc_list[m - 1]
        if skip_moc:
            moc = None
            logger.info(f"Skipping MOC {m} ...\n")
        else:
            moc = MOC.from_fits(mocf)
            logger.info(f"MOC {m} : {Path(mocf).stem} ...\n")

        # tgt1 is always a list of DESI type targets
        for t1, t2 in zip(tgt1, tgt2):
            # for logging purposes, no real use here
            bin1 = cu.CorrelationMeta.bins_tracers[t1]
            bin2 = cu.CorrelationMeta.bins_tracers[t2]

            logger.memory_usage()
            corrclass = cu.figure_out_class(t1, t2, jackknife)
            cc = corrclass(tgt1=t1, tgt2=t2, moc=moc, moc_index=m, **corrargs)
            logger.memory_usage()

            logger.info(
                ("=" * 80)
                + f"\nRunning for {t1}x{t2}, bin1 {bin1}, bin2 {bin2}, moc {m}\n"
                f"{corrclass} will be used for {t1}x{t2} ...\n"
            )

            for b1 in range(1, len(bin1)):
                for b2 in range(1, len(bin2)):
                    # autocorr skip non-matching bins
                    if t1 == t2 and b1 != b2:
                        continue

                    # for this run, calibrate bin 3 and 4 with DR2
                    # and calibrate 1 and 2 with DR1
                    # if b2 == 3 or b2 == 4:
                    #   continue
                    # if b2 == 1 or b2 == 2:
                    #   continue

                    # for this run (autocorrelations on small redshift bins)
                    # we only measure on nearby redshift bins (3*dz_phot)

                    calib_photoz_bias = True
                    if calib_photoz_bias:
                        if t2 == "HSC":
                            # moreover, if z < 0.9 use DR1 else use DR2
                            if (bin2[b2 - 1] + bin2[b2]) / 2 < 0.9:  # > if DR1
                                logger.info(
                                    f"Skipping cross-correlation for {t1}x{t2}, "
                                    f"bin 1 {b1} : {bin1[b1-1]:.2f}-{bin1[b1]:.2f}, bin 2 {b2} : {bin2[b2-1]:.2f}-{bin2[b2]:.2f} "
                                    f"(z > 0.9)"
                                )
                                continue
                            dz_phot = np.mean(np.diff(bin2))
                            # use 3.1 for float point differences
                            if (
                                bin1[b1 - 1] < bin2[b2 - 1] - 3.1 * dz_phot
                                or bin1[b1] > bin2[b2] + 3.1 * dz_phot
                            ):
                                logger.info(
                                    f"Skipping cross-correlation for {t1}x{t2}, "
                                    f"bin 1 {b1} : {bin1[b1-1]:.2f}-{bin1[b1]:.2f}, bin 2 {b2} : {bin2[b2-1]:.2f}-{bin2[b2]:.2f} "
                                    f"(redshift bins are too far apart)"
                                )
                                continue

                    tb1b2 = time.time()
                    cc.run(b1, b2, m)
                    txt = (
                        f"Finished {t1}x{t2}, {b1} : "
                        f"{bin1[b1-1]:.2f}-{bin1[b1]:.2f}, "
                        f"{b2} : {bin2[b2-1]:.2f}-{bin2[b2]:.2f} in {time.time()-tb1b2:.2f}s"
                    )
                    logger.info(txt)

            cc.save_zeff(t1, t2, m)


if __name__ == "__main__":
    logging.info("Starting cross-correlation script ...")
    ti = time.time()
    main()
    logging.info(f"Finished cross-correlation script in {time.time()-ti:.2f}s")
