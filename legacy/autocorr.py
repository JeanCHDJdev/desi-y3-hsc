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

from pycorr import TwoPointCorrelationFunction, setup_logging


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="autos/new/",
        help="Path to the output file (storing the auto-correlation). "
        "Default is autos/new/",
    )
    parser.add_argument(
        "-t",
        "--tgt",
        type=str,
        default=None,
        help="Target to auto-correlate " "Default is all targets",
    )
    parser.add_argument(
        "-r",
        "--sample_rate",
        type=int,
        default=1,
        help="Sampling rate for the randoms. " "Defaults to 1.",
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="Log file to store run settings"
    )
    parser.add_argument(
        "-w",
        "--nproc",
        type=int,
        help="Number of threads to use for auto-correlation. " "Default is all-2.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="desi",
        choices=["desi", "hsc"],
        help="Mode of auto-correlation. ",
    )
    parser.add_argument(
        "-p",
        "--patches",
        type=int,
        nargs="+",
        default=None,
        help="Patches of the sky to auto-correlate. ",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tgt = args.tgt
    output_dir = args.output_dir
    sample_rate = args.sample_rate
    nproc = args.nproc
    log = args.log
    mode = args.mode

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not Path(output_dir, "bins").exists():
        Path(output_dir, "bins").mkdir(parents=True, exist_ok=True)

    if log is None:
        logger = cu.setup_crosscorr_logging(log_file=str(Path(output_dir, "autolog")))
    else:
        logger = cu.setup_crosscorr_logging(log_file=log)
    setup_logging()

    if nproc is None:
        nproc = max(os.cpu_count() - 2, 1)

    # for now, we will only consider the following targets, could do more later
    avb_tgt = ["LRG", "ELGnotqso", "QSO", "BGS_ANY"]
    if tgt:
        assert tgt in avb_tgt, f"{tgt} not in {avb_tgt}"
        if isinstance(tgt, str):
            tgts = [tgt]
    else:
        tgts = avb_tgt

    print("=" * 80)

    bins_redshift = {
        "BGS_ANY": cu.bins_bgs,
        "LRG": cu.bins_lrg,
        "ELGnotqso": cu.bins_elg,
        "QSO": cu.bins_qso,
        "HSC": cu.bins_hsc,
        "distances": cu.bin_distances,
    }
    for k, v in bins_redshift.items():
        np.savetxt(Path(output_dir, "bins", f"bins_{k}.txt"), v)

    logger.info(f"Number of threads: {nproc}\n")
    logger.info(f"Output directory: {output_dir}\n")
    logger.info(f"Bins redshift :{bins_redshift}\n")
    logger.info(f"Fiducial bin distances: {cu.bin_distances}\n")

    moc_list = cu.moc_list

    for m in range(0, len(moc_list) - 3):
        mocf = moc_list[m]
        logger.info(f"Processing {mocf} ...\n")
        moc = MOC.from_fits(mocf)

        for t in tgts:
            if mode == "desi":
                bin1 = bins_redshift[t]
            else:
                bin1 = cu.bins_hsc
            logger.memory_usage()
            if mode == "hsc":
                cc = cu.HSCAutoCorrelation(
                    moc,
                    output_dir,
                    bin_distances=cu.bin_distances,
                    bin_redshift1=bin1,
                    nproc=nproc,
                    sample_rate_hsc=sample_rate,
                    logger=logger,
                )
            elif mode == "desi":
                cc = cu.DESIAutoCorrelation(
                    t,
                    moc,
                    output_dir,
                    bin_distances=cu.bin_distances,
                    bin_redshift1=bin1,
                    nproc=nproc,
                    sample_rate_desi=sample_rate,
                    logger=logger,
                )
            else:
                raise ValueError(f"Unknown mode {mode}")

            logger.memory_usage()

            for b in range(1, len(bin1)):
                tbc = time.time()
                cc.run(b, m)
                txt = (
                    f"Finished {t if t is not None else mode},"
                    f" {b} (desi) : {bin1[b-1]}-{bin1[b]} in {time.time()-tbc:.2f}s\n"
                )
                logger.info(txt)


if __name__ == "__main__":
    print("Starting auto-correlation script ...")
    ti = time.time()
    main()
    print(f"Finished auto-correlation script in {time.time()-ti:.2f}s")
