"""
Inference pipeline for the results.
"""

import numpy as np

from pathlib import Path
from pycorr import TwoPointEstimator

import src.statistics.corrfiles as cf
import src.statistics.combination as comb
import src.statistics.corrutils as cu
import src.statistics.cosmotools as ct


def combine_estimators(estimators, which_patches=None, rebin=1):
    """
    From the provided path, collate the measurements on each of the MOCs
    and return a dictionary of the results.
    """
    sep = estimators[0].sep
    # sep is just here to get the size of the arrays
    allcov = np.zeros((len(sep), len(sep)))

    if len(estimators) > 1:
        merged = np.sum(
            [
                est.normalize()
                for est_i, est in enumerate(estimators, start=1)
                if which_patches is None or est_i in which_patches
            ]
        )
    else:
        merged = estimators[0]
    if rebin > 1:
        merged.rebin(rebin)

    if hasattr(merged, "cov"):
        allcov = merged.cov()
    else:
        allcov = np.zeros((len(merged.sep), len(merged.sep)))

    return merged.sep, merged.corr, allcov


def single_bin_corr(
    estimators: list[TwoPointEstimator],
    scale_cuts: list,
    z: float,
    beta: float = -1,
    rebin: int = 1,
    which_patches: list[int] = None,
    integration="single-bin",
):
    """
    From the provided path, collate the measurements on each of the MOCs
    and return a single bin measurement with error bars.
    """
    # assume all estimators have the same binning
    # as estimator.sep
    if isinstance(estimators, TwoPointEstimator):
        estimators = [estimators]
    if len(estimators) == 0:
        raise ValueError("No estimators provided.")

    sep, corr, cov = combine_estimators(
        estimators, which_patches=which_patches, rebin=rebin
    )
    comovsep = ct.arcsec2hMpc(sep * 3600, z)

    if scale_cuts is not None:
        scale_mask = (
            (comovsep >= scale_cuts[0])
            & (comovsep <= scale_cuts[1])
            & ~np.isnan(comovsep)
        )
    else:
        # keep everything if no scale cuts are specified
        scale_mask = np.ones_like(comovsep, dtype=bool)
        scale_cuts = [np.min(comovsep), np.max(comovsep)]

    # keep measurements within scale cuts
    corr_sc = corr[scale_mask]
    comovsep_sc = comovsep[scale_mask]
    cov_sc = cov[scale_mask][:, scale_mask]

    if integration == "none":
        return corr_sc, np.sqrt(np.diag(cov_sc)), comovsep_sc

    # now do the single bin integration with $W(r)\propto r^{\beta}$ (default $\beta$ = -1)$
    wkernel = comovsep_sc ** (
        beta
    )  # note that we have already cut down weights to match the scale cuts.
    # divide by the integral of the kernel to normalize it
    wkernel /= np.trapezoid(y=wkernel, x=comovsep_sc)

    if integration == "single-bin":
        # compute the error bars given the covariance matrix
        w_bar = np.trapezoid(
            y=np.multiply(wkernel, corr[scale_mask]), x=comovsep[scale_mask]
        )
        # weights are trapezoidal integration weights so :
        delta_r = np.zeros_like(comovsep_sc)
        delta_r[1:-1] = (comovsep_sc[2:] - comovsep_sc[:-2]) / 2
        delta_r[0] = (comovsep_sc[1] - comovsep_sc[0]) / 2
        delta_r[-1] = (comovsep_sc[-1] - comovsep_sc[-2]) / 2

        # get the linear contribution vector to the covariance matrix
        # in a trapezoidal integration scheme
        v = wkernel * delta_r

        # v @ cov_sc @ v = v^T*\Sigma*v
        # \Sigma is the covariance matrix => take square root for error contribution
        w_err = np.sqrt(v @ cov_sc @ v)
        return w_bar, w_err, comovsep_sc

    raise NotImplementedError(f"integration method {integration} not implemented.")


def wss(
    bin_index1,
    bin_index2,
    tracer1=None,
    tracer2=None,
    path_NGC=None,
    path_SGC=None,
    scale_cuts=[],
    rebin: int = 1,
    integration="single-bin",
):
    """
    From the provided path, collate the wss measurements for DESI
    and return an array for the results.
    s : spectroscopic

    Parameters
    ----------
    bin_index1 : int
        The bin index for the first tracer.
    bin_index2 : int
        The bin index for the second tracer.
    tracer1 : str
        The first tracer to compute the wss for. Must be one of 'LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'.
    tracer2 : str
        The second tracer to compute the wss for. Must be one of 'LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'.
    path_NGC : str
        The path to the NGC catalog. Can be None if path_SGC is provided.
    path_SGC : str
        The path to the SGC catalog. Can be None if path_NGC is provided.
    scale_cuts : list
        The scale cuts to apply to the measurements. This is a list of two values, the lower and upper
        bounds of the scale cuts, in comoving Mpc/h. If None, no scale cuts are applied.
    rebin : int
        The rebinning factor to apply to the measurements.
    integration : str
        The integration method to use. Can be 'single-bin', 'euclid' or 'none'.
    """

    if path_NGC is not None:
        path_NGC = Path(path_NGC)
        assert path_NGC.exists(), f"Path {path_NGC} does not exist."
        frNGC = cf.CorrFileReader(path_NGC)
        bins_t1 = frNGC.get_bins(tracer1)
        bins_t2 = frNGC.get_bins(tracer2)
    else:
        frNGC = None
    if path_SGC is not None:
        path_SGC = Path(path_SGC)
        assert path_SGC.exists(), f"Path {path_SGC} does not exist."
        frSGC = cf.CorrFileReader(path_SGC)
        bins_t1 = frSGC.get_bins(tracer1)
        bins_t2 = frSGC.get_bins(tracer2)
    else:
        frSGC = None
    if frNGC is None and frSGC is None:
        raise ValueError("At least one of path_NGC or path_SGC must be provided.")

    capdict = cu.CorrelationMeta.capdict
    mocngc = []
    mocsgc = []
    for k, v in capdict.items():
        if v == "NGC":
            mocngc.append(k)
        if v == "SGC":
            mocsgc.append(k)

    estimatorNGC = None
    estimatorSGC = None

    if frNGC is not None:
        for moc_id in mocngc:
            filengc = frNGC.get_file(bin_index1, bin_index2, tracer1, tracer2, moc_id)
            try:
                estimatorNGC = TwoPointEstimator.load(filename=filengc)
            except FileNotFoundError:
                continue
    if frSGC is not None:
        for moc_id in mocsgc:
            filesgc = frSGC.get_file(bin_index1, bin_index2, tracer1, tracer2, moc_id)
            try:
                estimatorSGC = TwoPointEstimator.load(filename=filesgc)
            except FileNotFoundError:
                continue

    estimators = [est for est in [estimatorSGC, estimatorNGC] if est is not None]
    zloc1 = (bins_t1[bin_index1 - 1] + bins_t1[bin_index1]) / 2
    zloc2 = (bins_t2[bin_index2 - 1] + bins_t2[bin_index2]) / 2
    assert np.isclose(
        zloc1, zloc2, atol=0.001
    ), f"zloc1 {zloc1} != zloc2 {zloc2}, bin indices: bin_index1 {bin_index1} | bin_index2 {bin_index2}"
    zloc = (zloc1 + zloc2) / 2

    assert len(estimators) > 0, "desi ngc/sgc estimators not found"
    return single_bin_corr(
        estimators,
        beta=-1,
        z=zloc,
        rebin=rebin,
        integration=integration,
        scale_cuts=scale_cuts,
        which_patches=None,  # this is already determined by estimators...
        # not very efficient coding from my part here lol
    )


def wpp(
    path: str | Path,
    bin_index: int,
    scale_cuts: list,
    rebin: int = 1,
    integration="single-bin",
    which_patches: list[int] = None,
):
    """
    From the provided path, collate the wpp measurements for HSC over
    the MOCs and return an array for the results.
    p : photometric
    """
    assert bin_index is not None, "bin_index must be provided"

    estimators = []
    fr = cf.CorrFileReader(path)
    bins_hsc = fr.get_bins("HSC")

    files = fr.get_file(bin_index, bin_index, "HSC", "HSC", moc=None)
    estimators = [TwoPointEstimator.load(f) for f in files]
    zloc = (bins_hsc[bin_index - 1] + bins_hsc[bin_index]) / 2

    assert len(estimators) > 0, "hsc estimators not found"
    return single_bin_corr(
        estimators,
        z=zloc,
        beta=-1,
        rebin=rebin,
        integration=integration,
        scale_cuts=scale_cuts,
        which_patches=which_patches,
    )


def wsp(
    path: str | Path,
    tracer: str,
    tomo_bin: int,
    fine_bin: int,
    scale_cuts: list,
    rebin: int = 1,
    integration="single-bin",
    which_patches: list[int] = None,
):
    """
    From the provided path, collate the wsp measurements for HSC and DESI
    cross-correlations and return the measurement.
    s : spectroscopic
    p : photometric
    """

    fr = cf.CorrFileReader(path)
    estimators = []
    bins_tracer = fr.get_bins(tracer)

    files = fr.get_file(fine_bin, tomo_bin, tracer, "HSC", moc=None)

    estimators = [TwoPointEstimator.load(f) for f in files]
    zloc = (bins_tracer[fine_bin - 1] + bins_tracer[fine_bin]) / 2

    assert len(estimators) > 0, "HSCxDESI estimators not found"
    return single_bin_corr(
        estimators,
        rebin=rebin,
        z=zloc,
        beta=-1,
        integration=integration,
        scale_cuts=scale_cuts,
        which_patches=which_patches,
    )


def compute_npz(
    path_dictionary,
    tracer,
    fine_bin,
    tomo_bin,
    which_patches,
    scale_cuts,
    do_phot_correction=True,
    do_spec_correction=True,
    correct_for_wdm=True,
    rebin=1,
    return_chunks=False,
    precomp_wdm=None,
):
    """
    Computes n(z) for the provided tracer and binning.

    Parameters
    ----------

    path_dictionary : dict
        Dictionary containing the paths to the HSC and DESI catalogs.

        Scheme :
        --------
        {
            'HSC': `path_to_hsc`,
            'DESI_NGC': `path_to_desi_ngc`,
            'DESI_SGC': `path_to_desi_sgc`,
            'DESIxHSC': `path_to_desi_x_hsc`
        }
    tracer : str
        The tracer to compute n(z) for. Must be one of 'LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'.
    fine_bin : int
        The bin index for the fine binning. This bin is specific to the DESI tracer used to track
        down the dN/dz. Convention being that all bins are 1-indexed.
    tomo_bin : int
        The bin index for the tomographic binning. This bin is specific to the HSC tracer used to track
        down the dN/dz. This can be from 1 to 4 if using the conventional binning scheme.
        Convention being that all bins are 1-indexed and these are the default tomographic bins of HSC.
    scale_cuts : list
        The scale cuts to apply to the measurements. This is a list of two values, the lower and upper
        bounds of the scale cuts, in comoving Mpc/h.
    do_phot_correction : bool
        If do_phot_correction is True, will apply the bias correction to the measurement.
        The bias correction is done using the gamma and delta_gamma values from the powerlaw fit.
        If False, no bias correction is applied.
    return_chunks : bool
        If return_chunks is True, will return the individual values used to compute the n(z) for the tracer,
        in order : `wsp_meas, wpp_meas, wss_meas ...`.
    """
    assert tracer in [
        "LRG",
        "ELGnotqso",
        "ELG_LOPnotqso",
        "QSO",
        "BGS_ANY",
        "Merged",
    ], f"tracer {tracer} not a DESI tracer."
    # assert tomo_bin in [1, 2, 3, 4], f'tomo_bin {tomo_bin} not a valid bin.'

    # let's grab the binning scheme that we are using
    fr = cf.CorrFileReader(path_dictionary["DESIxHSC"])
    if tracer == "Merged":
        fine_redshift = _get_fine_redshift_bins(fr)
    else:
        fine_redshift = fr.get_bins(tracer)

    # this is the redshift we are at with the desi tracer
    zloc = (fine_redshift[fine_bin] + fine_redshift[fine_bin - 1]) / 2
    deltaz = fine_redshift[fine_bin] - fine_redshift[fine_bin - 1]

    wss_meas, wss_err, _ = wss(
        path_NGC=path_dictionary["DESI_NGC"],
        path_SGC=path_dictionary["DESI_SGC"],
        tracer1=tracer,
        tracer2=tracer,
        bin_index1=fine_bin,
        bin_index2=fine_bin,
        scale_cuts=scale_cuts,
        rebin=rebin,
        integration="single-bin",
    )
    wsp_meas, wsp_err, _ = wsp(
        path_dictionary["DESIxHSC"],
        tracer=tracer,
        fine_bin=fine_bin,
        tomo_bin=tomo_bin,
        scale_cuts=scale_cuts,
        rebin=rebin,
        which_patches=which_patches,
        integration="single-bin",
    )

    alpha, delta_alpha, gamma, delta_gamma = (
        _get_bias_correction(scale_cut=scale_cuts)
        if do_phot_correction
        else (1, 0, 0, 0)
    )
    if not do_spec_correction:
        wss_meas = 1
        wss_err = 0

    if not do_spec_correction and do_phot_correction:
        raise ValueError(
            "If doing photometric correction, must do spectroscopic correction too."
        )

    if precomp_wdm is None:
        print("Not accounting for DM evolution, precomp_wdm is None.")
        precomp_wdm = 1

    # case where we only use cross : we take into account w_dm only
    factor_wdm = precomp_wdm
    result = wsp_meas / factor_wdm
    combined_err = wsp_err / factor_wdm

    if do_spec_correction:
        if do_phot_correction:
            factor = deltaz  # wdm evolution will be encompassed by the powerlaw
        else:
            # if not doing phot correction then we take into account evolution of wdm with z for the phot sample
            # so a square root dependence on wdm, that we multiply by delta_z
            factor = np.sqrt(deltaz * precomp_wdm)
        result = wsp_meas / (np.sqrt(wss_meas) * factor)
        combined_err = comb.combine_error_bars(
            x=wsp_meas, xerr=wsp_err, y=wss_meas, yerr=wss_err
        ) / (factor)
    if do_phot_correction:
        result /= alpha * ((1 + zloc) ** gamma)
        combined_err_prealpha = np.sqrt(
            (combined_err / ((1 + zloc) ** gamma)) ** 2
            + np.log(1 + zloc) * combined_err * delta_gamma / ((1 + zloc) ** gamma)
        )
        combined_err = np.sqrt(
            (combined_err_prealpha / alpha) ** 2
            + (result * delta_alpha / alpha**2) ** 2
        )

    if return_chunks:
        return wsp_meas, wsp_err, wss_meas, wss_err, deltaz, zloc, result, combined_err
    return result, combined_err


def compute_npz_merged(
    path_dictionary,
    tracer,
    fine_bin,
    tomo_bin,
    scale_cuts,
    which_patches=None,
    do_phot_correction=True,
    do_spec_correction=True,
    rebin=1,
    return_chunks=False,
    verbose=False,
    precomp_wdm=None,
):

    if which_patches is not None:
        raise ValueError("which_patches must be None for merged catalogs")

    fine_redshift = _get_fine_redshift_bins(
        cf.CorrFileReader(path_dictionary["DESIxHSC"])
    )

    # this is the redshift we are at with the desi tracer
    zloc = (fine_redshift[fine_bin - 1] + fine_redshift[fine_bin]) / 2
    deltaz = fine_redshift[fine_bin] - fine_redshift[fine_bin - 1]

    frss = cf.CorrFileReader(path_dictionary["MergedxMerged"])
    file_ss = frss.get_file(fine_bin, fine_bin, "Merged", "Merged", "Merged")
    frsp = cf.CorrFileReader(path_dictionary["MergedxHSC"])
    file_sp = frsp.get_file(fine_bin, tomo_bin, "Merged", "HSC", "Merged")

    # in case the file is not available on the redshift range... mostly useful for tomo = 1
    if not Path(file_ss).exists() or not Path(file_sp).exists():
        return 0, 0

    estimators_ss = [TwoPointEstimator.load(file_ss)]
    wss_meas, wss_err, _ = single_bin_corr(
        estimators_ss,
        rebin=rebin,
        z=zloc,
        beta=-1,
        scale_cuts=scale_cuts,
    )

    estimators_sp = [TwoPointEstimator.load(file_sp)]
    wsp_meas, wsp_err, _ = single_bin_corr(
        estimators_sp,
        rebin=rebin,
        z=zloc,
        beta=-1,
        scale_cuts=scale_cuts,
    )

    alpha, delta_alpha, gamma, delta_gamma = (
        _get_bias_correction(scale_cut=scale_cuts)
        if do_phot_correction
        else (1, 0, 0, 0)
    )
    if not do_spec_correction:
        wss_meas = 1
        wss_err = 0

    if not do_spec_correction and do_phot_correction:
        raise ValueError(
            "If doing photometric correction, must do spectroscopic correction too."
        )

    # case where we only use cross : we take into account w_dm only
    factor_wdm = precomp_wdm
    result = wsp_meas / factor_wdm
    combined_err = wsp_err / factor_wdm

    if do_spec_correction:
        if do_phot_correction:
            factor = deltaz  # wdm evolution will be encompassed by the powerlaw
        else:
            # if not doing phot correction then we take into account evolution of wdm with z for the phot sample
            factor = np.sqrt(deltaz * precomp_wdm)
        result = wsp_meas / (np.sqrt(wss_meas) * factor)
        combined_err = comb.combine_error_bars(
            x=wsp_meas, xerr=wsp_err, y=wss_meas, yerr=wss_err
        ) / (factor)
    if do_phot_correction:
        result /= alpha * ((1 + zloc) ** gamma)
        combined_err_prealpha = np.sqrt(
            (combined_err / ((1 + zloc) ** gamma)) ** 2
            + np.log(1 + zloc) * combined_err * delta_gamma / ((1 + zloc) ** gamma)
        )
        combined_err = np.sqrt(
            (combined_err_prealpha / alpha) ** 2
            + (result * delta_alpha / alpha**2) ** 2
        )

    if return_chunks:
        return wsp_meas, wsp_err, wss_meas, wss_err, deltaz, zloc, result, combined_err
    return result, combined_err


def full_npz_tomo(
    path_dictionary,
    tracer,
    tomo_bin,
    scale_cuts,
    which_patches=[1, 2, 3, 4],
    do_phot_correction=True,
    do_spec_correction=True,
    rebin=1,
    return_chunks=False,
    precomp_wdm=None,
    mode="Standard",
):
    """
    Computes n(z) for the provided tracer and specific tomographic. Returns the array of n(z) values
    for that tomographic bin and tracer.

    Parameters
    ----------

    path_dictionary : dict
        Dictionary containing the paths to the HSC and DESI catalogs.

        Scheme :
        --------
        {
            'HSC': `path_to_hsc`,
            'DESI_NGC': `path_to_desi_ngc`,
            'DESI_SGC': `path_to_desi_sgc`,
            'DESIxHSC': `path_to_desi_x_hsc`
        }
    tracer : str
        The tracer to compute n(z) for. Must be one of 'LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'.
    tomo_bin : int
        The bin index for the tomographic binning. This bin is specific to the HSC tracer used to track
        down the dN/dz. This can be from 1 to 4. Convention being that all bins are 1-indexed.
    scale_cuts : list
        The scale cuts to apply to the measurements. This is a list of two values, the lower and upper
        bounds of the scale cuts, in comoving Mpc/h.
    do_phot_correction : bool
        If do_phot_correction is True, will apply the bias correction to the measurement.
        The bias correction is done using the gamma and delta_gamma values from the powerlaw fit.
        If False, no bias correction is applied.
    rebin : int
        Wether to rebin the measurements by a factor.
    verbose : bool
        If verbose is True, will print the values used to compute the n(z) for the tracer.
    """
    if not (do_phot_correction and do_spec_correction) and precomp_wdm is None:
        raise ValueError(
            "If not doing bias and spec correction, precomp_wdm must be provided."
        )

    # let's grab the binning scheme that we are using
    fr = cf.CorrFileReader(path_dictionary["DESIxHSC"])

    # calibration samples from HSC, if needed
    fr_hsc = cf.CorrFileReader(path_dictionary["HSC"])

    if mode == "Merged":
        fine_redshift = _get_fine_redshift_bins(fr, tracer=tracer)
    else:
        fine_redshift = fr.get_bins(tracer)
    hsc_redshift = fr_hsc.get_bins("HSC")

    # our calibration sample (wpp at zj where j is the fine bin index)
    # get the indices corresponding to the fine bins in the hsc_redshift
    hsc_bins = np.zeros(len(fine_redshift), dtype=int)
    for i in range(len(fine_redshift)):
        hsc_bins[i] = int(np.argmin(np.abs(hsc_redshift - fine_redshift[i])))
    assert len(hsc_bins) == len(
        fine_redshift
    ), f"len(hsc_bins) = {len(hsc_bins)} != len(fine_redshift) = {len(fine_redshift)}"

    results = []
    if mode == "Merged":
        print(f"Using merged method for tracer {tracer} and tomo bin {tomo_bin}.")
        func = compute_npz_merged
    else:
        print(f"Using standard method for tracer {tracer} and tomo bin {tomo_bin}.")
        func = compute_npz

    dz = fine_redshift[1] - fine_redshift[0]
    assert np.all(
        np.isclose(np.diff(fine_redshift), dz)
    ), "fine_redshift bins are not uniform"

    for i in range(1, len(fine_redshift)):
        zloc = (fine_redshift[i - 1] + fine_redshift[i]) / 2
        wdm = (
            precomp_wdm(zloc) / dz
        )  # integrated over the fine bin, since this is an interpolator

        out = func(
            path_dictionary,
            tracer=tracer,
            fine_bin=i,
            do_phot_correction=do_phot_correction,
            do_spec_correction=do_spec_correction,
            tomo_bin=tomo_bin,
            which_patches=which_patches,
            scale_cuts=scale_cuts,
            rebin=rebin,
            return_chunks=return_chunks,
            precomp_wdm=wdm,
        )

        if return_chunks:
            results.append(out)
        else:
            nz_s, nz_err_s = out
            results.append((nz_s, nz_err_s))

    if return_chunks:
        return np.array(results).T
    else:
        nz, nz_err = zip(*results)
        return np.array(nz), np.array(nz_err)


def _get_fine_redshift_bins(fr: cf.CorrFileReader, tracer="Merged"):
    dzall = []
    mint = 1089  # cmb redshift should be high enough
    maxt = 0
    if tracer == "Merged":
        tracers = ["LRG", "QSO", "BGS_ANY"]
    else:
        if isinstance(tracer, str):
            tracers = [tracer]
        else:
            tracers = tracer
    for t in tracers:
        bin_t = fr.get_bins(t)
        mint = min(mint, min(bin_t))
        maxt = max(maxt, max(bin_t))
        dz_t = np.diff(bin_t)
        assert all(np.isclose(dz_t, dz_t[0]))
        dzall.append(dz_t[0])
        del dz_t
    assert all(np.isclose(dzall, dzall[0]))
    dz = dzall[0]
    del dzall
    fine_redshift = np.arange(mint, maxt, dz)
    return fine_redshift


def _get_bias_correction(scale_cut):
    return ct._get_bias_correction(scale_cut=scale_cut)


def compute_rcc(
    path_dictionary,
    tracer1,
    tracer2,
    bin_index1,
    bin_index2,
    rebin=1,
    scale_cuts=None,
):
    """
    Computes r_cc coefficient for the provided tracer and binning.
    This is broadly similar to n(z) but there is no integration, we grab all scales.

    Refer to full_rcc for the parameters.
    """
    assert tracer1 in [
        "LRG",
        "ELGnotqso",
        "ELG_LOPnotqso",
        "QSO",
        "BGS_ANY",
    ], f"tracer {tracer1} not a DESI tracer."

    if tracer2 == "HSC":
        w22_meas, w22_cov, w22_sep = wpp(
            path_dictionary["HSC"],
            bin_index=bin_index2,
            scale_cuts=scale_cuts,
            integration="none",
            rebin=rebin,
        )
        w11_meas, w11_cov, w11_sep = wss(
            path_NGC=path_dictionary["DESI_NGC"],
            path_SGC=path_dictionary["DESI_SGC"],
            # only give the tracer1 here (DESI)
            tracer1=tracer1,
            tracer2=tracer1,
            bin_index1=bin_index1,
            bin_index2=bin_index1,
            integration="none",
            rebin=rebin,
            scale_cuts=scale_cuts,
        )
        w12_meas, w12_cov, w12_sep = wsp(
            path_dictionary["DESIxHSC"],
            tracer=tracer1,
            tomo_bin=bin_index2,
            fine_bin=bin_index1,
            integration="none",
            rebin=rebin,
            scale_cuts=scale_cuts,
        )
    else:
        w22_meas, w22_cov, w22_sep = wss(
            path_NGC=path_dictionary["DESI_NGC"],
            path_SGC=path_dictionary["DESI_SGC"],
            tracer1=tracer2,
            tracer2=tracer2,
            bin_index1=bin_index2,
            bin_index2=bin_index2,
            scale_cuts=scale_cuts,
            integration="none",
            rebin=rebin,
        )
        w11_meas, w11_cov, w11_sep = wss(
            path_NGC=path_dictionary["DESI_NGC"],
            path_SGC=path_dictionary["DESI_SGC"],
            tracer1=tracer1,
            tracer2=tracer1,
            bin_index1=bin_index1,
            bin_index2=bin_index1,
            integration="none",
            rebin=rebin,
            scale_cuts=scale_cuts,
        )
        w12_meas, w12_cov, w12_sep = wss(
            path_NGC=path_dictionary["DESI_NGC"],
            path_SGC=path_dictionary["DESI_SGC"],
            tracer1=tracer1,
            tracer2=tracer2,
            bin_index1=bin_index1,
            bin_index2=bin_index2,
            integration="none",
            rebin=rebin,
            scale_cuts=scale_cuts,
        )

    wsp_meas = np.array(w12_meas)
    wss_meas = np.array(w11_meas)
    wpp_meas = np.array(w22_meas)

    assert (
        wsp_meas.shape == wpp_meas.shape == wss_meas.shape
    ), f"wsp shape {wsp_meas.shape} != wpp shape {wpp_meas.shape} != wss shape {wss_meas.shape}"
    return wsp_meas / np.sqrt(wpp_meas * wss_meas)


def full_rcc(
    path_dictionary, tracer1, tracer2, rebin=1, verbose=False, scale_cuts=None
):
    """
    Computes r_cc for the provided tracer and specific tomographic. Returns the array of r_cc values
    for that tomographic bin and tracer. The r_cc values are computed using the wpp, wss and wsp
    measurements, HOWEVER note that these bins should completely overlap. There is no "tomographic bin"
    here as we want to assert that the wpp, wss and wsp measurements are done on the same binning scheme.

    Parameters
    ----------

    path_dictionary : dict
        Dictionary containing the paths to the HSC and DESI catalogs.

        Scheme :
        --------
        {
            'HSC': `path_to_hsc`,
            'DESI_NGC': `path_to_desi_ngc`,
            'DESI_SGC': `path_to_desi_sgc`,
            'DESIxHSC': `path_to_desi_x_hsc`
        }
    tracer : str
        The tracer to compute rcc for. Must be one of 'LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'.
    rebin : int
        The rebinning factor to apply to the measurements. This is used to reduce the number of bins
        in the measurements. The rebinning is done by averaging the values in the bins.
    scale_cuts : list
        The scale cuts to apply to the measurements. This is a list of two values, the lower and upper
        bounds of the scale cuts, in comoving Mpc/h. If None, no scale cuts are applied.
    verbose : bool
        If verbose is True, will print the values used to compute the n(z) for the tracer.
    """
    # let's grab the binning scheme that we are using
    if tracer1 in ["LRG", "ELGnotqso", "ELG_LOPnotqso", "QSO", "BGS_ANY"]:
        fr1 = cf.CorrFileReader(path_dictionary["DESI_NGC"])
        tracer1_redshift = fr1.get_bins(tracer1)
    else:
        raise NotImplementedError(
            f"{tracer1} not a DESI tracer is not implemented functionality."
        )

    # calibration samples
    if tracer2 in ["LRG", "ELGnotqso", "QSO", "BGS_ANY"]:
        fr2 = cf.CorrFileReader(path_dictionary["DESI_NGC"])
    else:
        # this is the HSC tracer
        fr2 = cf.CorrFileReader(path_dictionary["HSC"])
    tracer2_redshift = fr2.get_bins(tracer2)

    # let's check that the bins are all respectively the same sizes.
    delta_z = np.diff(tracer1_redshift)[0]
    assert np.all(
        np.isclose(np.diff(tracer1_redshift), delta_z)
    ), f"tracer1_redshift bins are not the same size : {np.diff(tracer1_redshift)}, {delta_z}"
    assert np.all(
        np.isclose(np.diff(tracer2_redshift), delta_z)
    ), f"tracer2_redshiftbins are not the same size : {np.diff(tracer2_redshift)}, {delta_z}"
    ## our calibration sample (wpp at zj where j is the fine bin index)
    # get the indices corresponding to the fine bins in the hsc_redshift
    # Ensure both redshift bin arrays are sorted
    tracer1_redshift = np.sort(tracer1_redshift)
    tracer2_redshift = np.sort(tracer2_redshift)

    z1_centers = 0.5 * (tracer1_redshift[:-1] + tracer1_redshift[1:])
    z2_centers = 0.5 * (tracer2_redshift[:-1] + tracer2_redshift[1:])

    t1_to_t2_bin_indices = np.digitize(z1_centers, tracer2_redshift)
    t2_to_t1_bin_indices = np.digitize(z2_centers, tracer1_redshift)

    t1_to_t2_bin_indices = t1_to_t2_bin_indices[
        (t1_to_t2_bin_indices > 0) & (t1_to_t2_bin_indices < len(z2_centers) + 1)
    ]
    t2_to_t1_bin_indices = t2_to_t1_bin_indices[
        (t2_to_t1_bin_indices > 0) & (t2_to_t1_bin_indices < len(z1_centers) + 1)
    ]

    print(
        f"tracer1_redshift : {tracer1_redshift}, t1_to_t2_bin_indices : {t1_to_t2_bin_indices}"
    )
    print(
        f"tracer2_redshift : {tracer2_redshift}, t2_to_t1_bin_indices : {t2_to_t1_bin_indices}"
    )

    rcc = []
    for bin_index1, bin_index2 in zip(t2_to_t1_bin_indices, t1_to_t2_bin_indices):
        print(f"bin_index1, bin_index2 : {bin_index1}, {bin_index2}")

        try:
            rcc.append(
                compute_rcc(
                    path_dictionary,
                    tracer1=tracer1,
                    tracer2=tracer2,
                    bin_index1=bin_index1,
                    bin_index2=bin_index2,
                    rebin=rebin,
                    verbose=verbose,
                    scale_cuts=scale_cuts,
                )
            )
        except AssertionError as e:
            # debug
            print(e)
            pass

    return rcc


def merge_estimators(
    path_dictionary: dict,
    outdir: str | Path,
    which_tomo="all",
    which_cap="all",
    which_patches="all",
    which_tracers="all",
    verbose=True,
):
    """
    For overlapping redshift bins (where there are two tracers) combine them into a single estimator,
    and save them to the provided paths. Also combine on MOCs.

    Parameters
    ----------
    path_dictionary : dict
        Dictionary containing the paths to the HSC and DESI catalogs.
    outdir : str or Path
        The output directory where the merged estimators will be saved.
    which_tomo : str or list of int, optional
        The tomographic bins to consider for merging. If 'all', all tomographic bins will be used.
        If a list of integers, only those tomographic bins will be considered.
    verbose : bool, optional
        If True, will print the progress of the merging.
    which_cap : str, optional
        The type of MOC to use for the merging of autocorrelations. Can be 'all', 'NGC', 'SGC'.
    which_patches : str or list of int, optional
        The patches to consider for merging. If 'all', all patches will be used.
        If a list of integers, only those patches will be considered. Can be [1, 2, 3, 4] for the four
        sky patches of HSC.
    which_tracers : str or list of str, optional
        The tracers to consider for merging. If 'all', all tracers will be used.
        If a list of strings, only those tracers will be considered. Default is 'all', e.g. :
        ['LRG', 'QSO', 'ELG_LOPnotqso', 'BGS_ANY'].
    """

    fr_cross = cf.CorrFileReader(path_dictionary["DESIxHSC"])
    if which_cap == "all":
        which_cap = ["NGC", "SGC"]
    if isinstance(which_cap, str):
        which_cap = [which_cap]
    ngc = path_dictionary.get("DESI_NGC")
    if ngc is not None and "NGC" in which_cap:
        fr_auto_NGC = cf.CorrFileReader(path_dictionary["DESI_NGC"])
    else:
        fr_auto_NGC = None
    sgc = path_dictionary.get("DESI_SGC")
    if sgc is not None and "SGC" in which_cap:
        fr_auto_SGC = cf.CorrFileReader(path_dictionary["DESI_SGC"])
    else:
        fr_auto_SGC = None
    assert fr_cross is not None, "cross must be provided."
    assert (
        fr_auto_NGC is not None or fr_auto_SGC is not None
    ), "At least one of auto_NGC or auto_SGC must be provided."

    if which_tomo == "all":
        ## get all tomgraphic bins available in the cross-correlations
        which_tomo = fr_cross.get_bins("HSC")
        # transform to 1-indexed bins
        which_tomo = np.arange(1, len(which_tomo), dtype=int)
        if verbose:
            print(f"Using all tomographic bins : {which_tomo}")

    if which_tracers == "all":
        tracers = ["LRG", "QSO", "ELGnotqso", "ELG_LOPnotqso", "BGS_ANY"]
    elif isinstance(which_tracers, str):
        tracers = [which_tracers]
    elif isinstance(which_tracers, list):
        tracers = which_tracers
    assert all(
        t in ["LRG", "QSO", "ELGnotqso", "ELG_LOPnotqso", "BGS_ANY"] for t in tracers
    ), f"which_tracers must be a list of strings in [LRG, QSO, ELGnotqso, ELG_LOPnotqso, BGS_ANY], got {tracers}"

    redshift_bins = _get_fine_redshift_bins(fr_cross, tracer=tracers)
    tracer_bins = {t: fr_cross.get_bins(t) for t in tracers}
    redshift_bin_centers = 0.5 * (redshift_bins[:-1] + redshift_bins[1:])
    dz = np.mean(np.diff(redshift_bins))

    if which_patches == "all":
        which_patches = [1, 2, 3, 4]  # all patches
    elif isinstance(which_patches, int):
        which_patches = [which_patches]
    assert all(
        w in [1, 2, 3, 4] for w in which_patches
    ), f"which_patches must be a list of integers in [1, 2, 3, 4], got {which_patches}"

    estimators_cross = []
    estimators_autos = []
    for zindr, zr in enumerate(redshift_bin_centers):
        if verbose:
            if (zindr) % (len(redshift_bins) // 10) == 0:
                print(
                    f"Processing redshift bin {zindr} (Completion : {(zindr+1)/len(redshift_bin_centers):.2%})"
                )
        # paths_cross also has to respect tomographic bins, so we will have a list of lists
        paths_cross = [[] for _ in range(len(which_tomo))]
        paths_autos = []
        for t in tracers:
            # find the index of bint where the two tracers match
            tracer_bin_centers = 0.5 * (tracer_bins[t][:-1] + tracer_bins[t][1:])
            for zindt, zt in enumerate(tracer_bin_centers, start=1):
                if np.isclose(zt, zr, atol=dz / 5):
                    # first deal with DESI autos :
                    paths_autos_z = []
                    if fr_auto_NGC is not None:
                        # moc 1 + skip moc (-k flag) is NGC config
                        paths_autos_z.extend(
                            fr_auto_NGC.get_file(zindt, zindt, t, t, moc=[1])
                        )
                    if fr_auto_SGC is not None:
                        # moc 3 + skip moc is by default the SGC config... not very clear I know
                        paths_autos_z.extend(
                            fr_auto_SGC.get_file(zindt, zindt, t, t, moc=[3])
                        )
                    assert len(paths_autos_z) > 0, (
                        "No valid autocorrelations. got "
                        f"{fr_auto_NGC.get_file(zindt, zindt, t, t, None)}"
                    )
                    paths_autos.extend(paths_autos_z)

                    # now deal with cross-correlations
                    # for each tomographic bin
                    for index_tomo, hsc_tomo in enumerate(which_tomo, start=1):
                        # grab all paths on MOC
                        paths_tomo_cross = fr_cross.get_file(
                            zindt, hsc_tomo, t, "HSC", moc=which_patches
                        )
                        # combine on MOCs for this tomo bin
                        paths_cross[index_tomo - 1].extend(paths_tomo_cross)

        estimators_cross.append(
            [
                np.sum([TwoPointEstimator.load(p).normalize() for p in paths])
                for paths in paths_cross
            ]
        )
        estimators_autos.append(
            np.sum([TwoPointEstimator.load(p).normalize() for p in paths_autos])
        )

    assert len(estimators_cross[-1]) == len(
        which_tomo
    ), f"estimators_cross[-1] should have 4 tomographic bins, got {len(estimators_cross[-1])}"
    assert len(estimators_autos) == len(
        estimators_cross
    ), f"estimators_autos should have the same length as estimators_cross, got {len(estimators_autos)} != {len(estimators_cross)}"
    if outdir is None:
        outdir = Path(path_dictionary["DESIxHSC"]).parent
    else:
        outdir = Path(outdir)
        if not outdir.exists():
            outdir.mkdir(parents=True, exist_ok=True)

    cross_dir = outdir / "MergedxHSC"
    cross_dir.mkdir(parents=True, exist_ok=True)
    autos_dir = outdir / "MergedxMerged"
    autos_dir.mkdir(parents=True, exist_ok=True)

    # now save the estimators to the provided paths
    for i in range(1, len(redshift_bin_centers) + 1):
        for j, (est, tomo_bin) in enumerate(
            zip(estimators_cross[i - 1], which_tomo), start=1
        ):
            # save the cross-correlations
            file_path = cross_dir / f"MergedxHSC_b1x{i}_b2x{tomo_bin}.npy"
            if isinstance(est, float):
                print(
                    f"It's likely the b1x{i}_b2x{tomo_bin} estimator has no data for the given redshift bin,\nor is not in the tomo bins of interest."
                )
                continue
            est.save(file_path)
            if verbose:
                print(f"Saved cross-correlation estimator to {file_path}")

        # save the auto-correlations
        file_path = autos_dir / f"MergedxMerged_b1x{i}_b2x{i}.npy"
        if isinstance(estimators_autos[i - 1], float):
            if verbose:
                print(f"Skipping empty auto-correlation estimator for b1x{i}")
                print(
                    "It's likely this estimator has no data for the given redshift bin"
                )
            continue
        estimators_autos[i - 1].save(file_path)
        if verbose:
            print(f"Saved auto-correlation estimator to {file_path}")

    return


def merge_results(zvals_merge, npz_merge, npz_err_merge, precision=0.0001):
    """
    Merge results from different tomographic bins using inverse variance weighting.

    Parameters:
    - zvals_merge: list of arrays, each containing redshift values for one tracer/bin
    - npz_merge: list of arrays, each containing values at zvals
    - npz_err_merge: list of arrays, each containing errors at zvals
    - precision: float, the precision to which redshift values are rounded for merging

    Returns:
    - zvals: sorted array of unique redshift values
    - npz: merged values at zvals
    - npz_err: corresponding merged errors
    """
    zvals_rounded = [np.round(z / precision) * precision for z in zvals_merge]
    zvals = np.unique(np.concatenate(zvals_rounded))
    npz = np.zeros_like(zvals)
    npz_err = np.zeros_like(zvals)
    weight_sum = np.zeros_like(zvals)
    value_weight_sum = np.zeros_like(zvals)

    for z_i, npz_i, err_i in zip(zvals_rounded, npz_merge, npz_err_merge):
        indices = np.searchsorted(zvals, z_i)
        valid_err = err_i > 0
        if not any(valid_err):
            continue
        weights = 1.0 / (err_i[valid_err] ** 2)
        value_weight_sum[indices[valid_err]] += npz_i[valid_err] * weights
        weight_sum[indices[valid_err]] += weights

    valid_z_range = weight_sum > 0
    assert any(valid_z_range), "No valid z range found after merging, check input data."
    npz = np.zeros_like(value_weight_sum, dtype=float)
    npz_err = np.zeros_like(value_weight_sum, dtype=float)

    npz[valid_z_range] = value_weight_sum[valid_z_range] / weight_sum[valid_z_range]
    npz_err[valid_z_range] = np.sqrt(1.0 / weight_sum[valid_z_range])

    return np.array(zvals), np.array(npz), np.array(npz_err)
