'''
Inference pipeline for the results.
'''

import numpy as np
import pandas as pd
import astropy.cosmology as acosmo
import fitsio as fio

from pathlib import Path
from pycorr import TwoPointEstimator
from astropy.coordinates import SkyCoord
from mocpy import MOC
from scipy.stats import multivariate_normal

import src.statistics.corrfiles as corrf
import src.statistics.combination as comb
import src.statistics.corrutils as cu
import src.statistics.cosmotools as ct

def combine_estimators(estimators, which_patches=None, rebin=1):
    '''
    From the provided path, collate the measurements on each of the MOCs
    and return a dictionary of the results.
    '''
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

    if hasattr(merged, 'cov'):
        allcov = merged.cov()   
    else:
        allcov = np.zeros((len(merged.sep), len(merged.sep)))

    return merged.sep, merged.corr, allcov

def single_bin_corr(
        estimators : list[TwoPointEstimator], 
        scale_cuts:list, 
        z:float, 
        beta:float=-1,
        rebin:int=1,
        which_patches:list[int]=None,
        integration='single-bin',
        ):
    '''
    Computes the single bin Landy-Szalay estimator (Schmidt+2013, Ménard+2013)
    for the wpp, wss and wsp measurements. 
    '''
    ## single bin integration
    # for now assume all estimators have the same binning
    # as estimator.sep
    if isinstance(estimators, TwoPointEstimator):
        estimators = [estimators]
    if len(estimators) == 0:
        raise ValueError('No estimators provided.')
    
    sep, corr, cov = combine_estimators(
        estimators, 
        which_patches=which_patches, 
        rebin=rebin
        )
    comovsep = ct.arcsec2hMpc(sep*3600, z)

    if scale_cuts is not None :
        scale_mask = (comovsep >= scale_cuts[0]) & (comovsep <= scale_cuts[1]) & ~np.isnan(comovsep)
    else:
        # keep everything if no scale cuts are specified
        scale_mask = np.ones_like(comovsep, dtype=bool)
        scale_cuts = [np.min(comovsep), np.max(comovsep)]

    corr_sc = corr[scale_mask]
    comovsep_sc = comovsep[scale_mask]
    cov_sc = cov[scale_mask][:, scale_mask]

    if integration == 'none':
        return corr_sc, np.sqrt(np.diag(cov_sc)), comovsep_sc

    # now do the single bin integration with $W(r)\propto r^{\beta}$ (default $\beta$ = -1)$
    wkernel = comovsep_sc**(beta)
    # divide by the integral of the kernel to normalize it
    wkernel /= np.trapz(wkernel, x=comovsep_sc)

    if integration == 'single-bin':
        # compute the error bars given the covariance matrix
        w_bar = np.trapz(y=np.multiply(wkernel, corr[scale_mask]), x=comovsep[scale_mask])
        # weights are trapezoidal integration weights so :
        delta_r = np.zeros_like(comovsep_sc)
        delta_r[1:-1] = (comovsep_sc[2:] - comovsep_sc[:-2]) / 2
        delta_r[0] = (comovsep_sc[1] - comovsep_sc[0]) / 2
        delta_r[-1] = (comovsep_sc[-1] - comovsep_sc[-2]) / 2

        # get the linear contribution vector to the covariance matrix
        v = wkernel * delta_r

        # v @ cov_sc @ v = v^T*\Sigma*v
        # where \Sigma is the covariance matrix
        w_err = np.sqrt(v @ cov_sc @ v)
        return w_bar, w_err, comovsep_sc
    
    raise NotImplementedError(f'integration method {integration} not implemented.')

def wss(
        bin_index1, 
        bin_index2, 
        tracer1=None,
        tracer2=None,
        path_NGC=None, 
        path_SGC=None, 
        scale_cuts=[], 
        rebin:int=1, 
        integration='single-bin',
        which_patches:list[int]=None
    ):
    '''
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
    '''

    if path_NGC is not None:
        path_NGC = Path(path_NGC)
        assert path_NGC.exists(), f'Path {path_NGC} does not exist.'
        frNGC = corrf.CorrFileReader(path_NGC)
        bins_t1 = frNGC.get_bins(tracer1)
        bins_t2 = frNGC.get_bins(tracer2)
    else:
        frNGC = None
    if path_SGC is not None:
        path_SGC = Path(path_SGC)
        assert path_SGC.exists(), f'Path {path_SGC} does not exist.'
        frSGC = corrf.CorrFileReader(path_SGC)
        bins_t1 = frNGC.get_bins(tracer1)
        bins_t2 = frNGC.get_bins(tracer2)
    else:
        frSGC = None
    if frNGC is None and frSGC is None:
        raise ValueError('At least one of path_NGC or path_SGC must be provided.')
    
    capdict = cu.CorrelationMeta.capdict
    mocngc = []
    mocsgc = []
    for k, v in capdict.items():
        if v == 'NGC':
            mocngc.append(k)
        if v == 'SGC':
            mocsgc.append(k)

    estimatorNGC = None
    estimatorSGC = None

    if frNGC is not None:
        for moc_id in mocngc:
            filengc = frNGC.get_file(bin_index1, bin_index2, tracer1, tracer2, moc_id)
            try:
                estimatorNGC = TwoPointEstimator.load(
                    filename = filengc
                )
            except FileNotFoundError:
                continue
    if frSGC is not None:
        for moc_id in mocsgc:
            filesgc = frSGC.get_file(bin_index1, bin_index2, tracer1, tracer2, moc_id)
            try:
                estimatorSGC = TwoPointEstimator.load(
                    filename = filesgc
                )
            except FileNotFoundError:
                continue
    estimators = [est for est in [estimatorSGC, estimatorNGC] if est is not None]
    zloc1 = (bins_t1[bin_index1-1] + bins_t1[bin_index1])/2
    zloc2 = (bins_t2[bin_index2-1] + bins_t2[bin_index2])/2
    assert np.isclose(zloc1, zloc2, atol=0.001), (
        f'zloc1 {zloc1} != zloc2 {zloc2}, bin_index1 {bin_index1} | bin_index2 {bin_index2}'
    )
    zloc = (zloc1 + zloc2)/2

    assert len(estimators) > 0, 'desi ngc/sgc estimators not found'
    return single_bin_corr(
        estimators, 
        beta=-1, 
        z=zloc, 
        rebin=rebin,
        integration=integration,
        scale_cuts=scale_cuts,
        which_patches=which_patches,
        )
            
def wpp(
        path:str | Path, 
        bin_index:int, 
        scale_cuts:list, 
        rebin:int=1, 
        integration='single-bin',
        which_patches:list[int]=None
        ):
    '''
    From the provided path, collate the wpp measurements for HSC over
    the MOCs and return an array for the results.
    p : photometric
    '''
    assert bin_index is not None, 'bin_index must be provided'

    estimators = []
    fr = corrf.CorrFileReader(path)
    bins_hsc = fr.get_bins('HSC')

    files = fr.get_file(bin_index, bin_index, 'HSC', 'HSC', moc=None)
    estimators = [
        TwoPointEstimator.load(f) for f in files
    ]
    zloc = (bins_hsc[bin_index-1] + bins_hsc[bin_index])/2

    assert len(estimators) > 0, 'hsc estimators not found'
    return single_bin_corr(
        estimators, 
        z=zloc, 
        beta=-1, 
        rebin=rebin,
        integration=integration,
        scale_cuts=scale_cuts,
        which_patches=which_patches
        )
    
def wsp(
        path:str | Path, 
        tracer:str, 
        tomo_bin:int, 
        fine_bin:int, 
        scale_cuts:list, 
        rebin:int=1, 
        integration='single-bin',
        which_patches:list[int]=None
        ):
    '''
    From the provided path, collate the wsp measurements for HSC and DESI
    cross-correlations and return the measurement.
    s : spectroscopic
    p : photometric
    '''

    fr = corrf.CorrFileReader(path)
    estimators = []
    bins_tracer = fr.get_bins(tracer)

    files = fr.get_file(fine_bin, tomo_bin, tracer, 'HSC', moc=None)

    estimators = [
        TwoPointEstimator.load(f) for f in files
    ]
    zloc = (bins_tracer[fine_bin-1] + bins_tracer[fine_bin])/2

    assert len(estimators) > 0, 'HSCxDESI estimators not found'
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
        scale_cuts, 
        do_bias_correction=True,
        rebin=1,
        return_chunks=False, 
        verbose=False
        ):
    '''
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
    do_bias_correction : bool
        If do_bias_correction is True, will apply the bias correction to the measurement.
        The bias correction is done using the gamma and delta_gamma values from the powerlaw fit.
        If False, no bias correction is applied.
    return_chunks : bool
        If return_chunks is True, will return the individual values used to compute the n(z) for the tracer,
        in order : `wsp_meas, wpp_meas, wss_meas ...`.
    verbose : bool
        If verbose is True, will print the values used to compute the n(z) for the tracer.
    '''
    assert tracer in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY', 'Merged'], f'tracer {tracer} not a DESI tracer.'
    assert tomo_bin in [1, 2, 3, 4], f'tomo_bin {tomo_bin} not a valid bin.'
    
    # let's grab the binning scheme that we are using
    fr = corrf.CorrFileReader(path_dictionary['DESIxHSC'])
    if tracer == 'Merged':
        fine_redshift = _get_fine_redshift_bins(fr)
    else:
        fine_redshift = fr.get_bins(tracer)

    # this is the redshift we are at with the desi tracer
    zloc = (fine_redshift[fine_bin] + fine_redshift[fine_bin-1])/2
    deltaz = fine_redshift[fine_bin] - fine_redshift[fine_bin-1]

    wss_meas, wss_err, _ = wss(
        path_NGC=path_dictionary['DESI_NGC'], 
        path_SGC=path_dictionary['DESI_SGC'], 
        tracer1=tracer,
        tracer2=tracer, 
        bin_index1=fine_bin,
        bin_index2=fine_bin,
        scale_cuts=scale_cuts,
        rebin=rebin,
        integration='single-bin'
    )
    wsp_meas, wsp_err, _  = wsp(
        path_dictionary['DESIxHSC'], 
        tracer=tracer,
        fine_bin=fine_bin,
        tomo_bin=tomo_bin,
        scale_cuts=scale_cuts,
        rebin=rebin,
        integration='single-bin'
    )
    combined_err = comb.combine_error_bars(
        x=wsp_meas, 
        xerr=wsp_err, 
        y=wss_meas, 
        yerr=wss_err
        )
    combined_err /= deltaz

    gamma, delta_gamma = _get_bias_correction(scale_cuts) if do_bias_correction else (0, 0)

    result = wsp_meas / (deltaz * np.sqrt((wss_meas)) * (1 + zloc) ** gamma)
    combined_err = np.sqrt(
        (combined_err / ((1 + zloc) ** gamma))**2 
        + np.log(1+zloc) * combined_err * delta_gamma /((1 + zloc) ** gamma)
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
        do_bias_correction=True,
        rebin=1,
        return_chunks=False, 
        verbose=False
        ):
    
    fine_redshift = _get_fine_redshift_bins(corrf.CorrFileReader(path_dictionary['DESIxHSC']))

    # this is the redshift we are at with the desi tracer
    zloc = (fine_redshift[fine_bin-1] + fine_redshift[fine_bin])/2
    deltaz = fine_redshift[fine_bin] - fine_redshift[fine_bin-1]

    frss = corrf.CorrFileReader(path_dictionary['MergedxMerged'])
    file_ss = frss.get_file(fine_bin, fine_bin, tracer, tracer, "Merged")
    estimators_ss = [TwoPointEstimator.load(file_ss)]
    wss_meas, wss_err, _ = single_bin_corr(
        estimators_ss, 
        rebin=rebin, 
        z=zloc, 
        beta=-1,  
        scale_cuts=scale_cuts
    )

    frsp = corrf.CorrFileReader(path_dictionary['MergedxHSC'])
    file_sp = frsp.get_file(fine_bin, tomo_bin, tracer, 'HSC', "Merged")
    estimators_sp = [TwoPointEstimator.load(file_sp)]
    wsp_meas, wsp_err, _  = single_bin_corr(
        estimators_sp, 
        rebin=rebin, 
        z=zloc, 
        beta=-1, 
        scale_cuts=scale_cuts
    )

    combined_err = comb.combine_error_bars(
        x=wsp_meas, 
        xerr=wsp_err, 
        y=wss_meas, 
        yerr=wss_err
        ) / deltaz
    
    gamma, delta_gamma = _get_bias_correction(scale_cuts) if do_bias_correction else (0, 0)

    result = wsp_meas / (deltaz * np.sqrt((wss_meas)) * (1 + zloc) ** gamma)
    combined_err = np.sqrt(
        (combined_err / ((1 + zloc) ** gamma))**2 
        + np.log(1+zloc) * combined_err * delta_gamma /((1 + zloc) ** gamma)
        )
    if return_chunks:
        return wsp_meas, wsp_err, wss_meas, wss_err, deltaz, zloc, result, combined_err
    return result, combined_err

def full_npz_tomo(
        path_dictionary, 
        tracer, 
        tomo_bin, 
        scale_cuts, 
        do_bias_correction=True,
        rebin=1,
        verbose=False, 
        return_chunks=False,
        ):
    '''
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
    do_bias_correction : bool
        If do_bias_correction is True, will apply the bias correction to the measurement.
        The bias correction is done using the gamma and delta_gamma values from the powerlaw fit.
        If False, no bias correction is applied.
    rebin : int
        Wether to rebin the measurements by a factor.
    verbose : bool
        If verbose is True, will print the values used to compute the n(z) for the tracer.
    '''
    # let's grab the binning scheme that we are using
    fr = corrf.CorrFileReader(path_dictionary['DESIxHSC'])

    # calibration samples from HSC, if needed
    fr_hsc = corrf.CorrFileReader(path_dictionary['HSC'])

    if tracer == 'Merged':
        fine_redshift = _get_fine_redshift_bins(fr)
    else:
        fine_redshift = fr.get_bins(tracer)
    hsc_redshift = fr_hsc.get_bins('HSC')
    if verbose:
        print(f"Using fine redshift : {fine_redshift}")

    # our calibration sample (wpp at zj where j is the fine bin index)
    # get the indices corresponding to the fine bins in the hsc_redshift
    hsc_bins = np.zeros(len(fine_redshift), dtype=int)
    for i in range(len(fine_redshift)):
        hsc_bins[i] = int(np.argmin(np.abs(hsc_redshift - fine_redshift[i])))
    assert len(hsc_bins) == len(fine_redshift), (
        f'len(hsc_bins) = {len(hsc_bins)} != len(fine_redshift) = {len(fine_redshift)}'
    )

    results = []
    if tracer == 'Merged':
        print(f'Using merged method for tracer {tracer} and tomo bin {tomo_bin}.')
        func = compute_npz_merged
    else:
        func = compute_npz
    for i in range(1, len(fine_redshift)):
        out = func(
            path_dictionary,
            tracer=tracer,
            fine_bin=i,
            do_bias_correction=do_bias_correction,
            tomo_bin=tomo_bin,
            scale_cuts=scale_cuts,
            rebin=rebin,
            verbose=verbose,
            return_chunks=return_chunks
        )
        
        if return_chunks:
            results.append(out)
        else:
            nz_s, nz_err_s = out
            results.append((nz_s, nz_err_s))

    if return_chunks:
        return np.array(results)
    else:
        nz, nz_err = zip(*results)
        return np.array(nz), np.array(nz_err)
    
def _get_fine_redshift_bins(fr: corrf.CorrFileReader):
    dzall = []
    mint = 1089 # cmb redshift should be high enough
    maxt = 0
    for t in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']:
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
    fine_redshift = np.arange(
        # we don't append dz because maxt+dz is the true max of the bins
        mint, maxt, dz
        )
    return fine_redshift

def _get_bias_correction(scale_cuts):
    if scale_cuts == [1, 5]:
        #scale cut = [1, 5]
        gamma = 0.4539423871141212
        delta_gamma = 0.03450372293660535
    else:
        raise NotImplementedError
    return gamma, delta_gamma
    
def compute_rcc(
        path_dictionary, 
        tracer1,
        tracer2, 
        bin_index1, 
        bin_index2, 
        rebin=1,
        scale_cuts=None,
        verbose=False
        ):
    '''
    Computes r_cc coefficient for the provided tracer and binning.
    This is broadly similar to n(z) but there is no integration, we grab all scales.

    Refer to full_rcc for the parameters.
    '''
    assert tracer1 in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'], f'tracer {tracer1} not a DESI tracer.'

    if tracer2 == 'HSC':
        w22_meas, w22_cov, w22_sep = wpp(
            path_dictionary['HSC'], 
            bin_index=bin_index2,
            scale_cuts=scale_cuts,
            integration='none',
            rebin=rebin
            )
        w11_meas, w11_cov, w11_sep = wss(
            path_NGC=path_dictionary['DESI_NGC'], 
            path_SGC=path_dictionary['DESI_SGC'], 
            # only give the tracer1 here (DESI)
            tracer1=tracer1,
            tracer2=tracer1, 
            bin_index1=bin_index1,
            bin_index2=bin_index1,
            integration='none',
            rebin=rebin,
            scale_cuts=scale_cuts
        )
        w12_meas, w12_cov, w12_sep = wsp(
            path_dictionary['DESIxHSC'], 
            tracer=tracer1,
            tomo_bin=bin_index2,
            fine_bin=bin_index1,
            integration='none',
            rebin=rebin,
            scale_cuts=scale_cuts
        )
    else:
        w22_meas, w22_cov, w22_sep = wss(
            path_NGC=path_dictionary['DESI_NGC'], 
            path_SGC=path_dictionary['DESI_SGC'],  
            tracer1=tracer2,
            tracer2=tracer2, 
            bin_index1=bin_index2,
            bin_index2=bin_index2,
            scale_cuts=scale_cuts,
            integration='none',
            rebin=rebin
        )
        w11_meas, w11_cov, w11_sep = wss(
            path_NGC=path_dictionary['DESI_NGC'], 
            path_SGC=path_dictionary['DESI_SGC'], 
            tracer1=tracer1, 
            tracer2=tracer1,
            bin_index1=bin_index1,
            bin_index2=bin_index1,
            integration='none',
            rebin=rebin,
            scale_cuts=scale_cuts
        )
        w12_meas, w12_cov, w12_sep = wss(
            path_NGC=path_dictionary['DESI_NGC'], 
            path_SGC=path_dictionary['DESI_SGC'],
            tracer1=tracer1,
            tracer2=tracer2,
            bin_index1=bin_index1,
            bin_index2=bin_index2,
            integration='none',
            rebin=rebin,
            scale_cuts=scale_cuts
        )
    
    wsp_meas = np.array(w12_meas)
    wss_meas = np.array(w11_meas)
    wpp_meas = np.array(w22_meas)

    assert wsp_meas.shape == wpp_meas.shape == wss_meas.shape, (
        f'wsp shape {wsp_meas.shape} != wpp shape {wpp_meas.shape} != wss shape {wss_meas.shape}'
    )
    return wsp_meas / np.sqrt(wpp_meas * wss_meas)

def full_rcc(
        path_dictionary, 
        tracer1, 
        tracer2,
        rebin=1,
        verbose=False,
        scale_cuts=None
        ):
    '''
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
    '''
    # let's grab the binning scheme that we are using
    if tracer1 in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']:
        fr1 = corrf.CorrFileReader(path_dictionary['DESI_NGC'])
        tracer1_redshift = fr1.get_bins(tracer1)
    else:
        raise NotImplementedError(f'{tracer1} not a DESI tracer is not implemented functionality.')

    # calibration samples
    if tracer2 in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']:
        fr2 = corrf.CorrFileReader(path_dictionary['DESI_NGC'])
    else:
        # this is the HSC tracer
        fr2 = corrf.CorrFileReader(path_dictionary['HSC'])
    tracer2_redshift = fr2.get_bins(tracer2)

    # let's check that the bins are all respectively the same sizes.
    delta_z = np.diff(tracer1_redshift)[0]
    assert np.all(np.isclose(np.diff(tracer1_redshift), delta_z)), (
        f'tracer1_redshift bins are not the same size : {np.diff(tracer1_redshift)}, {delta_z}'
    )
    assert np.all(np.isclose(np.diff(tracer2_redshift), delta_z)), (
        f'tracer2_redshiftbins are not the same size : {np.diff(tracer2_redshift)}, {delta_z}'
    )
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
        (t1_to_t2_bin_indices > 0) &
        (t1_to_t2_bin_indices < len(z2_centers) + 1)
    ]
    t2_to_t1_bin_indices = t2_to_t1_bin_indices[
        (t2_to_t1_bin_indices > 0) &
        (t2_to_t1_bin_indices < len(z1_centers) + 1)
    ]

    print(f'tracer1_redshift : {tracer1_redshift}, t1_to_t2_bin_indices : {t1_to_t2_bin_indices}')
    print(f'tracer2_redshift : {tracer2_redshift}, t2_to_t1_bin_indices : {t2_to_t1_bin_indices}')

    rcc = []
    for bin_index1, bin_index2 in zip(t2_to_t1_bin_indices, t1_to_t2_bin_indices):
        print(f'bin_index1, bin_index2 : {bin_index1}, {bin_index2}')

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
        path_dictionary, 
        outdir, 
        tomo_interest='all', 
        verbose=False,
        show_progress=True
        ):
    '''
    For overlapping redshift bins (where there are two tracers) combine them into a single estimator,
    and save them to the provided paths. Also combine on MOCs.
    '''
    tracers = ['LRG', 'QSO', 'ELGnotqso', 'BGS_ANY']

    fr_cross = corrf.CorrFileReader(path_dictionary['DESIxHSC'])
    ngc = path_dictionary['DESI_NGC']
    if ngc is not None:
        fr_auto_NGC = corrf.CorrFileReader(path_dictionary['DESI_NGC'])
    else:
        fr_auto_NGC = None
    sgc = path_dictionary['DESI_SGC']
    if sgc is not None:
        fr_auto_SGC = corrf.CorrFileReader(path_dictionary['DESI_SGC'])
    else:
        fr_auto_SGC = None
    assert fr_cross is not None, 'cross must be provided.'
    assert fr_auto_NGC is not None or fr_auto_SGC is not None, (
        'At least one of auto_NGC or auto_SGC must be provided.'
    )

    if tomo_interest == 'all':
        ## get all tomgraphic bins available in the cross-correlations
        tomo_interest = fr_cross.get_bins('HSC')
        # transform to 1-indexed bins
        tomo_interest = np.arange(1, len(tomo_interest), dtype=int)
        if verbose:
            print(f'Using all tomographic bins : {tomo_interest}')

    redshift_bins = _get_fine_redshift_bins(fr_cross)
    tracer_bins = {t: fr_cross.get_bins(t) for t in tracers}
    redshift_bin_centers = 0.5 * (redshift_bins[:-1] + redshift_bins[1:])
    dz = np.mean(np.diff(redshift_bins))

    estimators_cross = []
    estimators_autos = []
    for zindr, zr in enumerate(redshift_bin_centers):
        if show_progress:
            if (zindr) % (len(redshift_bins) // 10) == 0:
                print(f'Processing redshift bin {zindr} (Completion : {(zindr+1)/len(redshift_bin_centers):.2%})')
        # paths_cross also has to respect tomographic bins, so we will have a list of lists
        paths_cross = [[] for _ in range(len(tomo_interest))]
        paths_autos = []
        for t in tracers:
            # find the index of bint where the two tracers match
            tracer_bin_centers = 0.5 * (tracer_bins[t][:-1] + tracer_bins[t][1:])
            for zindt, zt in enumerate(tracer_bin_centers, start=1):
                if np.isclose(zt, zr, atol=dz/5):
                    # first deal with DESI autos :
                    paths_autos_z = []
                    if fr_auto_NGC is not None:
                        paths_autos_z.extend(fr_auto_NGC.get_file(zindt, zindt, t, t, None))
                    if fr_auto_SGC is not None:
                        paths_autos_z.extend(fr_auto_SGC.get_file(zindt, zindt, t, t, None))
                    assert len(paths_autos_z) > 0, (
                        "No valid autocorrelations. got "
                        f"{fr_auto_NGC.get_file(zindt, zindt, t, t, None)}"
                    )
                    paths_autos.extend(paths_autos_z)

                    # now deal with cross-correlations
                    # for each tomographic bin
                    for index_tomo, hsc_tomo in enumerate(tomo_interest, start=1):
                        # grab all paths on MOC
                        paths_tomo_cross = fr_cross.get_file(zindt, hsc_tomo, t, "HSC", None)
                        # combine on MOCs for this tomo bin
                        paths_cross[index_tomo-1].extend(paths_tomo_cross)   

        #if verbose:
        #    print("Paths for cross-correlations:", paths_cross)
        #    print("Paths for auto-correlations:", paths_autos)
        
        estimators_cross.append([
            np.sum(
                [TwoPointEstimator.load(p).normalize() for p in paths]
            ) 
            for paths in paths_cross
        ])
        estimators_autos.append(
            np.sum(
                [TwoPointEstimator.load(p).normalize() for p in paths_autos]
            )
        )

    assert len(estimators_cross[-1]) == len(tomo_interest), (
        f'estimators_cross[-1] should have 4 tomographic bins, got {len(estimators_cross[-1])}'
    )
    assert len(estimators_autos) == len(estimators_cross), (
        f'estimators_autos should have the same length as estimators_cross, got {len(estimators_autos)} != {len(estimators_cross)}'
    )
    if outdir is None:
        outdir = Path(path_dictionary['DESIxHSC']).parent
    else:
        outdir = Path(outdir)
        if not outdir.exists():
            outdir.mkdir(parents=True, exist_ok=True)

    cross_dir = outdir / 'MergedxHSC'
    cross_dir.mkdir(parents=True, exist_ok=True)
    autos_dir = outdir / 'MergedxMerged' 
    autos_dir.mkdir(parents=True, exist_ok=True)

    # now save the estimators to the provided paths
    for i in range(1, len(redshift_bin_centers)+1):
        for j, (est, tomo_bin) in enumerate(zip(estimators_cross[i-1], tomo_interest), start=1):
            # save the cross-correlations
            file_path = cross_dir / f'MergedxHSC_b1x{i}_b2x{tomo_bin}.npy'
            if isinstance(est, float):
                print(f"It's likely the b1x{i}_b2x{tomo_bin} estimator has no data for the given redshift bin,\nor is not in the tomo bins of interest.")
                continue
            est.save(file_path)
            if verbose:
                print(f'Saved cross-correlation estimator to {file_path}')

        # save the auto-correlations
        file_path = autos_dir / f'MergedxMerged_b1x{i}_b2x{i}.npy'
        if isinstance(estimators_autos[i-1], float):
            if verbose:
                print(f'Skipping empty auto-correlation estimator for b1x{i}')
                print("It's likely this estimator has no data for the given redshift bin")
            continue
        estimators_autos[i-1].save(file_path)
        if verbose:
            print(f'Saved auto-correlation estimator to {file_path}')
    
    return 


def photbias_correction(
        nz, 
        zedges,
        zloc,
        ):
    ct.get_wDM()
        
       
    