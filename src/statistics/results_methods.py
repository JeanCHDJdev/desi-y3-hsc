import numpy as np
import pandas as pd
import fitsio as fio

from pathlib import Path
from pycorr import TwoPointEstimator
from astropy.coordinates import SkyCoord
from mocpy import MOC
from scipy.stats import multivariate_normal
from scipy.integrate import simpson

import src.statistics.corrfiles as corrf
import src.statistics.corrutils as cu
import src.statistics.cosmotools as ct

def get_desi_ns(tracer, cap=None, **fetch_desi_kw):
    '''
    Get the number of sources per cap, per tracer. If cap is None,
    return the number of sources for the tracer by summing both caps.

    Parameters
    ----------
    cap : str
        The cap to get the number of sources for. If None, return the
        number of sources for the tracer by summing both caps.
    tracer : str
        The tracer to get the number of sources for.
    '''
    assert tracer in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    if cap is None:
        caps = ['NGC', 'SGC']
    else:
        caps = [cap]
    ncols = 0
    for c in caps:
        if c not in ['NGC', 'SGC']:
            raise ValueError(f'cap must be one of NGC, SGC, or None. Got {c}.')
        desi_f = corrf.fetch_desi_files(tgt=tracer, cap=c, **fetch_desi_kw)
        ncols += fio.FITS(desi_f)[1].get_nrows()
    return ncols

def get_hsc_ns(moc=None, **fetch_hsc_kw):
    '''
    Get the number of sources in the HSC MOC.
    If moc is None, return the number of sources in the HSC
    catalog. Defaults to None.

    Parameters
    ----------
    moc : MOC
        The MOC to get the number of sources for. If None,
        return the number of sources in the HSC catalog. Defaults to None.
    fetch_hsc_kw : dict
        Keyword arguments to pass to the fetch_hsc_files function.
        See the function for more details.
    '''
    hsc_f = corrf.fetch_hsc_files(**fetch_hsc_kw)
    hsc_hdu = fio.FITS(hsc_f)[1]

    if moc is not None:
        coords = hsc_hdu.read(columns=['RA', 'DEC'])
        hsc_moc = MOC.from_fits(moc)
        ra, dec = coords['ra'], coords['dec']
        hsc_coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
        hsc_moc_mask = hsc_moc.contains_skycoords(hsc_coords)
        return np.sum(hsc_moc_mask)
    else:
        return hsc_hdu.get_nrows()
    
def get_desi_ratios(tracer):
    '''
    Get the ratios of the number of sources in NGC/SGC compared to the full
    sample.
    '''
    assert tracer in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    ntot = get_desi_ns(tracer=tracer)
    nngc = get_desi_ns(tracer=tracer, cap='NGC')
    nsgc = get_desi_ns(tracer=tracer, cap='SGC')
    return nngc / ntot, nsgc / ntot

def get_hsc_ratios(path):
    '''
    Get the ratios of the number of sources in MOCs compared 
    '''
    assert isinstance(path, str) or isinstance(path, Path) or path is None, (
        'path must be a string or Path object, or None'
    )
    save_path = None
    if path is not None:
        save_path = Path(path, 'hsc_moc_ratios.txt')
        if save_path.exists():
            return np.loadtxt(save_path, dtype=float)

    moc_list = cu.CorrelationMeta.moc_list
    ntot = get_hsc_ns()
    ns = []
    for moc in moc_list:
        ns.append(get_hsc_ns(moc=moc))

    result = np.array(ns) / ntot
    if save_path is not None and not save_path.exists():
        np.savetxt(save_path, result, fmt='%f')
    
    return result

def desi_bias_evolution(z, tracer='QSO'):
    """
    Bias model fitted from DR1 unblinded data 
    (the formula from Laurent et al. 2016 (1705.04718))
    """
    if tracer == 'QSO':
        alpha = 0.237
        beta = 2.328
    elif tracer == 'LRG':
        alpha = 0.209
        beta = 2.790
    #elif tracer == 'ELG_LOPnotqso':
    #! WARNING: This is not the same as the LOP sample
    elif tracer == 'ELGnotqso': 
        alpha = 0.153
        beta = 1.541
    elif tracer == 'BGS_ANY':
        #! WARNING: Not too sure what to do here just yet, no model provided
        return 1.0 * np.ones_like(z)
    else:
        raise NotImplementedError(f'{tracer} not implemented.')
    return alpha * ((1+z)**2 - 6.565) + beta

def hsc_bias_evolution(z, b):
    '''
    Assume it's a ~constant bias of b*growth factor D(z)~1/(1+z)
    '''
    return b / (1 / (1 + z))

def single_bin_corr(
        estimators : list[TwoPointEstimator], 
        scale_cuts:list,
        beta:float = -1, 
        z:float = 0.1, 
        method='landy-szalay',
        integration='single-bin',
        skipped_ids=None,
        ratios=None,
        ):
    #! todo : assert this is correct implementation
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

    sep, corr, cov = combine_estimators(estimators, ratios=ratios, skipped_ids=skipped_ids)

    # debug plot to compare with the first estimator and check consistency
    '''
    sep_compare = estimators[0].sep
    corr_compare = estimators[0].corr
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(sep_compare, corr_compare, label='compare')
    plt.plot(sep, corr, label='combined')
    plt.grid()
    plt.xscale('log')
    plt.legend()
    plt.show()
    '''

    comovsep = ct.arcsec2hMpc(sep*3600, z)
    if scale_cuts is not None :
        scale_mask = (comovsep > scale_cuts[0]) & (comovsep < scale_cuts[1])
    else:
        # keep everything if no scale cuts are specified
        scale_mask = np.ones_like(comovsep, dtype=bool)
    comovsep = comovsep[scale_mask]

    if integration == 'none':
        return corr[scale_mask]

    # now do the single bin integration with $W(r)\propto r^{\beta} (default \beta = -1)$
    weights = np.zeros_like(comovsep)
    weights = comovsep**(beta)
    ## normalize the weights with it's integral
    weights /= simpson(weights, x=comovsep)
    
    if integration == 'single-bin':
        #return simpson(np.multiply(weights, estimators[0].corr[scale_mask]), x=comovsep)
        return simpson(np.multiply(weights, corr[scale_mask]), x=comovsep)
        
    elif integration == 'euclid':
        # weird integration scheme I can't really get to work
        
        Nd1 = 0
        Nd2 = 0
        Nr1 = 0
        Nr2 = 0
        for est in estimators:
            Nd1 += est.D1D2.size1
            Nd2 += est.D1D2.size2
            Nr1 += est.R1R2.size1
            Nr2 += est.R1R2.size2
        
        d1d2_counts = np.zeros_like(sep)
        r1d2_counts = np.zeros_like(sep)
        d1r2_counts = np.zeros_like(sep)
        r1r2_counts = np.zeros_like(sep)

        for est in estimators:
            d1d2_counts += est.D1D2.ncounts
            r1d2_counts += est.R1D2.ncounts
            d1r2_counts += est.D1R2.ncounts
            r1r2_counts += est.R1R2.ncounts
        
        rr = _integrate_over_pairs(
            weights, r1r2_counts[scale_mask], comovsep
        ) / (Nr1 + Nr2)
        dd = _integrate_over_pairs(
            weights, d1d2_counts[scale_mask], comovsep
        ) / (Nd1 + Nd2)
        rd = _integrate_over_pairs(
            weights, r1d2_counts[scale_mask], comovsep
        ) / (Nr1 + Nd2)
        dr = _integrate_over_pairs(
            weights, d1r2_counts[scale_mask], comovsep
        ) / (Nd1 + Nr2)

        if method=='landy-szalay':
            w_avg = 1/rr * (dd - rd - dr) + 1
        if method=='davis-peebles':
            w_avg = dd / (dr + rd) - 1
        if method=='peebles-hauser':
            w_avg = dd / rr - 1

        return w_avg
    
    else:
        raise NotImplementedError(f'integration method {integration} not implemented.')

def _integrate_over_pairs(weights, ncounts, sep):
    #! todo : maybe interpolate to acccount for border effects due to the scale cut ?
    # we integrate only on the scale cut because of weights being null
    # outside of the scale cut
    return simpson(np.multiply(weights, ncounts), x=sep)

def wss(path_NGC=None, path_SGC=None, tracer='LRG', bin_index=None, scale_cuts=[], integration='single-bin'):
    '''
    From the provided path, collate the wss measurements for DESI
    and return an array for the results.
    s : spectroscopic
    '''

    assert len(scale_cuts) == 2, 'scale_cuts must be a list of length 2'
    assert bin_index is not None, 'bin_index must be provided'

    assert tracer in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'], f'tracer {tracer} not a DESI tracer.'
    if path_NGC is not None:
        path_NGC = Path(path_NGC)
        assert path_NGC.exists(), f'Path {path_NGC} does not exist.'
        frNGC = corrf.CorrFileReader(path_NGC)
    else:
        frNGC = None
    if path_SGC is not None:
        path_SGC = Path(path_SGC)
        assert path_SGC.exists(), f'Path {path_SGC} does not exist.'
        frSGC = corrf.CorrFileReader(path_SGC)
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
            try:
                estimatorNGC = TwoPointEstimator.load(
                    frNGC.get_file(bin_index, bin_index, tracer, tracer, moc_id)
                )
            except FileNotFoundError:
                continue
    if frSGC is not None:
        for moc_id in mocsgc:
            try:
                estimatorSGC = TwoPointEstimator.load(
                    frSGC.get_file(bin_index, bin_index, tracer, tracer, moc_id)
                )
            except FileNotFoundError:
                continue
    if estimatorNGC is None and estimatorSGC is None:
        raise ValueError('No estimators found for the provided paths.')

    estimators = [est for est in [estimatorSGC, estimatorNGC] if est is not None]
    assert len(estimators) > 0, 'desi ngc/sgc estimators not found'
    return single_bin_corr(
        estimators, 
        beta=-1, 
        z=0.1, 
        integration=integration,
        method='landy-szalay', 
        ratios=get_desi_ratios(tracer), 
        scale_cuts=scale_cuts
        )
            
def wpp(path:str | Path, bin_index:int, scale_cuts:list, integration='single-bin'):
    '''
    From the provided path, collate the wpp measurements for HSC over
    the MOCs and return an array for the results.
    p : photometric
    '''
    assert bin_index is not None, 'bin_index must be provided'

    estimators = []
    fr = corrf.CorrFileReader(path)
    bins_hsc = fr.get_bins('HSC')
    moc_list = cu.CorrelationMeta.moc_list

    skipped_ids = []
    for i in range(len(moc_list)):
        try:
            estimators.append(
                TwoPointEstimator.load(
                    fr.get_file(bin_index, bin_index, 'HSC', 'HSC', i)
                )
            )
        except FileNotFoundError:
            skipped_ids.append(i)
            continue
    z = (bins_hsc[bin_index-1] + bins_hsc[bin_index])/2
    assert len(estimators) > 0, 'hsc estimators not found'
    return single_bin_corr(
        estimators, 
        z=z, 
        beta=-1, 
        integration=integration,
        method='landy-szalay', 
        skipped_ids=skipped_ids,
        scale_cuts=scale_cuts
        )
    
def wsp(path:str | Path, tracer:str, tomo_bin:int, fine_bin:int, scale_cuts:list, integration='single-bin'):
    '''
    From the provided path, collate the wsp measurements for HSC and DESI
    cross-correlations and return the measurement.
    s : spectroscopic
    p : photometric
    '''
    assert tracer in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'], f'tracer {tracer} not a DESI tracer.'

    fr = corrf.CorrFileReader(path)
    estimators = []
    bins_tracer = fr.get_bins(tracer)
    moc_list = cu.CorrelationMeta.moc_list
    skipped_ids = []

    for i in range(len(moc_list)):
        try:
            estimators.append(
                TwoPointEstimator.load(
                    fr.get_file(fine_bin, tomo_bin, tracer, 'HSC', i)
                )
            )
        except FileNotFoundError:
            skipped_ids.append(i)
            continue
    z = (bins_tracer[fine_bin-1] + bins_tracer[fine_bin])/2

    assert len(estimators) > 0, 'HSCxDESI estimators not found'
    return single_bin_corr(
        estimators, 
        z=z, 
        beta=-1, 
        integration=integration,
        method='landy-szalay', 
        skipped_ids=skipped_ids, 
        scale_cuts=scale_cuts
        )

def compute_npz(
        path_dictionary, 
        tracer, 
        fine_bin, 
        hsc_correction_bin, 
        tomo_bin, 
        scale_cuts, 
        sigmaj_correction,
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
        down the dN/dz. This can be from 1 to 4. Convention being that all bins are 1-indexed.
    return_chunks : bool
        If return_chunks is True, will return the individual values used to compute the n(z) for the tracer,
        in order : `wsp_meas, wpp_meas, wss_meas, hsc_bias, desi_bias, deltaz, zloc, result`.
    verbose : bool
        If verbose is True, will print the values used to compute the n(z) for the tracer.
    '''
    assert tracer in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'], f'tracer {tracer} not a DESI tracer.'
    assert tomo_bin in [1, 2, 3, 4], f'fine_bin {tomo_bin} not a valid bin.'
    
    # let's grab the binning scheme that we are using
    fr = corrf.CorrFileReader(path_dictionary['DESIxHSC'])
    fine_redshift = fr.get_bins(tracer)

    # this is the redshift we are at with the desi tracer
    zloc = (fine_redshift[fine_bin-1] + fine_redshift[fine_bin])/2
    deltaz = fine_redshift[fine_bin] - fine_redshift[fine_bin-1]

    wpp_meas = wpp(
        path_dictionary['HSC'], 
        bin_index=hsc_correction_bin,
        scale_cuts=scale_cuts
        )
    wss_meas = wss(
        path_dictionary['DESI_NGC'], 
        path_dictionary['DESI_SGC'], 
        tracer=tracer, 
        bin_index=fine_bin,
        scale_cuts=scale_cuts
    )
    wsp_meas = wsp(
        path_dictionary['DESIxHSC'], 
        tracer=tracer,
        fine_bin=fine_bin,
        tomo_bin=tomo_bin,
        scale_cuts=scale_cuts
    )
    hsc_bias = hsc_bias_evolution(z=zloc, b=0.95)
    desi_bias = desi_bias_evolution(z=zloc, tracer=tracer)
    if verbose:
        print(
            f'B : {hsc_bias:.4f}, {desi_bias:.4f}, prodsqrt : {np.sqrt(hsc_bias) * desi_bias:.4f}, ' 
            f'num : {np.sqrt((hsc_bias * wpp_meas) * (desi_bias * wss_meas)):.4f}, num_no_b : {np.sqrt((wpp_meas) * (wss_meas)):.4f}'
            )

    result = wsp_meas / (np.sqrt((wss_meas) * (wpp_meas * sigmaj_correction)))

    if return_chunks:
        return wsp_meas, wpp_meas, wss_meas, hsc_bias, desi_bias, deltaz, zloc, result
    #print(wpp_meas, wss_meas, wsp_meas, zloc)
    return result

def full_npz_tomo(
        path_dictionary, 
        tracer, 
        tomo_bin, 
        scale_cuts, 
        sigmaj_corrections,
        verbose=False, 
        return_chunks=False
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
    verbose : bool
        If verbose is True, will print the values used to compute the n(z) for the tracer.
    '''
    # let's grab the binning scheme that we are using
    fr = corrf.CorrFileReader(path_dictionary['DESIxHSC'])

    # calibration samples from HSC
    fr_hsc = corrf.CorrFileReader(path_dictionary['HSC'])

    fine_redshift = fr.get_bins(tracer)
    hsc_redshift = fr_hsc.get_bins('HSC')
    ## our calibration sample (wpp at zj where j is the fine bin index)
    # get the indices corresponding to the fine bins in the hsc_redshift
    hsc_bins = np.zeros(len(fine_redshift), dtype=int)
    for i in range(len(fine_redshift)):
        hsc_bins[i] = int(np.argmin(np.abs(hsc_redshift - fine_redshift[i])))
    assert len(hsc_bins) == len(fine_redshift), (
        f'len(hsc_bins) = {len(hsc_bins)} != len(fine_redshift) = {len(fine_redshift)}'
    )
    if verbose:
        print(f'fine_redshift : {fine_redshift}')
        print(f'hsc_redshift : {hsc_redshift}')
        print(f'hsc_bins that match fine_redshift : {hsc_bins}')
    print('sigmaj_corrections : ', sigmaj_corrections, len(sigmaj_corrections))
    
    sjcorr = np.zeros(len(fine_redshift))
    for i in range(len(fine_redshift)):
        index = int(np.argmin(np.abs(hsc_redshift - fine_redshift[i])))
        sjcorr[i] = sigmaj_corrections[index-1]

    nz = []
    for i in range(1, len(fine_redshift)):
        nz.append(
            compute_npz(
                path_dictionary, 
                tracer=tracer,  
                fine_bin=i, 
                hsc_correction_bin=hsc_bins[i],
                sigmaj_correction=sigmaj_corrections[i],
                tomo_bin=tomo_bin,
                scale_cuts=scale_cuts,
                verbose=verbose,
                return_chunks=return_chunks
            )
        )
    return np.array(nz)

def combine_estimators(estimators, ratios=None, skipped_ids=None):
    '''
    From the provided path, collate the measurements on each of the MOCs
    and return a dictionary of the results.
    '''
    if ratios is None:
        # hardcoded MOC ratios by source counts in HSC
        ratios = [
            0.584270,
            0.075393,
            0.232669,
            0.107665,
        ]
    if skipped_ids is None:
        skipped_ids = []
    else:
        ratios = [ratios[i] for i in range(len(ratios)) if i not in skipped_ids]
        ratios = ratios / np.sum(ratios)

    #print(f'ratios : {ratios}')

    sep = estimators[0].sep
    allsep = np.zeros_like(sep)
    allcorr = np.zeros_like(sep)
    allcov = np.zeros((len(sep), len(sep)))

    for i, est in enumerate(estimators):
        sep = est.sep
        corr = est.corr
        if hasattr(est, 'cov'):
            cov = est.cov()
        else:
            cov = np.zeros((len(sep), len(sep)))

        allsep += sep * ratios[i]
        allcorr += corr * ratios[i]
        allcov += cov * ratios[i]

    return allsep, allcorr, allcov
    

def hsc_dnnz_error(expect, mids, num_samples=1000): 
    '''
    Compute the error on the n(z) using a log-normal distribution.
    
    Parameters
    ----------

    expect : array-like
        The expected n(z) values.
    mids : array-like
        The midpoints of the bins.
    num_samples : int
        The number of samples to draw from the log-normal distribution.
    '''
    var = (0.15 * expect)**2
    mu = np.log(expect**2/np.sqrt(var + expect**2))
    sig_2 = np.log(var/expect**2 + 1.0)
    samples = multivariate_normal.rvs(mu, np.diag(sig_2), size=num_samples)
    pz = np.exp(samples)
    pz = np.array([el/np.trapz(el, mids) for el in pz])
    return pz, mu, np.diag(sig_2)
    
def compute_rcc(
        path_dictionary, 
        tracer, 
        fine_bin, 
        hsc_correction_bin, 
        verbose=False
        ):
    '''
    Computes r_cc coefficient for the provided tracer and binning.
    This is broadly similar to n(z) but there is no integration, we grab all scales.
    There is also no sigmaj correction applied to the wpp measurement.

    Refer to full_rcc_tomo for the parameters.
    '''
    assert tracer in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY'], f'tracer {tracer} not a DESI tracer.'

    wpp_meas = wpp(
        path_dictionary['HSC'], 
        bin_index=hsc_correction_bin,
        scale_cuts=None
        )
    wss_meas = wss(
        path_dictionary['DESI_NGC'], 
        path_dictionary['DESI_SGC'], 
        tracer=tracer, 
        bin_index=fine_bin,
        # there is no scale cut : retain all scales
        scale_cuts=None
    )
    wsp_meas = wsp(
        path_dictionary['DESIxHSC'], 
        tracer=tracer,
        fine_bin=fine_bin,
        tomo_bin=hsc_correction_bin,
        scale_cuts=None
    )
    if verbose:
        print(
            f'wsp : {wsp_meas:.4f},\nwpp : {wpp_meas:.4f},\nwss : {wss_meas:.4f}, '
            )
        
    wsp_meas = np.array(wsp_meas)
    wpp_meas = np.array(wpp_meas)
    wss_meas = np.array(wss_meas)
    assert all(wss_meas > 0), 'wss must be positive'
    assert all(wpp_meas > 0), 'wpp must be positive'

    assert wsp_meas.shape == wpp_meas.shape == wss_meas.shape, (
        f'wsp shape {wsp_meas.shape} != wpp shape {wpp_meas.shape} != wss shape {wss_meas.shape}'
    )
    return wsp_meas / np.sqrt(wpp_meas * wss_meas)

def full_rcc(
        path_dictionary, 
        tracer, 
        verbose=False
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
    tomo_bin : int
        The bin index for the tomographic binning. .
    verbose : bool
        If verbose is True, will print the values used to compute the n(z) for the tracer.
    '''
    # let's grab the binning scheme that we are using
    fr = corrf.CorrFileReader(path_dictionary['DESIxHSC'])

    # calibration samples from HSC
    fr_hsc = corrf.CorrFileReader(path_dictionary['HSC'])

    fine_redshift = fr.get_bins(tracer)
    hsc_redshift = fr_hsc.get_bins('HSC')
    ## our calibration sample (wpp at zj where j is the fine bin index)
    # get the indices corresponding to the fine bins in the hsc_redshift
    hsc_bins = np.zeros(len(fine_redshift), dtype=int)
    for i in range(len(fine_redshift)):
        hsc_bins[i] = int(np.argmin(np.abs(hsc_redshift - fine_redshift[i])))
    assert len(hsc_bins) == len(fine_redshift), (
        f'len(hsc_bins) = {len(hsc_bins)} != len(fine_redshift) = {len(fine_redshift)}'
    )
    if verbose:
        print(f'fine_redshift : {fine_redshift}')
        print(f'hsc_redshift : {hsc_redshift}')
        print(f'hsc_bins that match fine_redshift : {hsc_bins}')
    
    # let's check that the bins are all respectively the same sizes.
    delta_z = np.diff(fine_redshift)[0]
    assert np.all(np.diff(fine_redshift) == delta_z), (
        f'fine_redshift bins are not the same size : {np.diff(fine_redshift)}'
    )
    assert np.all(np.diff(hsc_redshift) == delta_z), (
        f'hsc_redshift bins are not the same size : {np.diff(hsc_redshift)}'
    )

    rcc = []
    for i in range(1, len(fine_redshift)):
        rcc.append(
            compute_rcc(
                path_dictionary, 
                tracer=tracer,  
                fine_bin=i, 
                hsc_correction_bin=hsc_bins[i],
                verbose=verbose,
            )
        )
    return np.array(rcc)