import numpy as np
import pandas as pd
import fitsio as fio

from pathlib import Path
from pycorr import TwoPointEstimator
from astropy.coordinates import SkyCoord
from mocpy import MOC
from scipy.integrate import simpson
from scipy.interpolate import interp1d

import src.statistics.corrfiles as corrf
import src.statistics.corrutils as cu
import src.statistics.cosmotools as ct

# scale cut in Mpc/h
scale_cuts = [0.5, 8]

def get_desi_ns(target, cap=None, **fetch_desi_kw):
    '''
    Get the number of sources per cap, per target. If cap is None,
    return the number of sources for the target by summing both caps.

    Parameters
    ----------
    cap : str
        The cap to get the number of sources for. If None, return the
        number of sources for the target by summing both caps.
    target : str
        The target to get the number of sources for.
    '''
    assert target in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    if cap is None:
        caps = ['NGC', 'SGC']
    else:
        caps = [cap]
    ncols = 0
    for c in caps:
        if c not in ['NGC', 'SGC']:
            raise ValueError(f'cap must be one of NGC, SGC, or None. Got {c}.')
        desi_f = corrf.fetch_desi_files(tgt=target, cap=c, **fetch_desi_kw)
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
    
def get_desi_ratios(target=None, ):
    '''
    Get the ratios of the number of sources in NGC/SGC compared to the full
    sample.
    '''
    assert target in ['LRG', 'ELGnotqso', 'QSO', 'BGS_ANY']
    ntot = get_desi_ns(target=target)
    nngc = get_desi_ns(target=target, cap='NGC')
    nsgc = get_desi_ns(target=target, cap='SGC')
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

    if save_path is not None and not save_path.exists():
        np.savetxt(save_path, ns, fmt='%f')
    
    return np.array(ns) / ntot

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
    else:
        raise NotImplementedError(f'{tracer} not implemented.')
    return alpha * ((1+z)**2 - 6.565) + beta

def hsc_bias_evolution(z, b):
    '''
    Assume it's a ~constant bias of b*growth factor D(z)~1/(1+z)
    '''
    return b / (1 + z)

def calculate_denom(tracer):
    '''
    Computes denominator for n(z)
    '''

def calculate_numer(hsc_path, desi_path, tracer='LRG'):
    '''
    Computes numerator for n(z)
    This is simply w_sp(r, z_i)
    '''
    wpp_sep, wpp_corr, wpp_cov = wpp(hsc_path)
    wss_sep, wss_corr, wss_cov = wss(desi_path, tracer)

def single_bin_corr(estimator : TwoPointEstimator, beta=-1, z:float=0.1, method='landy-szalay'):
    '''
    Computes the single bin Landy-Szalay estimator (Schmidt+2013, Ménard+2013)
    for the wpp, wss and wsp measurements. 
    '''
    ## single bin integration
    # for now assume all estimators have the same binning
    # as estimator.sep
    sep = estimator.sep
    comovsep = ct.arcsec2hMpc(sep*3600, z)

    scale_mask = (comovsep > scale_cuts[0]) & (comovsep < scale_cuts[1])

    d1d2 = estimator.D1D2
    r1d2 = estimator.R1D2
    d1r2 = estimator.D1R2
    r1r2 = estimator.R1R2

    d1d2_counts = d1d2.ncounts
    r1d2_counts = r1d2.ncounts
    d1r2_counts = d1r2.ncounts
    r1r2_counts = r1r2.ncounts

    # now, get the Nr, Nd number counts for each tracer
    Nd1 = d1d2.size1
    Nd2 = d1d2.size2
    Nr1 = r1r2.size1
    Nr2 = r1r2.size2

    # now do the single bin integration with $W(r)\propto r^{\beta} (default \beta = -1)$
    weights = np.zeros_like(comovsep)
    weights[scale_mask] = comovsep[scale_mask]**(beta)
    
    randrand = _integrate_over_pairs(
        weights, r1r2_counts, comovsep
    ) / (Nr1 + Nr2)
    datadata = _integrate_over_pairs(
        weights, d1d2_counts, comovsep
    ) / (Nd1 + Nd2)
    randdata = _integrate_over_pairs(
        weights, r1d2_counts, comovsep
    ) / (Nr1 + Nd2)
    datarand = _integrate_over_pairs(
        weights, d1r2_counts, comovsep
    ) / (Nd1 + Nr2)

    if method=='landy-szalay':
        w_avg = 1 + 1/randrand * (datadata - randdata - datarand)
    if method=='davis-peebles':
        w_avg = 1 / datarand * ()

    return w_avg

def _integrate_over_pairs(weights, ncounts, sep):
    # we integrate only on the scale cut because of weights being null
    # outside of the scale cut
    return simpson(np.multiply(weights, ncounts), x=sep)

def wss(path, target):
    '''
    From the provided path, collate the wss measurements for DESI
    and return a dictionary of the results.
    '''

def wpp(path):
    '''
    From the provided path, collate the wpp measurements on each of the MOCs
    and return a dictionary of the results.
    '''
    moc_list = cu.CorrelationMeta.moc_list
    fr = corrf.CorrFileReader(path)

    combine_ratios = get_hsc_ratios(path)
    print(f'Combine ratios per MOC: {combine_ratios}')
    bins_hsc = fr.get_bins('HSC')

    allcorr = []
    allcov = []
    allsep = []

    for j in range(1, len(bins_hsc)):
        for i in range(len(moc_list)):
            result = TwoPointEstimator.load(
                fr.get_file(j, j, 'HSC', 'HSC', i)
            )
            sep = result.sep
            corr = result.corr
            if i==0:
                mocsep = np.zeros_like(sep)
                moccorr = np.zeros_like(corr)
                moccov = np.zeros((len(sep), len(sep)))
            if hasattr(result, 'cov'):
                cov = result.cov()
            else:
                cov = np.zeros((len(sep), len(sep)))

            mocsep += sep * combine_ratios[i]
            moccorr += corr * combine_ratios[i]
            moccov += cov * combine_ratios[i]

        allcorr.append(moccorr)
        allcov.append(moccov)
        allsep.append(mocsep)

    print(allsep, corr, cov)

    return allsep, allcorr, allcov
    

    