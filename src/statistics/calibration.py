'''
Cross calibration of the tracers
'''
import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from functools import partial

from src.statistics import cosmotools as ct
from src.statistics import corrutils as cu
from src.statistics import corrfiles as cf

def model_function(pi, params):
    nz, nzerr, x, x_bgs, x_lrg, x_elg, x_qso, dz = params

    x_tracers = {
        'BGS_ANY': x_bgs,
        'LRG': x_lrg,
        'ELGnotqso': x_elg,
        'QSO': x_qso
    }
    nzsum = np.sum([
        (pi/x - nz[t]/x_tracers[t])**2
        for t in x_tracers.keys()
    ])
    nzerrsum = np.sum([
        (nzerr[t]/x_tracers[t])**2
        for t in x_tracers.keys()
    ])
    main_constraint = nzsum / nzerrsum

    int_constraint = np.sum([
        (pi - x)**2 / dz 
    ])

    return main_constraint + int_constraint

def fit_model(nz, nzerr, pi, x, x_bgs, x_lrg, x_elg, x_qso, dz):
    '''
    Fit the model function to the data.
    
    Parameters:
    - nz: dict of number densities for each tracer
    - nzerr: dict of errors on number densities for each tracer
    - pi: dict of p_i values for each tracer
    - x: dict of x values for each tracer
    - x_bgs, x_lrg, x_elg, x_qso: specific x values for each tracer
    - dz: redshift bin width
    
    Returns:
    - popt: optimal parameters from the fit
    '''
    # we minimize over x, x_bgs, x_lrg, x_elg, x_qso
    # over the fit to the curve model_function
    minimize(
        fun = model_function,
        x0 = np.ones_like(nz['BGS_ANY']),
        options = {'disp': True, 'maxiter': 10000},
    )

def fit_model(nz, nzerr, pi, x, x_bgs, x_lrg, x_elg, x_qso, dz):
    # fixed scalars we want to minimize over
    xparams = (
        x, x_bgs, x_lrg, x_elg, x_qso, dz
    )
    # data
    nz, nzerr = []
    partial_model = partial(model_function(params=xparams))
    popt, _ = curve_fit(
        model_function,
        x = 0, #TODO : zgrid here
        y = 
        np.zeros_like(nz['BGS_ANY']),
        maxfev=10000
    )
    return popt

def calibrate_tomo_bin(path_dictionary:dict, nzs_per_tracer:dict, tomo_bin:int):
    '''

    '''
    assert tomo_bin > 0 and tomo_bin < 5, 'tomo_bin should be between 1 and 4 (HSC bins)'
    ## on which zbins to we want to smooth out the p_i ? 
    z_bins = np.arange(0, 2.85, 0.2)
    tracers = list(nzs_per_tracer.keys())

    # we want to evaluate chi2 on each z bin and fit our constants per tracer
    nzs_tomo = {}
    for tracer, nzs in nzs_per_tracer.items():
        nzs_tomo[tracer] = nzs[tomo_bin]
    
    ## let's get the nij for each tracer j and redshift bin i
    # this is the uncalibrated nz basically

    
    curve_fit()
