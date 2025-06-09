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

def model(nzdata: tuple[dict, dict], optparams:list[float]):
    '''
    Our parameterized model function, per
    tomographic bin. Integral must be 1.

    Parameters:
    nzdata : tuple[dict, dict]
        A tuple containing the n(z) data:
        - nz : dict
            Dictionary with keys as tracers and values as the number density
            for each tracer over the redshift grid. Is 0 if there is no coverage
            for that tracer in that redshift bin.
        - nzerr : dict
            Dictionary with keys as tracers and values as the error on the
            number density for each tracer over the redshift grid.
    optparams : list[float]
        A list of parameters to fit, which includes:
        xparams : tuple[float, float, float, float, float]
            A tuple containing the fixed parameters:
            - x : float
                The normalization factor for the number density.
            - x_bgs : float
                The normalization factor for the BGS tracer. 
            - etc for the other tracers.
        fitparams : list[float]
            A list of the parameters to fit, which are the pi values for each
            redshift bin. These are the parameters we want to optimize.
    dint : float
        - A sequence converging to 0 to impose a normalization constraint
        on the model, ensuring that the integral of the model is equal to x,
        our normalization factor.
    '''
    nz, nzerr = nzdata
    # assumes all tracers have same redshift grid size which is true 
    # given the way we construct the n(z) data
    n_z = len(nz[list(nz.keys())[0]])
    fitparams = optparams[:n_z]

    x_bgs, x_lrg, x_elg, x_qso = optparams[n_z:]

    x_tracers = {
        'BGS_ANY': x_bgs,
        'LRG': x_lrg,
        'ELGnotqso': x_elg,
        'QSO': x_qso
    }

    return np.sum(
            [
                ((fitparams[i] * x_tracers[t] - nz[t][i])**2) / (nzerr[t][i]**2)
                if nzerr[t][i] > 0 else 0
                for t in nzerr.keys()
                for i in range(n_z)
        ]
    )

def logmodel(nzdata: tuple[dict, dict], optparams:list[float]):
    '''
    Our parameterized model function, per
    tomographic bin. Integral must be 1.

    Parameters:
    nzdata : tuple[dict, dict]
        A tuple containing the n(z) data:
        - nz : dict
            Dictionary with keys as tracers and values as the number density
            for each tracer over the redshift grid. Is 0 if there is no coverage
            for that tracer in that redshift bin.
        - nzerr : dict
            Dictionary with keys as tracers and values as the error on the
            number density for each tracer over the redshift grid.
    optparams : list[float]
        A list of parameters to fit, which includes:
        xparams : tuple[float, float, float, float, float]
            A tuple containing the fixed parameters:
            - x : float
                The normalization factor for the number density.
            - x_bgs : float
                The normalization factor for the BGS tracer. 
            - etc for the other tracers.
        fitparams : list[float]
            A list of the parameters to fit, which are the pi values for each
            redshift bin. These are the parameters we want to optimize.
    dint : float
        - A sequence converging to 0 to impose a normalization constraint
        on the model, ensuring that the integral of the model is equal to x,
        our normalization factor.
    '''
    nz, nzerr = nzdata
    # assumes all tracers have same redshift grid size which is true 
    # given the way we construct the n(z) data
    n_z = len(nz[list(nz.keys())[0]])
    fitparams = optparams[:n_z]

    x_bgs, x_lrg, x_elg, x_qso = optparams[n_z:]

    x_tracers = {
        'BGS_ANY': x_bgs,
        'LRG': x_lrg,
        'ELGnotqso': x_elg,
        'QSO': x_qso
    }

    return np.sum(
            [
                ((np.exp(fitparams[i]) * x_tracers[t] - nz[t][i])**2) / (nzerr[t][i]**2)
                if nzerr[t][i] > 0 else 0
                for t in nzerr.keys()
                for i in range(n_z)
        ]
    )

def calibrate_tomo_bin(path_dictionary:dict, nzs_per_tracer:dict, tomo_bin:int, only_nz:bool=False, method:str='standard'):
    '''

    '''
    if method not in ['standard', 'log']:
        raise ValueError("Method must be 'standard' or 'log'")

    assert tomo_bin > 0 and tomo_bin < 5, 'tomo_bin should be between 1 and 4 (HSC bins)'

    nz_nzerr_tomo = {}
    for tracer, nzs in nzs_per_tracer.items():
        nz_nzerr_tomo[tracer] = nzs[tomo_bin]

    desi_fr = cf.CorrFileReader(path_dictionary['DESI_NGC'])
    ## on which zbins : most likely the 0.05 bins
    # params :
    zmin = 0.05
    zmax = 2.5
    dz = 0.05
    zgrid = np.arange(zmin+dz/2, zmax + 3*dz/2, dz)

    tracer_list = list(nz_nzerr_tomo.keys())
    # constraints for x parameters
    bds_x = {
        1: [(0.01, 10), (0.01, 10), (0.01, 10), (1, 1)],  
        2: [(0.01, 10), (0.01, 10),  (0.01, 10), (0.01, 10)], 
        3: [(0.01, 10), (0.01, 10), (0.01, 10), (0.01, 10)],  
        4: [(1,1), (0.01, 10), (0.01, 10), (0.01, 10)]
    }
    tracer_to_remove = []
    for tracer_index, b_tuple in enumerate(bds_x[tomo_bin]):

        if b_tuple[0] == b_tuple[1]:
            tracer_to_remove.append(tracer_index)
    for tracer in tracer_to_remove:
        nz_nzerr_tomo.pop(tracer_list[tracer])

    # nzdata preparation : select on tomo bin
    # we want to evaluate chi2 on each z bin and fit our constants per tracer
    nzs_tomo = {}
    nzs_err_tomo = {}
    for tracer, nzs in nz_nzerr_tomo.items():
        nzs_tomo[tracer] = np.zeros_like(zgrid)
        nzs_err_tomo[tracer] = np.zeros_like(zgrid)
        #nzs[tomo_bin] is a tuple of arrays on the z grid but needs to know about where on the grid
        #so we can set the rest to 0

        # get effective redshift for the tracer
        valid_bins = desi_fr.get_zeff(tracer, tracer)
        for i, z in enumerate(zgrid):
            close_z = np.isclose(valid_bins, z, atol=dz/4)  # Allow some tolerance for matching
            if np.any(close_z):
                match_index = np.where(close_z)[0][0]
                #print(f"Matching z={z} for tracer {tracer} at index {match_index}")
                nzs_tomo[tracer][i] = nzs[0][match_index]
                nzs_err_tomo[tracer][i] = nzs[1][match_index]
            else:
                nzs_tomo[tracer][i] = 0
                nzs_err_tomo[tracer][i] = 0

    if only_nz:
        return nzs_tomo, nzs_err_tomo, zgrid
        
    n_z = len(zgrid)

    pi0 = np.zeros(n_z) 
    x0 = [1, 1, 1, 1]  # start guesses for x_bgs, x_lrg, ...
    init_params = np.concatenate([pi0, x0])

    if method == 'standard':
        func = lambda p: model(nzdata=(nzs_tomo, nzs_err_tomo), optparams=p)
        constraint = lambda p: np.trapz(p[:n_z], x=zgrid) - 1
        bds = [(0, None)] * n_z + bds_x[tomo_bin]
        init_params = np.concatenate([pi0, x0])
    if method == 'log':
        func = lambda p: logmodel(nzdata=(nzs_tomo, nzs_err_tomo), optparams=p)
        constraint = lambda g: np.exp(g)*dz - 1
        bds = [(None, None)] * n_z + bds_x[tomo_bin]
        init_params = np.concatenate([np.log(pi0 + 1e-10), x0])

    result = minimize(
        fun=func,
        x0=init_params,
        method='SLSQP',
        bounds=bds,  # bounds for pi and x parameters
        constraints=[{'type': 'eq', 'fun': constraint}],  # normalization constraint
        options={'disp': True, 'maxiter': 1000}
    )

    return result
