import numpy as np
import pandas as pd
import fitsio as fio
import matplotlib.pyplot as plt

from pathlib import Path
from pycorr import TwoPointEstimator, utils
from astropy.coordinates import SkyCoord
from mocpy import MOC
from scipy.stats import multivariate_normal
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from functools import partial

import src.statistics.corrfiles as cf
import src.statistics.corrutils as cu
import src.statistics.cosmotools as ct

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

def trapz_weights(x):
    '''
    Compute the trapezoidal weights for a given set of x and y values.
    
    Parameters
    ----------
    x : array-like
        The x values.
    y : array-like
        The y values.
    
    Returns
    -------
    weights : array-like
        The trapezoidal weights.
    '''
    dx = np.diff(x)
    weights = np.zeros_like(x)
    weights[0] = dx[0] / 2                    # First point: only right interval
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2    # Interior: left + right intervals  
    weights[-1] = dx[-1] / 2                  # Last point: only left interval
    
    return weights

def combine_error_bars(x, xerr, y, yerr):
    '''
    Combine error bars for two samples with z+/-zerr=(x+/-xerr)/sqrt(y+/-yerr)
    Returns z, zerr.
    '''
    return np.sqrt(
        (xerr/np.sqrt(y))**2
         + (((x/(2*np.sqrt(y)))/np.abs(y))*yerr)**2
        )

def get_norm_corr(
    path_dictionary,
    nzs_per_tracer,
    tracer1,
    tracer2,
    tomo_bin,
):
    '''
    Get the normalization correction for the two tracers in the given tomographic bin.
    '''
    # find the overlapping bins
    desi_fr = cf.CorrFileReader(path_dictionary['DESI_NGC'])
    
    tracer1_redshift = np.sort(desi_fr.get_bins(tracer1))
    tracer2_redshift = np.sort(desi_fr.get_bins(tracer2))
    z1_centers = 0.5 * (tracer1_redshift[:-1] + tracer1_redshift[1:])
    z2_centers = 0.5 * (tracer2_redshift[:-1] + tracer2_redshift[1:])

    nz1, nz1_err = nzs_per_tracer[tracer1][tomo_bin]
    nz2, nz2_err = nzs_per_tracer[tracer2][tomo_bin]

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
    #print(f"Tracer {tracer1} has {len(t1_to_t2_bin_indices)} overlapping bins with {tracer2}.")
    #print(f"Tracer {tracer2} has {len(t2_to_t1_bin_indices)} overlapping bins with {tracer1}.")
    #print('t2_to_t1_bin_indices :', t2_to_t1_bin_indices)
    #print('t1_to_t2_bin_indices :', t1_to_t2_bin_indices)
    #print(f"Redshift centers for {tracer1} : {z1_centers}")
    #print(f"Redshift centers for {tracer2} : {z2_centers}")

    nz1, nz1_err = nzs_per_tracer[tracer1][tomo_bin]
    nz2, nz2_err = nzs_per_tracer[tracer2][tomo_bin]
    
    mask1 = np.zeros(len(nz1), dtype=bool)
    mask2 = np.zeros(len(nz2), dtype=bool)
    mask1[t2_to_t1_bin_indices - 1] = 1
    mask2[t1_to_t2_bin_indices - 1] = 1
    nz1_overlap = nz1[mask1]
    nz2_overlap = nz2[mask2]
    nz1_err_overlap = nz1_err[mask1]
    nz2_err_overlap = nz2_err[mask2]

    def chi2(alpha, x1, x2, sigma_x1, sigma_x2):
        num = (x1 - alpha * x2)**2
        den = sigma_x1**2 + (alpha**2) * sigma_x2**2
        return np.sum(num / den)

    chi2_func = partial(
        chi2, x1=nz1_overlap, x2=nz2_overlap, sigma_x1=nz1_err_overlap, sigma_x2=nz2_err_overlap
        )
    # alpha ratio such that on the overlapping bins, the two n(z) are equal (chi2 minimized)
    # x1 \approx alpha * x2
    mini_alpha = minimize_scalar(
        chi2_func, 
        bounds=(0., 5), 
        method='bounded'
    )
    if not mini_alpha.success:
        raise ValueError(
            f"Minimization failed for {tracer1} and {tracer2} in tomo bin {tomo_bin}. "
            f"Message: {mini_alpha.message}"
        )
    alpha12 = mini_alpha.x
    # now we can scale the n(z) of tracer2 by alpha12
    nz2_scaled = alpha12 * nz2
    
    # now we can compute the full integral for the two bins such that the integral of
    # the pdf for this bin is 1
    joint_nz1 = list(nz1) + list(nz2_scaled[~mask2])
    joint_z1 = list(z1_centers) + list(z2_centers[~mask2])
    norm1 = simpson(joint_nz1, x=joint_z1)

    joint_nz2 = (list(nz1[~mask1]/alpha12) + list(nz2))
    joint_z2 = list(z1_centers[~mask1]) + list(z2_centers)
    norm2 = simpson(joint_nz2, x=joint_z2)

    plt.plot(joint_z1, joint_nz1, label=f'{tracer1} + {tracer2} n(z)', color='blue')
    plt.plot(joint_z2, joint_nz2, label=f'{tracer2} + {tracer1} n(z)', color='orange')
    plt.xlabel('Redshift')
    plt.ylabel('n(z)')
    plt.title(f'Overlap between {tracer1} and {tracer2} in tomo bin {tomo_bin}')
    plt.legend()
    plt.grid()
    plt.show()
    
    return 1/norm1, 1/norm2

def get_norm_corr_interp_interg(
    path_dictionary,
    nzs_per_tracer,
    tracer1,
    tracer2,
    tomo_bin,
):
    """
    Compute normalization corrections for two tracers in a given tomographic bin by
    finding independent scaling factors for each n(z) that enforce continuity and total normalization.
    """

    desi_fr = cf.CorrFileReader(path_dictionary['DESI_NGC'])
    z1_edges = np.sort(desi_fr.get_bins(tracer1))
    z2_edges = np.sort(desi_fr.get_bins(tracer2))
    z1_centers = 0.5 * (z1_edges[:-1] + z1_edges[1:])
    z2_centers = 0.5 * (z2_edges[:-1] + z2_edges[1:])

    nz1, nz1_err = nzs_per_tracer[tracer1][tomo_bin]
    nz2, nz2_err = nzs_per_tracer[tracer2][tomo_bin]

    # interpolation operators
    interp1 = lambda arr: interp1d(z1_centers, arr, kind='cubic', bounds_error=False)
    interp2 = lambda arr: interp1d(z2_centers, arr, kind='cubic', bounds_error=False)

    zmin_overlap = max(z1_centers.min(), z2_centers.min())
    zmax_overlap = min(z1_centers.max(), z2_centers.max())
    z_overlap = np.linspace(zmin_overlap, zmax_overlap, 300)

    # interpolated n(z) and errors on the overlap region
    nz_interp1 = interp1(nz1)
    nz_interp2 = interp2(nz2)
    nz1_overlap = nz_interp1(z_overlap)
    nz2_overlap = nz_interp2(z_overlap)
    nz1_err_overlap = interp1(nz1_err)(z_overlap)
    nz2_err_overlap = interp2(nz2_err)(z_overlap)

    z_total = np.linspace(
        min(z1_centers.min(), z2_centers.min()),
        max(z1_centers.max(), z2_centers.max()), 
        20000
    )

    def chi2(params):
        a, b = params
        diff = (a * nz1_overlap - b * nz2_overlap)
        sigma2 = (a * nz1_err_overlap)**2 + (b * nz2_err_overlap)**2
        return np.sum(diff**2 / sigma2)
    
    def int_constraint(params):
        a, b = params
        # pre-overlap, overlap, and post-overlap
        z_pre = z_total[z_total < zmin_overlap]
        z_mid = z_total[(z_total >= zmin_overlap) & (z_total <= zmax_overlap)]
        z_post = z_total[z_total > zmax_overlap]

        int_pre = simpson(a * nz_interp1(z_pre), x=z_pre) if len(z_pre) > 0 else 0.
        w = 0.5
        nz_mid = w * a * nz_interp1(z_mid) + (1 - w) * b * nz_interp2(z_mid)
        
        int_mid = simpson(nz_mid, x=z_mid)
        int_post = simpson(b * nz_interp2(z_post), x=z_post) if len(z_post) > 0 else 0.

        return int_pre + int_mid + int_post - 1.0

    result = minimize(
        chi2,
        x0=[1.0, 1.0],
        constraints={'type': 'eq', 'fun': int_constraint},
        bounds=[(0.001, 1000), (0.001, 1000)],
        tol=1e-8,
        options={'ftol': 1e-10, 'maxiter': 2000},
    )

    a_opt, b_opt = result.x

    nz1_scaled = a_opt * nz_interp1(z_total)
    nz2_scaled = b_opt * nz_interp2(z_total)

    joint_nz = np.where(
        (z_total >= zmin_overlap) & (z_total <= zmax_overlap),
        0.5 * (nz1_scaled + nz2_scaled),
        np.where(z_total < zmin_overlap, nz1_scaled, nz2_scaled)
    )

    norm_check = simpson(joint_nz, x=z_total)

    # Plot
    plt.figure(figsize=(4, 3))
    plt.plot(z_total, nz1_scaled, label=f'{tracer1} scaled (a={a_opt:.3f})', color='blue')
    plt.plot(z_total, nz2_scaled, label=f'{tracer2} scaled (b={b_opt:.3f})', color='orange')
    plt.plot(z_total, joint_nz, label='Combined', color='green', linestyle='--')
    plt.xlabel('Redshift')
    plt.ylabel('n(z)')
    plt.title(f'Normalized n(z) combination for {tracer1} + {tracer2} (bin {tomo_bin})\nArea = {norm_check:.4f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return a_opt, b_opt

def get_norm_corr_interp(
    path_dictionary,
    nzs_per_tracer,
    tracer1,
    tracer2,
    tomo_bin,
):
    """
    Compute relative scaling factors for two tracers in a given tomographic bin by
    matching their n(z) values in the overlapping redshift region.
    One tracer is fixed (a=1), and the other is scaled by b.
    The final combined n(z) is normalized to have unit integral.
    """

    desi_fr = cf.CorrFileReader(path_dictionary['DESI_NGC'])
    z1_edges = np.sort(desi_fr.get_bins(tracer1))
    z2_edges = np.sort(desi_fr.get_bins(tracer2))
    z1_centers = 0.5 * (z1_edges[:-1] + z1_edges[1:])
    z2_centers = 0.5 * (z2_edges[:-1] + z2_edges[1:])

    nz1, nz1_err = nzs_per_tracer[tracer1][tomo_bin]
    nz2, nz2_err = nzs_per_tracer[tracer2][tomo_bin]

    interp1 = lambda arr: interp1d(z1_centers, arr, kind='linear', bounds_error=False, fill_value=0)
    interp2 = lambda arr: interp1d(z2_centers, arr, kind='linear', bounds_error=False, fill_value=0)

    zmin_overlap = max(z1_centers.min(), z2_centers.min())
    zmax_overlap = min(z1_centers.max(), z2_centers.max())
    z_overlap = np.linspace(zmin_overlap, zmax_overlap, 300)

    nz1_overlap = interp1(nz1)(z_overlap)
    nz2_overlap = interp2(nz2)(z_overlap)
    nz1_err_overlap = interp1(nz1_err)(z_overlap)
    nz2_err_overlap = interp2(nz2_err)(z_overlap)

    z_total = np.linspace(
        min(z1_centers.min(), z2_centers.min()),
        max(z1_centers.max(), z2_centers.max()),
        20000
    )

    nz_interp1 = interp1(nz1)
    nz_interp2 = interp2(nz2)

    def chi2(b):
        diff = nz1_overlap - b * nz2_overlap
        sigma2 = nz1_err_overlap**2 + (b * nz2_err_overlap)**2
        return np.sum(diff**2 / sigma2)

    result = minimize(
        chi2,
        x0=[1.0],
        bounds=[(0.001, 1000)],
        tol=1e-8,
        options={'ftol': 1e-10, 'maxiter': 1000},
    )

    b_opt = result.x[0]
    a_opt = 1.0

    nz1_scaled = a_opt * nz_interp1(z_total)
    nz2_scaled = b_opt * nz_interp2(z_total)

    joint_nz = np.where(
        (z_total >= zmin_overlap) & (z_total <= zmax_overlap),
        0.5 * (nz1_scaled + nz2_scaled),
        np.where(z_total < zmin_overlap, nz1_scaled, nz2_scaled)
    )

    norm_factor = simpson(joint_nz, x=z_total)
    joint_nz /= norm_factor

    plt.figure(figsize=(4, 3))
    plt.plot(z_total, nz1_scaled, label=f'{tracer1} scaled (a=1)', color='blue')
    plt.plot(z_total, nz2_scaled, label=f'{tracer2} scaled (b={b_opt:.3f})', color='orange')
    plt.plot(z_total, joint_nz, label='Combined (normalized)', color='green', linestyle='--')
    plt.xlabel('Redshift')
    plt.ylabel('n(z)')
    plt.title(f'n(z) combination for {tracer1} + {tracer2} (bin {tomo_bin})\nArea normalized to 1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return a_opt / norm_factor, b_opt / norm_factor


def get_norm_per_tracer(path_dictionary, nz_per_tracer):
    norm_per_tracer = {}
    for t in ['BGS_ANY', 'LRG', 'ELGnotqso', 'QSO']:
        norm_per_tracer[t] = {}

    method = get_norm_corr_interp
    #method = get_norm_corr

    print(f'Bin 1')
    norm1, norm2 = method(
        path_dictionary=path_dictionary,
        nzs_per_tracer=nz_per_tracer,
        tracer1='BGS_ANY',
        tracer2='LRG',
        tomo_bin=1,
    )
    norm_per_tracer['BGS_ANY'][1] = norm1
    norm_per_tracer['LRG'][1] = norm2

    print(f'Bin 2')
    norm1, norm2 = method(
        path_dictionary=path_dictionary,
        nzs_per_tracer=nz_per_tracer,
        tracer1='LRG',
        tracer2='ELGnotqso',
        tomo_bin=2,
    )
    norm_per_tracer['LRG'][2] = norm1
    norm_per_tracer['ELGnotqso'][2] = norm2

    print(f'Bin 3')
    norm1, norm2 = method(
        path_dictionary=path_dictionary,
        nzs_per_tracer=nz_per_tracer,
        tracer1='LRG',
        tracer2='ELGnotqso',
        tomo_bin=3,
    )
    norm_per_tracer['LRG'][3] = norm1
    norm_per_tracer['ELGnotqso'][3] = norm2

    norm1, norm2 = method(
        path_dictionary=path_dictionary,
        nzs_per_tracer=nz_per_tracer,
        tracer1='ELGnotqso',
        tracer2='QSO',
        tomo_bin=3,
    )
    norm_per_tracer['ELGnotqso'][3] = norm1
    norm_per_tracer['QSO'][3] = norm2

    print(f'Bin 4')
    norm1, norm2 = method(
        path_dictionary=path_dictionary,
        nzs_per_tracer=nz_per_tracer,
        tracer1='ELGnotqso',
        tracer2='QSO',
        tomo_bin=4,
    )
    norm_per_tracer['ELGnotqso'][4] = norm1
    norm_per_tracer['QSO'][4] = norm2

    return norm_per_tracer