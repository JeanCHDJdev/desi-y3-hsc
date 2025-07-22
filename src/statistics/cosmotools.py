import numpy as np
import pyccl as ccl
import math

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import vstack, Table
from scipy.interpolate import interp1d

import src.statistics.corrfiles as cf

# Define the cosmology model and global constants.

# parameters
# We use: ΩDM = 0.258868, Ωb = 0.048252, ℎ = 0.6777, 𝑛𝑠 = 0.95 and 𝜎8 = 0.8 
# (from HSC WL-photoz tomographic analysis, see https://arxiv.org/pdf/2211.16516)
omega_c = 0.258868
omega_b = 0.048252
omega_m = omega_c + omega_b
h = 0.6777
H0 = h * 100
sigma8 = 0.8
n_s = 0.95

# from Core Cosmology Library (CCL)
COSMO_ccl = ccl.Cosmology(
    Omega_c=omega_c, Omega_b=omega_b, h=h, sigma8=sigma8, n_s=n_s
)
# from astropy.cosmology
COSMO_astropy = FlatLambdaCDM(
    H0=H0,
    Om0=omega_m,
    Ob0=omega_b,
)

def arcsec2hMpc(theta, z):
    """
    Convert angular separation (in arcseconds) to 
    transverse comoving separation (in h^-1 Mpc).
    
    Parameters:
    theta: float
       Angular separation in arcseconds
    z: float
       Redshift
    """
    theta = theta * u.arcsec
    d_pm = COSMO_astropy.comoving_transverse_distance(z) 
    x = (theta * d_pm).to(u.Mpc, u.dimensionless_angles()) 
    x *= COSMO_astropy.h 
    return x.value

def hMpc2arcsec(rp, z):
    """
    Convert transverse comoving separation (in h^-1 Mpc) 
    to angular separation (in arcseconds).
    
    Parameters:
    x: float
        Transverse separation in h^-1 Mpc
    z: float
        Redshift
    """
    d_pm = COSMO_astropy.comoving_transverse_distance(z)
    rp_hMpc = rp / COSMO_astropy.h
    theta = (rp_hMpc * u.Mpc / d_pm).to(u.arcsec, u.dimensionless_angles())
    return theta.value

def z2dist(z):
    """
    Convert redshift to comoving distance (in h^-1 Mpc).
    
    Parameters:
    -----------

    z: float | list[float] | np.ndarray[float]
        Redshift
    """
    return np.array(
        COSMO_astropy.comoving_distance(z).value / COSMO_astropy.h, 
        dtype=float
        )

def weights(rp, beta=-1):
    return rp**beta / np.trapz(rp**beta, x=rp)

def chi_ccl(z):
    return ccl.comoving_radial_distance(COSMO_ccl, a=1/(1+z))

def w_dm_ang(rp_vals, z, integrate=False, ell_max=12000):
    """
    Compute angular dark matter correlation function w(theta) at a given redshift z.
    
    Parameters
    ----------
    rp_vals : float | list[float] | np.ndarray[float]
        Comoving separation in h^-1 Mpc.
    z : float
        Redshift at which to compute the angular correlation function.
    COSMO_ccl : ccl.Cosmology, optional
        CCL cosmology object. If None, uses default Planck18 cosmology.
    ell_max : int
        Maximum ell for Cl computation (default: 3000).
    
    Returns
    -------
    wtheta : ndarray
        Angular correlation function values at theta_vals_deg.
    """
    # Define delta-function redshift distribution (normalized)
    zmin = z - 0.025
    zmax = z + 0.025
    zarr = np.linspace(zmin, zmax, 100)
    dndz = np.ones_like(zarr)
    dndz /= np.trapz(dndz, zarr)
    bias = np.ones_like(zarr)  # unit bias for DM

    angular_vals_deg = hMpc2arcsec(rp_vals, z) / 3600  # convert h^-1 Mpc to degrees

    # Create number counts tracer
    tracer = ccl.NumberCountsTracer(
        COSMO_ccl, dndz=(zarr, dndz), bias=(zarr, bias), has_rsd=False
    )

    # Compute Cls
    ells = np.arange(1, ell_max)
    Cls = ccl.angular_cl(
        COSMO_ccl, 
        tracer, 
        tracer, 
        ells, 
        p_of_k_a='delta_matter:delta_matter'
        )

    # Compute w(theta)
    wtheta = ccl.correlations.correlation(
        COSMO_ccl, 
        ell=ells, 
        C_ell=Cls,
        theta=angular_vals_deg, 
        type='NN'
    )
    if integrate:
        # Integrate w(theta) over the angular separation
        w = np.trapz(
            np.multiply(wtheta, weights(angular_vals_deg)),
            x=angular_vals_deg
        )
        return w
    else:
        return wtheta
    
def p_mat_nonlin(l,z):
    return ccl.power.nonlin_power(COSMO_ccl, k=(l+0.5)/chi_ccl(z), a=1/(1+z), p_of_k_a='delta_matter:delta_matter')

def p_mat_lin(l,z):
    return ccl.power.linear_power(COSMO_ccl, k=(l+0.5)/chi_ccl(z), a=1/(1+z), p_of_k_a='delta_matter:delta_matter')

def w_dm(rp_vals, z, integrate=False, ell_max=12000):
    '''
    w_dm expects rp_vals in h^-1 Mpc.
    '''
    rp_vals_Mpc = rp_vals / COSMO_astropy.h  # convert to Mpc (1h^-1 Mpc ~1.43 Mpc)
    c_light = 299792.458  # speed of light in km/s
    Ell = range(1, ell_max)

    Hz = COSMO_astropy.H(z).value
    P_delta = [p_mat_nonlin(l, z) for l in Ell]

    theta = rp_vals_Mpc/chi_ccl(z)*360/(2*math.pi) 
    norm = Hz/c_light*(1/chi_ccl(z)**2)
    xi_dm = norm * ccl.correlations.correlation(
        COSMO_ccl, ell=Ell, C_ell=P_delta, theta=theta, type='NN', #method='Legendre'
        )
    
    if integrate:
        return np.trapz(
            np.multiply(xi_dm, weights(rp_vals_Mpc)),
            x=rp_vals_Mpc
        )
    else:
        return xi_dm

def redshift_distribution(bounds, tracer, discretization=100):
    centers = []
    if isinstance(bounds, (list, np.ndarray)):
        if isinstance(bounds[0], (float, int)):
            assert len(bounds) == 2, "Bounds must be in the form [z_min, z_max]"
            bounds = [tuple(bounds)]
        elif isinstance(bounds[0], (list, tuple)):
            assert all(len(b) == 2 for b in bounds), "Bounds must be in the form [[z_min, z_max], ...]"
            # convert to tuples for consistency
            bounds = [tuple(b) for b in bounds]
        else:
            raise TypeError("Bounds must be a list of two floats or a list of lists/tuples of two floats")
    else:
        raise TypeError("Bounds must be a list or numpy array")
    for b in bounds:
        assert b[0] < b[1], "Bounds must be in the form [z_min, z_max] with z_min < z_max"
        centers.append(0.5 * (b[0] + b[1]))
    dz = np.diff(centers)
    assert all(np.isclose(d, dz[0]) for d in dz), "Bounds must be equally spaced in redshift"
    dz = dz[0]  # use the first dz as the common interval

    assert isinstance(discretization, int) and discretization > 0, "Discretization must be a positive integer"
    assert tracer in ['all', 'BGS_ANY', 'LRG', 'ELG_LOPnotqso', 'QSO']

    zdata = None
    files = {
        t : np.array([cf.fetch_desi_files(t, randoms=False, cap=cap) for cap in ['NGC', 'SGC']]).flatten() 
            for t in ['BGS_ANY', 'LRG', 'ELG_LOPnotqso', 'QSO']
        }
    if tracer == 'all':
        # get the redshift file to get the distribution
        allf = []
        for k, f in files.items():
            allf.extend(f)
    else:
        # get the redshift file for the specific tracer
        allf = files[tracer]
    zdata = vstack([Table.read(f) for f in allf])

    return zdata["Z"].data

def spectroscopic_bias_model(alpha, beta, z):
    return alpha * ((1+z)**2 - 6.565) + beta

def parametrize_bias(tracer, tomo_bin):
    '''
    Returns the alpha and bias models for the magnification correction.
    These are the models used in the HSC WL-photoz tomographic analysis.
    '''
    # -------------------
    # photo-z bias model
    bias_model_p = lambda z: (1+z)**0.917
    # tomographic bins. These measurements are pretty rough.
    match tomo_bin:
        case 1:
            alpha_model_p = lambda z: -0.990
        case 2:
            alpha_model_p = lambda z: -0.701
        case 3:
            alpha_model_p = lambda z: -0.369
        case 4:
            alpha_model_p = lambda z: -0.065
        case _:
            raise ValueError(f"Unknown tomographic bin: {tomo_bin}. Must be one of [1, 2, 3, 4]")

    # -------------------
    # spectroscopic bias model
    match tracer:
        # Galaxy bias : 
        # BGS_ANY: alpha = 0.342 ± 0.012, beta = 2.812 ± 0.059
        # LRG: alpha = 0.332 ± 0.008, beta = 3.245 ± 0.029
        # ELG_LOPnotqso: alpha = 0.197 ± 0.006, beta = 1.354 ± 0.012
        # QSO: alpha = 0.271 ± 0.008, beta = 2.285 ± 0.017
        case 'BGS_ANY':
            pz_BGS = np.array([0.211, 0.352])
            alpha_bgs = 2.5*np.array([0.81, 0.80])-1
            interpolated_BGS = interp1d(
                pz_BGS,
                alpha_bgs,
                bounds_error=False,
                fill_value='extrapolate'
            )
            alpha_model_s = lambda z: interpolated_BGS(z)
            bias_model_s = lambda z: spectroscopic_bias_model(
                alpha=0.342,
                beta=2.812,
                z=z
            )
        case 'LRG':
            pz_cuts_south_LRG = np.array([0.4, 0.47, 0.54, 0.6265, 0.713, 0.7865, 0.86, 0.92, 1.02])
            pz_cuts_north_LRG = np.array([0.4, 0.4725, 0.545, 0.632, 0.719, 0.785, 0.851, 0.92, 1.024])
            pz_cuts_combined_LRG = (pz_cuts_north_LRG + pz_cuts_south_LRG) / 2

            combined_s_LRG     = np.array([1.008, 0.954, 0.988, 1.040, 1.047, 0.999, 0.957, 0.914, 1.078])
            combined_s_LRG_err = np.array([0.007, 0.027, 0.025, 0.021, 0.018, 0.021, 0.017, 0.018, 0.020])
            interpolated_lrg = interp1d(
                pz_cuts_combined_LRG, 
                combined_s_LRG, 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            alpha_model_s = lambda z: 2.5*interpolated_lrg(z)-1
            bias_model_s = lambda z: spectroscopic_bias_model(
                alpha=0.332,
                beta=3.245,
                z=z
            )
        case 'ELG_LOPnotqso' | 'ELGnotqso':
            alpha_ELG = [1.7223330925515012, 2.1021482852604767]
            alpha_ELG_err = [0.009850409808008864, 0.010948105366427699]
            interpolated_ELG = interp1d(
                [(0.75+1.15)/2, (1.15+1.55)/2],
                alpha_ELG,
                bounds_error=False, 
                fill_value='extrapolate'
            )
            alpha_model_s = lambda z: interpolated_ELG(z) # np.sum(alpha_ELG)/2
            bias_model_s = lambda z: spectroscopic_bias_model(
                # this has issues with weights. maybe we should be using the parameters 
                # from Edmond's bias model.
                #alpha=0.197,
                #beta=1.354,
                # Edmond's bias parameters
                alpha=0.153,
                beta=1.541, 
                z=z
            )
        case 'QSO':
            # https://arxiv.org/pdf/2506.22416v1
            pz_qso_edges = np.array([0.8, 2.1, 2.5, 3.5])
            pz_qso = [1.44, 2.27, 2.75]
            qso_mag = 2.5*np.array([0.099, 0.185, 0.244])-1
            interpolated_QSO = interp1d(
                pz_qso, 
                qso_mag, 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            alpha_model_s = lambda z: interpolated_QSO(z)
            bias_model_s = lambda z: spectroscopic_bias_model(
                alpha=0.271,
                beta=2.285,
                z=z
            )
        case _:
            raise ValueError(f"Unknown tracer: {tracer}. Must be one of ['BGS_ANY', 'ELG_LOPnotqso', 'QSO', 'LRG']")
        
    return alpha_model_p, alpha_model_s, bias_model_p, bias_model_s

def magnification_coefficients(
        zi_ind : int, 
        zvalues : np.ndarray,
        alpha_model_p : callable, 
        alpha_model_s : callable, 
        bias_model_p : callable, 
        bias_model_s : callable, 
        w_dm_values : np.ndarray = None,
        contribution : str = 'all'
        ) -> np.ndarray:
    '''
    Computes the magnification correction coefficients for a given redshift index.

    Parameters
    ----------
    zi_ind : int
        The index of the redshift bin to compute the magnification correction for.
    zvalues : np.ndarray
        The redshift values corresponding to the n(z) values.
    alpha_model_p : callable
        The alpha model for the photometric tracer.
    alpha_model_s : callable
        The alpha model for the spectroscopic tracer.
    bias_model_p : callable
        The bias model for the photometric tracer.
    bias_model_s : callable
        The bias model for the spectroscopic tracer.
    w_dm_values : np.ndarray, optional
        The dark matter correlation function values at each redshift.
    contribution : str or list, optional
        The contribution(s) to include: 'uD', 'Du', 'DD', or 'all'.
        
    Returns
    -------
    np.ndarray
        The magnification correction coefficients.
    '''
    assert zi_ind < len(zvalues), "zi_ind must be less than the length of zvalues"
    assert zi_ind >= 0, "zi_ind must be non-negative"
    if isinstance(contribution, str):
        if contribution == 'all':
            contribution = ['uD', 'Du', 'DD']
        else:
            contribution = [contribution]
    assert all(c in ['uD', 'Du', 'DD'] for c in contribution), (
        "contribution must be 'uD', 'Du', 'DD' or 'all'"
    )
    if w_dm_values is None:
        raise ValueError(
            "w_dm_values must be provided to compute the magnification correction"
        )

    # preload the cosmological parameters
    _c = 299792.458  # speed of light in km/s
    _H0 = COSMO_astropy.H0.value # Hubble constant in km/s/Mpc
    _Om0 = COSMO_astropy.Om0 # matter density parameter
    _H = COSMO_astropy.H # Hubble parameter at redshift z in km/s/Mpc (NOTE: is callable)
    cosmofactor = (3 * _H0 **2 * _Om0 / _c)
    dz = np.mean(np.diff(zvalues))  # mean redshift interval

    zi = zvalues[zi_ind]

    def _Dn_ij(zi, zj):
        cosmotransverse = ((chi_ccl(zj)-chi_ccl(zi))/chi_ccl(zj))*chi_ccl(zi)
        return cosmofactor * ((1 + zi) / _H(zi).value) * cosmotransverse * dz
    
    magnification = np.zeros_like(zvalues)

    mag1_const = alpha_model_s(zi)/(bias_model_p(zi)*bias_model_s(zi))
    mag2_const = 1 / bias_model_s(zi)

    # order : spectroscopic x photometric
    for zj_ind, zj in enumerate(zvalues):
        # magnification x galaxy contribution (magnification from the spectroscopic tracer)
        if zj_ind < zi_ind and 'uD' in contribution:
            Dn_ji = _Dn_ij(zj, zi)
            magnification[zj_ind] = (
               mag1_const * bias_model_p(zj) * Dn_ji * w_dm_values[zj_ind] / w_dm_values[zi_ind]
            )
        # galaxy x galaxy contribution
        elif zj_ind == zi_ind and 'DD' in contribution:
            magnification[zj_ind] = 1
        # galaxy x magnification contribution (magnification from the photometric tracer)
        elif zj_ind > zi_ind and 'Du' in contribution:
            Dn_ij = _Dn_ij(zi, zj)
            magnification[zj_ind] = (
                mag2_const * alpha_model_p(zj) * Dn_ij
            )

    return magnification

def solve_magnification(
        meas,
        scale_cut,
        tracer,
        tomo_bin,
        zvalues,
        return_matrices=False,
    ):
    # extract from tuple
    meas_vals, meas_err = meas

    # first, compute w_dm_values for all redshifts
    rp_vals = np.linspace(scale_cut[0], scale_cut[-1], 101)  # in h^-1 Mpc
    print(f'Computing w_dm for {len(zvalues)} redshifts and {len(rp_vals)} rp values...')

    # precompute the angular dark matter correlation function contribution first
    w_dm_values = np.array([
        w_dm(rp_vals, z, integrate=True) 
        for z in zvalues
    ])

    # make the magnification matrix
    print(f'Computing magnification matrix for {len(zvalues)} redshifts...')
    
    # obtain bias, alpha models (parametrize bias has them hardcoded)
    alpha_p, alpha_s, bias_p, bias_s = parametrize_bias(tracer=tracer, tomo_bin=tomo_bin)
    
    Mag = np.array([
        magnification_coefficients(
            zi_ind=i, 
            zvalues=zvalues, 
            alpha_model_p=alpha_p,
            alpha_model_s=alpha_s,
            bias_model_p=bias_p,
            bias_model_s=bias_s,
            w_dm_values=w_dm_values, 
            contribution='all'
            )
        for i in range(len(zvalues))
    ]) 

    # solve the linear system
    print(f'Solving the linear system for {len(zvalues)} redshifts...')
    Mag_inv = np.linalg.inv(Mag)
    npz = Mag_inv @ meas_vals

    # TODO : propagate errors from bias models ?
    dMag = 0

    npz_err = Mag_inv @ meas_err
    #dMag = np.std(Mag, axis=0)  
    #npz_err = np.linalg.solve(Mag, (meas_err + dMag @ npz))
    
    if return_matrices:
        return npz, npz_err, w_dm_values, Mag, dMag
    else:
        return npz, npz_err