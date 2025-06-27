import numpy as np
import pyccl as ccl
import math

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import vstack, Table
from scipy.integrate import quad

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

def w_dm_ang(comov_sep, z, ell_max=10000):
    """
    Compute angular dark matter correlation function w(theta) at a given redshift z.
    
    Parameters
    ----------
    comov_sep : float | list[float] | np.ndarray[float]
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

    angular_vals_deg = hMpc2arcsec(comov_sep, z) / 3600  # convert h^-1 Mpc to arcseconds

    # Create number counts tracer
    tracer = ccl.NumberCountsTracer(
        COSMO_ccl, dndz=(zarr, dndz), bias=(zarr, bias), has_rsd=False
    )

    # Compute Cls
    ells = np.arange(1, ell_max)
    Cls = ccl.angular_cl(COSMO_ccl, tracer, tracer, ells)

    # Compute w(theta)
    wtheta = ccl.correlations.correlation(
        COSMO_ccl, 
        ell=ells, 
        C_ell=Cls,
        theta=angular_vals_deg, 
        type='NN'
    )

    return wtheta

def weights(rp, beta=-1):
    return rp**beta / np.trapz(rp**beta, x=rp)

def chi(z):
    return ccl.comoving_radial_distance(COSMO_ccl, 1/(1+z))

def PNL(l,z):
    return ccl.power.nonlin_power(COSMO_ccl, k=(l+0.5)/chi(z), a=1/(1+z), p_of_k_a='delta_matter:delta_matter')

def Plin(l,z):
    return ccl.power.linear_power(COSMO_ccl, k=(l+0.5)/chi(z), a=1/(1+z), p_of_k_a='delta_matter:delta_matter')

def w_dm(rp_vals, z, integrate=True, ell_max=10000):
    '''
    w_dm expects rp_vals in h^-1 Mpc.
    '''
    rp_vals /= COSMO_astropy.h  # convert to Mpc (1h^-1 Mpc ~1.43 Mpc)
    c_light = 299792.458  # speed of light in km/s
    Ell = range(1, ell_max)

    Hz = COSMO_astropy.H(z).value
    P_delta = [PNL(l, z) for l in Ell]


    theta = rp_vals/chi(z)*360/(2*math.pi) 
    norm = Hz/c_light*(1/chi(z)**2)
    xi_dm=norm*ccl.correlations.correlation(
        COSMO_ccl, ell=Ell, C_ell=P_delta, theta=theta, type='NN', #method='Legendre'
        )
    
    if integrate:
        return np.trapz(
            np.multiply(xi_dm, weights(rp_vals)),
            x=rp_vals
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
        
            

        


def parametrize_magnification():
    '''
    Returns the alpha and bias models for the magnification correction.
    These are the models used in the HSC WL-photoz tomographic analysis.
    '''
    alpha_model_p = lambda z: 2.5*0.004-1
    # LRG / QSO alpha model
    alpha_model_s = lambda z: 2.5*1.5-1 if z < 1.1 else 2.5*0.3-1
    bias_model_p = lambda z: (1+z)**0.5
    bias_model_s = lambda z: 1
    return alpha_model_p, alpha_model_s, bias_model_p, bias_model_s

def mag_coeffs( 
        zindex : int, 
        zvalues : np.ndarray,
        contribution : str = 'all'
    ) -> float:
    '''
    Returns the magnification correction coefficients.
    This is the function used in the HSC WL-photoz tomographic analysis.
    '''
    alpha_model_p, alpha_model_s, bias_model_p, bias_model_s = parametrize_magnification()
    return _magnification_coefficients(
        alpha_model_p, 
        alpha_model_s, 
        bias_model_p, 
        bias_model_s, 
        zindex, 
        zvalues,
        contribution=contribution
    )
    
def _magnification_coefficients( 
        alpha_model_p : callable, 
        alpha_model_s : callable, 
        bias_model_p : callable, 
        bias_model_s : callable, 
        zindex : int, 
        zvalues : np.ndarray,
        contribution : str = 'all'
        ):
    '''
    Computes the magnification correction for a given redshift index and value and cosmology.

    Parameters
    ----------
    cosmology : acosmo
        The cosmology to use for the magnification correction.
    alpha_model_p : callable
        The alpha model for the photometric tracer.
    alpha_model_s : callable
        The alpha model for the spectroscopic tracer.
    bias_model_p : callable
        The bias model for the photometric tracer.
    bias_model_s : callable
        The bias model for the spectroscopic tracer.
    zindex : int
        The index of the redshift bin to compute the magnification correction for.
    zvalues : np.ndarray
        The redshift values corresponding to the n(z) values.
    '''
    assert zindex < len(zvalues), "zindex must be less than the length of zvalues"
    assert zindex >= 0, "zindex must be non-negative"
    if isinstance(contribution, str):
        if contribution == 'all':
            contribution = ['uD', 'Du', 'DD']
        else:
            contribution = [contribution]
    assert all(c in ['uD', 'Du', 'DD'] for c in contribution), (
        "contribution must be 'uD', 'Du', 'DD' or 'all'"
    )

    # preload the cosmological parameters
    _c = 299792.458  # speed of light in km/s
    chi = COSMO_astropy.comoving_transverse_distance # comoving transverse distance in Mpc
    _H0 = COSMO_astropy.H0.value # Hubble constant in km/s/Mpc
    _Om0 = COSMO_astropy.Om0 # matter density parameter
    _H = COSMO_astropy.H # Hubble parameter at redshift zvalues in km/s/Mpc
    cosmofactor = (3 * _H0 **2 * _Om0 / _c)
    dz = np.mean(np.diff(zvalues))  # mean redshift interval

    zi = zvalues[zindex]
    
    def _wDM(z_low, z_high, tracer='all'):
        return get_wDM(
            angular_vals=arcsec2hMpc(1, z_low),
            zbin_edges=[z_low, z_high],
            dndz=np.ones_like([z_low, z_high])
        )

    def _Dn_ij(zi, zj):
        cosmotransverse = ((chi(zj)-chi(zi))/chi(zj))*chi(zi)
        return cosmofactor * ((1 + zi) / _H(zi).value) * cosmotransverse.value * dz
    
    magnification = np.zeros_like(zvalues)

    mag1_const = alpha_model_s(zi)/(bias_model_p(zi)*bias_model_s(zi))
    mag2_const = 1 / bias_model_p(zi)
    for j, zj in enumerate(zvalues):
        if j < zindex and 'uD' in contribution:
            Dn_ji = _Dn_ij(zj, zi)
            magnification[j] = (
               mag1_const * bias_model_p(zj) * Dn_ji
            )
        elif j == zindex and 'DD' in contribution:
            magnification[j] = 1
        elif j > zindex and 'Du' in contribution:
            Dn_ij = _Dn_ij(zi, zj)
            magnification[j] = (
                mag2_const * alpha_model_p(zj) * Dn_ij
            )

    return magnification

def solve_magnification(
        zgrid
    ):
    pass