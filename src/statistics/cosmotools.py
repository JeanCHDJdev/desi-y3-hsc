import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18, FlatLambdaCDM

import pyccl as ccl

# Define the cosmology model

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
    x /= COSMO_astropy.h 
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
    rp_hMpc = rp * COSMO_astropy.h  # Convert h^-1 Mpc to Mpc
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

def get_wDM(angular_bins, zbin_edges, dndz):
    '''
    Using CCL (Core Cosmology Library) to estimate wCM (the dark matter angular correlation function).
    NOTE : CCL uses Limber approximation to compute w(theta) from C_ell and 
    halofit model for the non-linear power spectrum.
    '''
    # instaniate tracer
    dndz_zbins = 0.5 * (zbin_edges[:-1] + zbin_edges[1:])
    #if bias_with_growthfactor:
        # bias with growth factor (0.95/D(z) growth factor)
        #bias = (dndz_zbins, 0.95 / (1 / (1 + dndz_zbins)))
    #else:
    bias = (dndz_zbins, np.ones_like(dndz_zbins))
    tracer = ccl.NumberCountsTracer(
        COSMO_ccl,
        has_rsd=False,
        dndz=(dndz_zbins, dndz),  # normalized PDF
        bias=bias
    )

    # angular power spectrum from C_ells
    ell = np.linspace(0.1, 30000, 5000)
    cl = ccl.angular_cl(COSMO_ccl, tracer, tracer, ell)

    # w(theta) from C_ells using Limber approximation
    wtheta = ccl.correlation(
        COSMO_ccl, 
        ell=ell, 
        C_ell=cl, 
        theta=angular_bins,
        type='NN'
        )

    return wtheta

def parametrize_magnification():
    '''
    Returns the alpha and bias models for the magnification correction.
    These are the models used in the HSC WL-photoz tomographic analysis.
    '''
    alpha_model_p = lambda z: -1
    alpha_model_s = lambda z: 1
    bias_model_p = lambda z: 1
    bias_model_s = lambda z: 1
    return alpha_model_p, alpha_model_s, bias_model_p, bias_model_s

def mag_coeffs( 
        zindex : int, 
        zvalues : np.ndarray,
        contribution : str = 'all'
    ) -> float:
    '''
    Returns the magnification correction function.
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
        contribution : str = 'both'
        ):
    if isinstance(contribution, str):
        if contribution == 'all':
            contribution = ['uD', 'Du', 'DD']
        else:
            contribution = [contribution]
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

    _c = 299792.458  # speed of light in km/s
    chi = COSMO_astropy.comoving_transverse_distance # comoving transverse distance in Mpc
    _H0 = COSMO_astropy.H0.value # Hubble constant in km/s/Mpc
    _Om0 = COSMO_astropy.Om0 # matter density parameter
    cosmofactor = (3 * _H0 **2 * _Om0 / _c)

    zi = zvalues[zindex]

    def _Dn_ij(zi, zj):
        cosmotransverse = ((chi(zj)-chi(zi))/chi(zj))*chi(zi)
        return cosmofactor * (1 + zi)**2 * cosmotransverse.value
    
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