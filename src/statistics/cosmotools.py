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

def magnification_correction( 
        alpha_model_p : callable, 
        alpha_model_s : callable, 
        bias_model_p : callable, 
        bias_model_s : callable, 
        np_z : np.ndarray, 
        zindex : int, 
        zvalues : np.ndarray
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
    np_z : np.ndarray
        The n(z) values for the redshift bins.
    zindex : int
        The index of the redshift bin to compute the magnification correction for.
    zvalues : np.ndarray
        The redshift values corresponding to the n(z) values.
    '''
    
    def _Dn_ij(zi, zj):
        c = 299792.458  # speed of light in km/s
        chi = COSMO_astropy.comoving_transverse_distance
        cosmofactor = (3 * COSMO_astropy.H0.value**2 * COSMO_astropy.Om0.value / (c**2))
        cosmotransverse = ((chi(zi)-chi(zj))/chi(zi))*chi(zj) # todo : include delta_chi_j ? (Gatti. et al.)
        return  cosmofactor * (1+zi) * cosmotransverse # 1+zi = 1/a(zi)
    
    zi = zvalues[zindex]
    magnification = 0
    magnification += np_z[zindex] 
    sum1 = 0
    for j in range(len(np_z)):
        if j > zindex:
            sum1 += np_z[j] * _Dn_ij(COSMO_astropy, zi, zvalues[j])
    sum2 = 0
    for j in range(len(np_z)):
        if j > zindex:
            sum2 += np_z[zindex] * _Dn_ij(COSMO_astropy, zi, zvalues[j])

    magnification += alpha_model_p(zi) * sum1 / bias_model_p(zi)
    magnification += alpha_model_s(zi) * sum2 / bias_model_s(zi)

    return magnification