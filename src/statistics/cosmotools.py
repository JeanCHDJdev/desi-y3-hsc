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
cosmo = FlatLambdaCDM(
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
    d_pm = cosmo.comoving_transverse_distance(z) 
    x = (theta * d_pm).to(u.Mpc, u.dimensionless_angles()) 
    x /= cosmo.h 
    return x.value

def hMpc2arcsec(x, z):
    """
    Convert transverse comoving separation (in h^-1 Mpc) 
    to angular separation (in arcseconds).
    
    Parameters:
    x: float
        Transverse separation in h^-1 Mpc
    z: float
        Redshift
    """
    d_pm = cosmo.comoving_transverse_distance(z)
    theta = (x * u.Mpc / d_pm).to(u.arcsec, u.dimensionless_angles())
    theta = theta * cosmo.h 
    return theta.value # angular separation in arcseconds

def z2dist(z):
    """
    Convert redshift to comoving distance (in h^-1 Mpc).
    
    Parameters:
    -----------

    z: float | list[float] | np.ndarray[float]
        Redshift
    """
    return np.array(
        cosmo.comoving_distance(z).value / cosmo.h, 
        dtype=float
        )

def get_wCM(angular_bins, zbin_edges, zbin_counts):
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
        dndz=(dndz_zbins, zbin_counts / np.trapz(zbin_counts, dndz_zbins)),  # normalized PDF
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
        theta=angular_bins
        )

    return wtheta