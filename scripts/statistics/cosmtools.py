import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

# Define the cosmology model
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)

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
    d_A = cosmo.angular_diameter_distance(z) 
    x = (theta * d_A).to(u.Mpc, u.dimensionless_angles()) 
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
    d_A = cosmo.angular_diameter_distance(z) 
    theta = (x * u.Mpc * cosmo.h / d_A).to(u.arcsec, u.dimensionless_angles())
    return theta.value

def z2dist(z):
    """
    Convert redshift to comoving distance (in h^-1 Mpc).
    
    Parameters:
    -----------

    z: float | list[float] | np.ndarray[float]
        Redshift
    """
    return cosmo.comoving_distance(z).value / cosmo.h