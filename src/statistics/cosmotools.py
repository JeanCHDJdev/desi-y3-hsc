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
COSMO_ccl = ccl.Cosmology(Omega_c=omega_c, Omega_b=omega_b, h=h, sigma8=sigma8, n_s=n_s)
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
        COSMO_astropy.comoving_distance(z).value / COSMO_astropy.h, dtype=float
    )


def weights(rp, beta=-1):
    return rp**beta / np.trapz(rp**beta, x=rp)


def chi_ccl(z):
    return ccl.comoving_radial_distance(COSMO_ccl, a=1 / (1 + z))


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
        COSMO_ccl, tracer, tracer, ells, p_of_k_a="delta_matter:delta_matter"
    )

    # Compute w(theta)
    wtheta = ccl.correlations.correlation(
        COSMO_ccl, ell=ells, C_ell=Cls, theta=angular_vals_deg, type="NN"
    )
    if integrate:
        # Integrate w(theta) over the angular separation
        w = np.trapz(np.multiply(wtheta, weights(angular_vals_deg)), x=angular_vals_deg)
        return w
    else:
        return wtheta


def p_mat_nonlin(l, z):
    return ccl.power.nonlin_power(
        COSMO_ccl,
        k=(l + 0.5) / chi_ccl(z),
        a=1 / (1 + z),
        p_of_k_a="delta_matter:delta_matter",
    )


def p_mat_lin(l, z):
    return ccl.power.linear_power(
        COSMO_ccl,
        k=(l + 0.5) / chi_ccl(z),
        a=1 / (1 + z),
        p_of_k_a="delta_matter:delta_matter",
    )


def w_dm(rp_vals, z, integrate=True, ell_max=12000):
    """
    w_dm expects rp_vals in h^-1 Mpc.
    """
    rp_vals_Mpc = rp_vals / COSMO_astropy.h  # convert to Mpc (1h^-1 Mpc ~1.43 Mpc)
    c_light = 299792.458  # speed of light in km/s
    Ell = range(1, ell_max)

    Hz = COSMO_astropy.H(z).value
    P_delta = [p_mat_nonlin(l, z) for l in Ell]

    theta = rp_vals_Mpc / chi_ccl(z) * 360 / (2 * math.pi)
    norm = Hz / c_light * (1 / chi_ccl(z) ** 2)
    xi_dm = norm * ccl.correlations.correlation(
        COSMO_ccl, ell=Ell, C_ell=P_delta, theta=theta, type="NN", method="Legendre"
    )

    if integrate:
        return np.trapz(np.multiply(xi_dm, weights(rp_vals_Mpc)), x=rp_vals_Mpc)
    else:
        return xi_dm


def redshift_distribution(bounds, tracer, discretization=100):
    centers = []
    if isinstance(bounds, (list, np.ndarray)):
        if isinstance(bounds[0], (float, int)):
            assert len(bounds) == 2, "Bounds must be in the form [z_min, z_max]"
            bounds = [tuple(bounds)]
        elif isinstance(bounds[0], (list, tuple)):
            assert all(
                len(b) == 2 for b in bounds
            ), "Bounds must be in the form [[z_min, z_max], ...]"
            # convert to tuples for consistency
            bounds = [tuple(b) for b in bounds]
        else:
            raise TypeError(
                "Bounds must be a list of two floats or a list of lists/tuples of two floats"
            )
    else:
        raise TypeError("Bounds must be a list or numpy array")
    for b in bounds:
        assert (
            b[0] < b[1]
        ), "Bounds must be in the form [z_min, z_max] with z_min < z_max"
        centers.append(0.5 * (b[0] + b[1]))
    dz = np.diff(centers)
    assert all(
        np.isclose(d, dz[0]) for d in dz
    ), "Bounds must be equally spaced in redshift"
    dz = dz[0]  # use the first dz as the common interval

    assert (
        isinstance(discretization, int) and discretization > 0
    ), "Discretization must be a positive integer"
    assert tracer in ["all", "BGS_ANY", "LRG", "ELG_LOPnotqso", "QSO"]

    zdata = None
    files = {
        t: np.array(
            [cf.fetch_desi_files(t, randoms=False, cap=cap) for cap in ["NGC", "SGC"]]
        ).flatten()
        for t in ["BGS_ANY", "LRG", "ELG_LOPnotqso", "QSO"]
    }
    if tracer == "all":
        # get the redshift file to get the distribution
        allf = []
        for k, f in files.items():
            allf.extend(f)
    else:
        # get the redshift file for the specific tracer
        allf = files[tracer]
    zdata = vstack([Table.read(f) for f in allf])

    return zdata["Z"].data


def spec_bias(z, tracer="QSO", return_coeffs=False):
    """
    Bias model for the different DESI tracer (measured from DR2 data).
    Credit : E. Chaussidon, DESI Collaboration, private communication.
    """
    params = {
        # commented values are measurements given by DESI CAI, values in parentheses
        # are the values used in this work.
        "BGS_BRIGHT-21.35": (0.60646037, 0.52389492),  # (0.606, 0.524),
        "LRG": (0.23553567, 1.3458994),  # (0.236, 1.346),
        "ELG_LOPnotqso": (0.15066781, 0.59463735),  # (0.151, 0.595),
        "ELG": (0.15487521, 0.59464828),  # (0.155, 0.595),
        "QSO": (0.25207547, 0.71020952),  # (0.252, 0.710)
    }

    if tracer in params:
        alpha, beta = params[tracer]
    else:
        print(f"Tracer: {tracer} is not ready...")

    if return_coeffs:
        return alpha, beta
    else:
        # Laurent+2017 on QSOs
        return alpha * (1 + z) ** 2 + beta


def _get_bias_correction(scale_cut):
    """
    NOTE : This function is actually returning the wpp correction,
    hence significant differences between scale cuts. One should correct for dark matter
    autocorrelation to recover bias.
    """
    if scale_cut == [0.3, 3.0]:
        # with DR1 ELGs
        # g1 = 0.409
        # delta_g1 = 0.006
        # g2 = 0.466
        # delta_g2 = 0.023
        # without DR1 ELGs
        # alpha = 0.41209728134282336 ± 0.0039027072216504567
        # beta  = 0.4516232747100852 ± 0.014501757769134687
        g1 = 0.41209728134282336
        delta_g1 = 0.0039027072216504567
        g2 = 0.4516232747100852
        delta_g2 = 0.014501757769134687
    elif scale_cut == [1, 5]:
        # with DR1 ELGs
        # g1 = 0.295
        # delta_g1 = 0.007
        # g2 = 0.565
        # delta_g2 = 0.036
        # without DR1 ELGs
        # alpha = 0.30532131486961517 ± 0.004379251403019057
        # beta  = 0.5322244999391279 ± 0.021227813479119593
        g1 = 0.30532131486961517
        delta_g1 = 0.004379251403019057
        g2 = 0.5322244999391279
        delta_g2 = 0.021227813479119593
    else:
        raise ValueError(
            f"Scale cut {scale_cut} not recognized. Available options are [.3, 3.] and [1, 5]."
        )
    # g1*(1+z)**g2 with associated errorbars if necessary
    return g1, delta_g1, g2, delta_g2


def parametrize_bias(tracer, tomo_bin, wdm, scale_cut):
    """
    Returns the alpha and bias models for the magnification correction.
    These are the models used in the HSC WL-photoz tomographic analysis.
    """
    # --------------------------------------
    # galaxy bias for the photometric tracer. we note that a, b are g1, g2 and _, _ are the errors on these
    a, _, b, _ = _get_bias_correction(scale_cut=scale_cut)
    # small tomographic bins are 0.1 in size
    dzp = 0.1
    # wdm is passed as precomputed over the tomographic bins
    bias_model_p = lambda z: a * (1 + z) ** b * np.sqrt(dzp / wdm(z))

    # --------------------------------------
    # magnification bias for the photometric tracer
    match tomo_bin:
        case 1:
            alpha_model_p = lambda z: -0.996  # -0.990
        case 2:
            alpha_model_p = lambda z: -0.837  # -0.701
        case 3:
            alpha_model_p = lambda z: -0.646  # -0.369
        case 4:
            alpha_model_p = lambda z: -0.485  # -0.065
        case _:
            raise ValueError(
                f"Unknown tomographic bin: {tomo_bin}. Must be one of [1, 2, 3, 4]"
            )

    # --------------------------------------
    # galaxy and magnification bias for the spectroscopic tracer
    match tracer:
        case "BGS_ANY":
            pz_BGS = np.array([0.211, 0.352])
            alpha_bgs = 2.5 * np.array([0.81, 0.80]) - 1
            interpolated_BGS = interp1d(
                pz_BGS, alpha_bgs, bounds_error=False, fill_value="extrapolate"
            )
            alpha_model_s = lambda z: interpolated_BGS(z)
            bias_model_s = lambda z: spec_bias(z=z, tracer="BGS_BRIGHT-21.35")
        case "LRG":
            pz_cuts_south_LRG = np.array(
                [0.4, 0.47, 0.54, 0.6265, 0.713, 0.7865, 0.86, 0.92, 1.02]
            )
            pz_cuts_north_LRG = np.array(
                [0.4, 0.4725, 0.545, 0.632, 0.719, 0.785, 0.851, 0.92, 1.024]
            )
            pz_cuts_combined_LRG = (pz_cuts_north_LRG + pz_cuts_south_LRG) / 2

            combined_s_LRG = np.array(
                [1.008, 0.954, 0.988, 1.040, 1.047, 0.999, 0.957, 0.914, 1.078]
            )
            combined_s_LRG_err = np.array(
                [0.007, 0.027, 0.025, 0.021, 0.018, 0.021, 0.017, 0.018, 0.020]
            )
            interpolated_lrg = interp1d(
                pz_cuts_combined_LRG,
                combined_s_LRG,
                bounds_error=False,
                fill_value="extrapolate",
            )
            alpha_model_s = lambda z: 2.5 * interpolated_lrg(z) - 1
            bias_model_s = lambda z: spec_bias(z=z, tracer="LRG")
        case "ELG_LOPnotqso" | "ELGnotqso":
            alphas = [1.258148799455872, 1.5334325766616752]
            alphas_error = [0.01081768382236435, 0.011938464095389137]
            alpha_ELG = 2.5 / np.log(10) * np.array(alphas) - 1
            interpolated_ELG = interp1d(
                [(0.75 + 1.15) / 2, (1.15 + 1.55) / 2],
                alpha_ELG,
                bounds_error=False,
                fill_value="extrapolate",
            )
            alpha_model_s = lambda z: interpolated_ELG(z)  # np.sum(alpha_ELG)/2
            if tracer == "ELG_LOPnotqso":
                bias_model_s = lambda z: spec_bias(z=z, tracer="ELG_LOPnotqso")
            else:
                bias_model_s = lambda z: spec_bias(z=z, tracer="ELG")
        case "QSO":
            # https://arxiv.org/pdf/2506.22416v1
            pz_qso_edges = np.array([0.8, 2.1, 2.5, 3.5])
            pz_qso = [1.44, 2.27, 2.75]
            qso_mag = 2.5 * np.array([0.099, 0.185, 0.244]) - 1
            interpolated_QSO = interp1d(
                pz_qso, qso_mag, bounds_error=False, fill_value="extrapolate"
            )
            alpha_model_s = lambda z: interpolated_QSO(z)
            bias_model_s = lambda z: spec_bias(z=z, tracer="QSO")
        case _:
            raise ValueError(
                f"Unknown tracer: {tracer}. Must be one of ['BGS_ANY', 'ELG_LOPnotqso', 'QSO', 'LRG']"
            )

    return alpha_model_p, alpha_model_s, bias_model_p, bias_model_s


def magnification_coefficients(
    zi_ind: int,
    zvalues: np.ndarray,
    alpha_model_p: callable,
    alpha_model_s: callable,
    bias_model_p: callable,
    bias_model_s: callable,
    w_dm_values: np.ndarray = None,
    contribution: str = "all",
) -> np.ndarray:
    """
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
        The contribution(s) to incluge: 'ug', 'gu', 'gg', or 'all'.

    Returns
    -------
    np.ndarray
        The magnification correction coefficients.
    """
    assert zi_ind < len(zvalues), "zi_ind must be less than the length of zvalues"
    assert zi_ind >= 0, "zi_ind must be non-negative"
    if isinstance(contribution, str):
        if contribution == "all":
            contribution = ["ug", "gu", "gg"]
        else:
            contribution = [contribution]
    assert all(
        c in ["ug", "gu", "gg"] for c in contribution
    ), "contribution must be 'ug', 'gu', 'gg' or 'all'"
    if w_dm_values is None:
        raise ValueError(
            "w_dm_values must be provided to compute the magnification correction"
        )

    # preload the cosmological parameters
    _c = 299792.458  # speed of light in km/s
    _H0 = COSMO_astropy.H0.value  # Hubble constant in km/s/Mpc
    _Om0 = COSMO_astropy.Om0  # matter density parameter
    _H = (
        COSMO_astropy.H
    )  # Hubble parameter at redshift z in km/s/Mpc (NOTE: is callable)
    cosmofactor = 3 * _H0**2 * _Om0 / _c
    dz = np.mean(np.diff(zvalues))  # mean redshift interval (assumes uniform binning)

    zi = zvalues[zi_ind]

    def _Dn_ij(zi, zj):
        cosmotransverse = ((chi_ccl(zj) - chi_ccl(zi)) / chi_ccl(zj)) * chi_ccl(zi)
        return cosmofactor * ((1 + zi) / _H(zi).value) * cosmotransverse * dz

    magnification = np.zeros_like(zvalues)

    mag1_const = alpha_model_s(zi) / (bias_model_p(zi) * bias_model_s(zi))
    mag2_const = 1 / bias_model_p(zi)

    # order : spectroscopic x photometric
    for zj_ind, zj in enumerate(zvalues):
        # magnification x galaxy contribution (magnification from the spectroscopic tracer)
        if zj_ind < zi_ind and "ug" in contribution:
            Dn_ji = _Dn_ij(zj, zi)
            magnification[zj_ind] = (
                mag1_const
                * bias_model_p(zj)
                * Dn_ji
                * w_dm_values[zj_ind]
                / w_dm_values[zi_ind]
            )
        # galaxy x galaxy contribution
        elif zj_ind == zi_ind and "gg" in contribution:
            magnification[zj_ind] = 1
        # galaxy x magnification contribution (magnification from the photometric tracer)
        elif zj_ind > zi_ind and "gu" in contribution:
            Dn_ij = _Dn_ij(zi, zj)
            magnification[zj_ind] = mag2_const * alpha_model_p(zj) * Dn_ij

    return magnification


def solve_magnification(
    meas,
    scale_cut,
    tracer,
    tomo_bin,
    zvalues,
    return_matrices=False,
):
    meas_vals, meas_err = meas

    rp_vals = np.linspace(scale_cut[0], scale_cut[-1], 101)  # in h^-1 Mpc
    # precompute the angular dark matter correlation function contribution first
    w_dm_values = np.array([w_dm(rp_vals, z, integrate=True) for z in zvalues])
    w_dm_interp = interp1d(zvalues, w_dm_values, axis=0, fill_value="extrapolate")

    # obtain bias, alpha models (parametrize bias has them hardcoded)
    alpha_p, alpha_s, bias_p, bias_s = parametrize_bias(
        tracer=tracer, tomo_bin=tomo_bin, wdm=w_dm_interp, scale_cut=scale_cut
    )

    Mag = np.array(
        [
            magnification_coefficients(
                zi_ind=i,
                zvalues=zvalues,
                alpha_model_p=alpha_p,
                alpha_model_s=alpha_s,
                bias_model_p=bias_p,
                bias_model_s=bias_s,
                w_dm_values=w_dm_values,
                contribution="all",
            )
            for i in range(len(zvalues))
        ]
    )

    # solve the linear system
    Mag_inv = np.linalg.inv(Mag)
    # Mag is assumed to be perfectly known, so no error propagation
    dMag = 0

    npz = Mag_inv @ meas_vals
    npz_err = Mag_inv @ meas_err

    if return_matrices:
        return npz, npz_err, w_dm_values, Mag, dMag
    else:
        return npz, npz_err
