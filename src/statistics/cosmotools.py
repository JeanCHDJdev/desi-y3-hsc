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


def _get_bias_correction(scale_cut, mode="powerlaw"):
    """
    NOTE : This function is actually returning the wpp correction,
    hence significant differences between scale cuts. One should correct for dark matter
    autocorrelation to recover bias.

    Only the power law `g1 * (1 + z) ** g2` is expressible as this 4-tuple. The degree-2
    polynomial has three (strongly correlated) parameters, so it goes through
    `get_photo_bias_model` instead, which carries the full covariance.
    """
    if mode != "powerlaw":
        raise ValueError(
            f"_get_bias_correction only handles the powerlaw, got mode={mode!r}. "
            "Use get_photo_bias_model(scale_cut, mode=...) for the other models."
        )
    if scale_cut == [0.3, 3.0]:
        # with DR1 ELGs
        # g1 = 0.409
        # delta_g1 = 0.006
        # g2 = 0.466
        # delta_g2 = 0.023
        # without DR1 ELGs
        # alpha = 0.41590054607117416 ± 0.0038119837752912016
        # beta  = 0.4304153126089022 ± 0.013780481431903305
        
        ## old version, pre correction
        #g1 = 0.41590054607117416
        #delta_g1 = 0.0038119837752912016
        #g2 = 0.4304153126089022
        #delta_g2 = 0.013780481431903305

        ## new version, post correction and error additions
        g1 = 0.47585346685220986
        delta_g1 = 0.007846294054192455
        g2 = 0.2755655035016896
        delta_g2 = 0.024161451950752217
    elif scale_cut == [1, 5]:
        # with DR1 ELGs
        # g1 = 0.295
        # delta_g1 = 0.007
        # g2 = 0.565
        # delta_g2 = 0.036
        # without DR1 ELGs
        # alpha = 0.3074501687394755 ± 0.0042949661424663745
        # beta  = 0.5117933464025347 ± 0.02022743194083983

        ## old version, pre correction
        #g1 = 0.3074501687394755
        #delta_g1 = 0.0042949661424663745
        #g2 = 0.5117933464025347
        #delta_g2 = 0.02022743194083983

        ## new version, post correction and error additions
        g1 = 0.3431763273863235
        delta_g1 = 0.009051681533269088
        g2 = 0.3914273704554439 
        delta_g2 = 0.03765699355702185
    else:
        raise ValueError(
            f"Scale cut {scale_cut} not recognized. Available options are [.3, 3.] and [1, 5]."
        )
    # g1*(1+z)**g2 with associated errorbars if necessary
    return g1, delta_g1, g2, delta_g2


#: default version of the cached photometric-bias fits (see src/statistics/biasfit.py)
PHOTO_BIAS_VERSION = "v_1p1"


def get_photo_bias_model(
    scale_cut, mode="powerlaw", version=None, use_covariance=None, root=None
):
    """
    Return a `biasfit.PhotoBiasModel` for the photometric bias correction.

    Parameters
    ----------
    scale_cut : list
        [rp_min, rp_max] in h^-1 Mpc.
    mode : {'powerlaw', 'polynomial'}
    version : str, optional
        Version of the cached fit. Defaults to `PHOTO_BIAS_VERSION`.
    use_covariance : bool, optional
        Whether to propagate the full parameter covariance. Defaults to False.
    """
    import src.statistics.biasfit as biasfit

    if use_covariance is None:
        use_covariance = mode != "powerlaw"
    return biasfit.PhotoBiasModel.from_cache(
        scale_cut,
        version if version is not None else PHOTO_BIAS_VERSION,
        mode=mode,
        root=root,
        use_covariance=use_covariance,
    )


def parametrize_bias(tracer, tomo_bin, wdm, scale_cut, bias_mode="powerlaw"):
    """
    Returns the alpha and bias models for the magnification correction.
    These are the models used in the HSC WL-photoz tomographic analysis.
    """
    # --------------------------------------
    # galaxy bias for the photometric tracer.
    # small tomographic bins are 0.1 in size
    dzp = 0.1
    # wdm is passed as precomputed over the tomographic bins
    if bias_mode == "powerlaw":
        # a, b are g1, g2 and _, _ are the errors on these
        a, _, b, _ = _get_bias_correction(scale_cut=scale_cut, mode="powerlaw")
        bias_model_p = lambda z: a * (1 + z) ** b * np.sqrt(dzp / wdm(z))
    else:
        _bmodel = get_photo_bias_model(scale_cut=scale_cut, mode=bias_mode)
        bias_model_p = lambda z: _bmodel.value(z) * np.sqrt(dzp / wdm(z))

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
    alpha_offset_p: float = 0.0,
    alpha_offset_s: float = 0.0,
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
    alpha_offset_p, alpha_offset_s : float, optional
        Additive shift of the photometric / spectroscopic magnification coefficients,
        alpha -> alpha(z) + alpha_offset. 0.0 (the default) leaves them untouched.
        Passed in rather than drawn here so a realization stays coherent across every
        row of the magnification matrix (see `solve_magnification`).

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

    mag1_const = (alpha_model_s(zi) + alpha_offset_s) / (
        bias_model_p(zi) * bias_model_s(zi)
    )
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
            mag_contribution = alpha_model_p(zj) + alpha_offset_p
            magnification[zj_ind] = mag2_const * mag_contribution * Dn_ij

    return magnification


def draw_alpha_offsets(mag_sigma, rng=None, size=None):
    """
    Draw additive shifts `N(0, mag_sigma)` for the magnification coefficients alpha.

    `mag_sigma` is an *absolute* error: 0.1 is a 1-sigma error of 0.1 on alpha itself,
    independent of how large alpha is. That is the form alpha is measured in -- from the
    slope of the number counts, with absolute quoted errors.

    Draw once per realization and reuse across the whole tomographic bin / tracer loop:
    the coefficients are a systematic, not a per-redshift noise term.

    Parameters
    ----------
    mag_sigma : float
        Absolute 1-sigma error on alpha. 0 returns exactly 0.0 (unperturbed).
    rng : np.random.Generator, optional
    size : int or tuple, optional
        Shape of the draw. None returns a scalar.
    """
    if not mag_sigma:
        return 0.0 if size is None else np.zeros(size)
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0.0, mag_sigma, size=size)


def solve_magnification(
    meas,
    scale_cut,
    tracer,
    tomo_bin,
    zvalues,
    return_matrices=False,
    alpha_offset_p=0.0,
    alpha_offset_s=0.0,
    bias_mode="powerlaw",
    w_dm_values=None,
):
    """
    Invert the magnification matrix to recover the magnification-corrected n(z).

    Parameters
    ----------
    meas : tuple
        (n(z) values, n(z) errors) before magnification correction.
    alpha_offset_p, alpha_offset_s : float, optional
        Additive shift of the photometric / spectroscopic magnification coefficients,
        alpha -> alpha(z) + alpha_offset; 0.0 (the default) is the unperturbed,
        fiducial analysis. Use `draw_alpha_offsets` for a perturbed realization.
    bias_mode : {'powerlaw', 'polynomial'}, optional
        Model used for the photometric galaxy bias entering the magnification kernel.
    w_dm_values : np.ndarray, optional
        Pre-computed w_dm at `zvalues`. It only depends on the scale cut and the
        redshift grid, so caching it across realizations avoids
        redoing the (expensive) CCL calls for every realization.
    """
    meas_vals, meas_err = meas

    rp_vals = np.linspace(scale_cut[0], scale_cut[-1], 101)  # in h^-1 Mpc
    # precompute the angular dark matter correlation function contribution first
    if w_dm_values is None:
        w_dm_values = np.array([w_dm(rp_vals, z, integrate=True) for z in zvalues])
    else:
        w_dm_values = np.asarray(w_dm_values)
        if w_dm_values.shape[0] != len(zvalues):
            raise ValueError("w_dm_values must have one entry per value in zvalues.")
    w_dm_interp = interp1d(zvalues, w_dm_values, axis=0, fill_value="extrapolate")

    # obtain bias, alpha models (parametrize bias has them hardcoded)
    alpha_p, alpha_s, bias_p, bias_s = parametrize_bias(
        tracer=tracer,
        tomo_bin=tomo_bin,
        wdm=w_dm_interp,
        scale_cut=scale_cut,
        bias_mode=bias_mode,
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
                alpha_offset_p=alpha_offset_p,
                alpha_offset_s=alpha_offset_s,
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
