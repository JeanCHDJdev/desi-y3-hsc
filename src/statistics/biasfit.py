"""
Photometric galaxy-bias correction models fitted on :math:`\\sqrt{\\bar\\omega_{pp}}`.

The fiducial analysis uses a power law :math:`\\alpha (1+z)^\\beta`. This module adds a
degree-2 polynomial alternative :math:`\\alpha (1+z)^2 + \\beta (1+z) + \\gamma` and, most
importantly, keeps the **full parameter covariance** of either fit.

Keeping the covariance matters: the three polynomial coefficients are strongly
(anti-)correlated, so propagating only the diagonal of ``pcov`` inflates the error on
the correction by roughly an order of magnitude. See ``refit_photoz_bias`` for the
refit that regenerates the cache, and ``PhotoBiasModel`` for the evaluation interface
used by ``inference.compute_npz``.

The refit needs ``pyccl``/``pycorr`` (``desi`` env). Everything downstream only reads
the cached ``.npz``, so it also works from the ``pymc`` env.
"""

import numpy as np

from pathlib import Path

# --------------------------------------------------------------------------------------
# model definitions
# --------------------------------------------------------------------------------------

MODES = ("powerlaw", "polynomial")

#: number of free parameters per mode
N_PARAMS = {"powerlaw": 2, "polynomial": 3}


def powerlaw_1pz(z, alpha, beta):
    """alpha * (1 + z) ** beta"""
    return alpha * (1 + z) ** beta


def deg2_poly(z, alpha, beta, gamma):
    """alpha * (1 + z) ** 2 + beta * (1 + z) + gamma"""
    return alpha * (1 + z) ** 2 + beta * (1 + z) + gamma


MODEL_FUNCS = {"powerlaw": powerlaw_1pz, "polynomial": deg2_poly}


def _jacobian(mode, z, params):
    """
    d f / d params, shape (len(z), n_params).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if mode == "powerlaw":
        alpha, beta = params
        J = np.empty((z.size, 2))
        J[:, 0] = (1 + z) ** beta
        J[:, 1] = alpha * (1 + z) ** beta * np.log(1 + z)
    elif mode == "polynomial":
        J = np.empty((z.size, 3))
        J[:, 0] = (1 + z) ** 2
        J[:, 1] = 1 + z
        J[:, 2] = 1.0
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}.")
    return J


# --------------------------------------------------------------------------------------
# hardcoded fallbacks (diagonal only -- these are the values inlined in cosmotools)
# --------------------------------------------------------------------------------------

#: {(scale_cut): {mode: (params, param_errors)}}. Kept as a fallback when no cached fit
#: file is available. The covariance is then assumed diagonal, which is a poor
#: approximation for the polynomial.
_HARDCODED = {
    (0.3, 3.0): {
        "powerlaw": (
            (0.47585346685220986, 0.2755655035016896),
            (0.007846294054192455, 0.024161451950752217),
        ),
        "polynomial": (
            (0.0035720046689597684, 0.07096613342496609, 0.41702843216591756),
            (0.016994429547050518, 0.0648486512692185, 0.06010590320271767),
        ),
    },
    (1.0, 5.0): {
        "powerlaw": (
            (0.3431763273863235, 0.3914273704554439),
            (0.009051681533269088, 0.03765699355702185),
        ),
    },
}


def _key(scale_cut):
    return (float(scale_cut[0]), float(scale_cut[-1]))


def scale_cut_tag(scale_cut):
    """
    Filename tag for a scale cut, matching the convention used elsewhere in `results/`
    (e.g. [0.3, 3] -> '0.3_3', [1, 5] -> '1_5').
    """
    return f"{float(scale_cut[0]):g}_{float(scale_cut[-1]):g}"


def fit_cache_path(scale_cut, version, root=None):
    """Location of the cached fit for this scale cut / version."""
    if root is None:
        import src.statistics.corrfiles as cf

        root = cf.get_base_dir()
    return (
        Path(root)
        / "results"
        / "photoz_bias_fits"
        / f"photoz_bias_fit_{scale_cut_tag(scale_cut)}_{version}.npz"
    )



class PhotoBiasModel:
    """
    Parameters
    ----------
    mode : {'powerlaw', 'polynomial'}
    params : array_like
        Best-fit parameters.
    cov : array_like
        Full parameter covariance matrix.
    use_covariance : bool
    """

    def __init__(self, mode, params, cov, use_covariance=True):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}.")
        self.mode = mode
        self.params = np.asarray(params, dtype=float)
        self.cov = np.atleast_2d(np.asarray(cov, dtype=float))
        self.use_covariance = use_covariance
        if self.cov.shape != (self.params.size, self.params.size):
            raise ValueError(
                f"cov has shape {self.cov.shape}, expected "
                f"{(self.params.size, self.params.size)}"
            )

    @classmethod
    def from_cache(cls, scale_cut, version, mode="powerlaw", root=None, **kwargs):
        path = fit_cache_path(scale_cut, version, root=root)
        if path.exists():
            with np.load(path) as f:
                if f"{mode}/params" in f:
                    return cls(mode, f[f"{mode}/params"], f[f"{mode}/cov"], **kwargs)
            print(f"[biasfit] {path} has no {mode!r} fit, falling back on hardcoded.")
        else:
            print(f"[biasfit] no cached fit at {path}, falling back on hardcoded.")
        return cls.hardcoded(scale_cut, mode=mode, **kwargs)

    @classmethod
    def hardcoded(cls, scale_cut, mode="powerlaw", **kwargs):
        """Build from the values inlined in this module (diagonal covariance only)."""
        key = _key(scale_cut)
        if key not in _HARDCODED or mode not in _HARDCODED[key]:
            raise ValueError(
                f"No hardcoded {mode!r} fit for scale cut {list(key)}. "
                "Run biasfit.refit_photoz_bias to generate one."
            )
        params, errs = _HARDCODED[key][mode]
        return cls(mode, params, np.diag(np.asarray(errs, dtype=float) ** 2), **kwargs)

    def value(self, z):
        """f(z)."""
        scalar = np.ndim(z) == 0
        out = MODEL_FUNCS[self.mode](np.atleast_1d(np.asarray(z, float)), *self.params)
        return float(out[0]) if scalar else out

    def sigma(self, z):
        """1-sigma uncertainty on f(z), from J C J^T."""
        scalar = np.ndim(z) == 0
        zz = np.atleast_1d(np.asarray(z, float))
        J = _jacobian(self.mode, zz, self.params)
        cov = self.cov if self.use_covariance else np.diag(np.diag(self.cov))
        var = np.einsum("ij,jk,ik->i", J, cov, J)
        out = np.sqrt(np.clip(var, 0, None))
        return float(out[0]) if scalar else out

    def rel_sigma(self, z):
        """sigma(z) / f(z)."""
        return self.sigma(z) / self.value(z)

    def __repr__(self):
        errs = np.sqrt(np.diag(self.cov))
        ps = ", ".join(f"{p:.5g}+-{e:.3g}" for p, e in zip(self.params, errs))
        return (
            f"PhotoBiasModel(mode={self.mode!r}, {ps}, "
            f"use_covariance={self.use_covariance})"
        )


def correlation_matrix(cov):
    """Covariance -> correlation matrix."""
    cov = np.asarray(cov, dtype=float)
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def refit_photoz_bias(
    scale_cut,
    version,
    root=None,
    z_range=(0.0, 1.6),
    dz_phot=0.1,
    save=True,
    overwrite=False,
    verbose=True,
):
    """
    Returns
    -------
    dict
        ``{'z', 'vals', 'errs', '<mode>/params', '<mode>/cov', ...}``
    """
    import scipy.interpolate as interp
    import scipy.optimize as opt

    import src.statistics.corrfiles as cf
    import src.statistics.cosmotools as ct
    import src.statistics.inference as inference
    import src.statistics.combination as comb

    if root is None:
        root = cf.get_base_dir()
    root = Path(root)
    sc = [float(scale_cut[0]), float(scale_cut[-1])]
    tag = scale_cut_tag(sc)

    out_path = fit_cache_path(sc, version, root=root)
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"[biasfit] {out_path} already exists, loading it.")
        with np.load(out_path) as f:
            return {k: f[k] for k in f.files}

    corr_root = root / "src" / "statistics" / "outputs" / "correction"
    path_hsc = corr_root / "autos_HSC"
    fr_hsc = cf.CorrFileReader(path_hsc)

    # photometric (HSC) fine bins
    bins_z_photo = inference._get_fine_redshift_bins(fr=fr_hsc, tracer="HSC")
    vals_z_photo = (bins_z_photo[:-1] + bins_z_photo[1:]) / 2

    # dark matter angular correlation, integrated over the scale cut
    vals_z_wdm = np.linspace(0.01, 3, 150)
    rp_wdm = np.linspace(sc[0], sc[1], 100)
    wdm_values = np.array(
        [ct.w_dm(rp_vals=rp_wdm, z=z, integrate=True) for z in vals_z_wdm]
    )
    wdm_interpolator = interp.interp1d(
        vals_z_wdm, wdm_values, bounds_error=False, fill_value="extrapolate"
    )
    # w_dm integrated over each photometric bin
    wdm_integrated = np.array(
        [
            np.trapezoid(
                wdm_interpolator(np.linspace(z - dz_phot / 2, z + dz_phot / 2, 101)),
                np.linspace(z - dz_phot / 2, z + dz_phot / 2, 101),
            )
            for z in vals_z_photo
        ]
    )

    galbias_file = (
        root / "results" / f"photoz_bias_splines_{tag}_{version}" / "tomo_photoz.npz"
    )
    data_galbias = np.load(galbias_file)

    corr_factor, corr_factor_err, means_npk = [], [], []
    for i in range(len(vals_z_photo)):
        redshifts = data_galbias[f"{i+1}/redshifts"]
        nz_med = data_galbias[f"{i+1}/nz_median"]
        uncertainty = (
            data_galbias[f"{i+1}/nz_upper"] - data_galbias[f"{i+1}/nz_lower"]
        ) / 2
        means_npk.append(
            np.trapezoid(redshifts * nz_med, x=redshifts)
            / np.trapezoid(nz_med, x=redshifts)
        )

        wdm_inter = wdm_interpolator(redshifts)
        weights_trapz = comb.trapz_weights(redshifts)

        num = wdm_integrated[i] / (dz_phot**2)
        denom = np.trapezoid(np.multiply(np.array(nz_med) ** 2, wdm_inter), x=redshifts)
        delta_D = np.sqrt(
            np.sum((2 * nz_med * wdm_inter * weights_trapz) ** 2 * uncertainty**2)
        )
        corr_factor.append(num / denom)
        corr_factor_err.append(num / (denom**2) * delta_D)

    means_npk = np.asarray(means_npk)

    wpp_scaled, wpp_err_scaled = [], []
    for i in range(1, len(vals_z_photo) + 1):
        wpp_meas, wpp_err_meas, _ = inference.wpp(
            path=path_hsc, scale_cuts=sc, bin_index=i
        )
        wpp_scaled.append(wpp_meas)
        wpp_err_scaled.append(wpp_err_meas)

    mask = (vals_z_photo >= z_range[0]) & (vals_z_photo <= z_range[1])
    z = means_npk[mask]
    wpp_m = np.array(wpp_scaled)[mask]
    wpp_err_m = np.array(wpp_err_scaled)[mask]
    corr_m = np.array(corr_factor)[mask]
    corr_err_m = np.array(corr_factor_err)[mask]

    vals = np.sqrt(wpp_m * corr_m)
    errs = comb.combine_error_bars_mult(wpp_m, wpp_err_m, corr_m, corr_err_m)

    result = {
        "z": z,
        "vals": vals,
        "errs": errs,
        "wpp": wpp_m,
        "wpp_err": wpp_err_m,
        "scale_cut": np.asarray(sc, dtype=float),
    }

    for mode in MODES:
        p0 = {"powerlaw": (0.4, 0.4), "polynomial": (0.0, 0.1, 0.4)}[mode]
        popt, pcov = opt.curve_fit(
            MODEL_FUNCS[mode], z, vals, p0=p0, sigma=errs, absolute_sigma=True
        )
        result[f"{mode}/params"] = popt
        result[f"{mode}/cov"] = pcov
        resid = (vals - MODEL_FUNCS[mode](z, *popt)) / errs
        dof = len(z) - N_PARAMS[mode]
        result[f"{mode}/chi2"] = np.array(float(np.sum(resid**2)))
        result[f"{mode}/dof"] = np.array(dof)
        if verbose:
            errs_p = np.sqrt(np.diag(pcov))
            print(f"[biasfit] {mode}:")
            for name, p, e in zip("abcdefg", popt, errs_p):
                print(f"    {name} = {p:.10g} +- {e:.10g}")
            print(f"    chi2/dof = {np.sum(resid**2):.2f}/{dof}")
            print(f"    correlation matrix:\n{correlation_matrix(pcov)}")

    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **result)
        if verbose:
            print(f"[biasfit] saved -> {out_path}")

    return result
