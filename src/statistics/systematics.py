"""
Helpers shared by the systematics variants of the clustering-redshift analysis.

Two systematics are studied, each against the fiducial analysis and each in its own
set of notebooks:

``magabs``
    Perturb the magnification coefficients, alpha -> alpha + N(0, 0.1), and regenerate
    several realizations of the magnification-corrected n(z). Only ``npz_bs_bp_mag``
    changes, so the realizations come from re-solving the magnification system on the
    fiducial measurements.

``polybias``
    Replace the power-law photometric bias correction by a degree-2 polynomial and
    propagate that fit's (full-covariance) uncertainty into n(z).

Nothing here imports ``pyccl``/``pycorr``/``pymc`` at module level, so this module can
be imported from both the ``desi`` and the ``pymc`` environments; the heavy imports
happen inside the functions that need them.
"""

import numpy as np
import pandas as pd

from pathlib import Path

# --------------------------------------------------------------------------------------
# analysis configuration (mirrors nb/nz.ipynb and nb/nz_plots.ipynb)
# --------------------------------------------------------------------------------------

#: which spectroscopic tracers enter each HSC tomographic bin
TOMO_TO_TRACER = {
    1: ["BGS_ANY", "LRG"],
    2: ["BGS_ANY", "LRG", "QSO"],
    3: ["LRG", "ELGnotqso", "QSO"],
    4: ["LRG", "ELGnotqso", "QSO"],
}

#: redshift range kept for each tomographic bin when normalizing
BOUNDS = {"1": (0, 0.8), "2": (0.3, 1.3), "3": (0.3, 2.1), "4": (0.7, 2.1)}

#: sampler settings shared by the fiducial nz_splines.ipynb, the systematics spline
#: notebooks and scripts/fit_splines.py, so every fit compared against another is drawn
#: from the same posterior definition
SPLINE_FIT_KWARGS = dict(
    n_tune=400,
    n_samples=1600,
    target_accept=0.99,
    prior_concentration=3,
    base_alpha=0.05,
)

TOMO_BINS = [1, 2, 3, 4]
PATCHES = [1, 2, 3, 4]
NAMES = ["npz_cross", "npz_bs", "npz_bs_bp", "npz_bs_bp_mag"]


def stem_for_tomo(tomo):
    """DESI data release used for a given tomographic bin."""
    return "dr1" if tomo in (1, 2) else "dr2"


def scale_cut_tag(scale_cut):
    """[0.3, 3] -> '0.3_3', [1, 5] -> '1_5' (the convention used across results/)."""
    return f"{float(scale_cut[0]):g}_{float(scale_cut[-1]):g}"


def build_path_dictionary(corr_root, stem, version):
    corr_root = Path(corr_root)
    return {
        "HSC": corr_root / "v12_correction" / "autos_HSC",
        "DESI_NGC": corr_root / stem / "autos_NGC",
        "DESI_SGC": corr_root / stem / "autos_SGC",
        "DESIxHSC": corr_root / stem / "cross",
        "MergedxMerged": corr_root / f"merged_{stem}_{version}",
        "MergedxHSC": corr_root / f"merged_{stem}_{version}",
    }


def variant_dir(root, kind, scale_cut, version, what="distributions"):
    """
    Output directory for a systematics variant, e.g.
    ``results/distributions/magabs_0.3_3_v_1p1`` or ``results/splines_magabs_...``.
    """
    tag = f"{kind}_{scale_cut_tag(scale_cut)}_{version}"
    root = Path(root)
    return root / "results" / "distributions" / tag if what == "distributions" else (
        root / "results" / f"{what}_{tag}"
    )


# --------------------------------------------------------------------------------------
# w_dm caching
# --------------------------------------------------------------------------------------


class WdmCache:
    """
    Cache of ``w_dm`` evaluated on a redshift grid for a fixed scale cut.

    ``solve_magnification`` needs w_dm at every redshift of the tracer grid. It does not
    depend on the magnification coefficients, so it is computed once per (tracer, grid)
    and reused across every realization. Requires ``pyccl`` (``desi`` env).
    """

    def __init__(self, scale_cut, n_rp=101):
        self.scale_cut = [float(scale_cut[0]), float(scale_cut[-1])]
        self.n_rp = n_rp
        self._cache = {}

    def get(self, zvalues):
        import src.statistics.cosmotools as ct

        zvalues = np.asarray(zvalues, dtype=float)
        key = tuple(np.round(zvalues, 8))
        if key not in self._cache:
            rp_vals = np.linspace(self.scale_cut[0], self.scale_cut[-1], self.n_rp)
            self._cache[key] = np.array(
                [ct.w_dm(rp_vals, z, integrate=True) for z in zvalues]
            )
        return self._cache[key]

    def __len__(self):
        return len(self._cache)


def precompute_wdm_interpolator(scale_cut, z_min=0.01, z_max=3.0, n_z=150, n_rp=100):
    """Interpolator of the scale-cut-integrated w_dm, as used in nb/nz.ipynb."""
    import scipy.interpolate as interp
    import src.statistics.cosmotools as ct

    vals_z = np.linspace(z_min, z_max, n_z)
    rp = np.linspace(scale_cut[0], scale_cut[-1], n_rp)
    vals = np.array([ct.w_dm(rp_vals=rp, z=z, integrate=True) for z in vals_z])
    return interp.interp1d(vals_z, vals, bounds_error=False, fill_value="extrapolate")


# --------------------------------------------------------------------------------------
# magnification perturbation
# --------------------------------------------------------------------------------------


#: how the magnification coefficients alpha are perturbed
#:   "relative" -- alpha -> alpha * (1 + N(0, sigma)); sigma is a *fractional* error
#:   "absolute" -- alpha -> alpha + N(0, sigma);       sigma is an error on alpha itself


def draw_realization_alpha(
    mag_sigma,
    realization,
    seed=20260823,
    tomo_bins=None,
    tracers=None,
):
    """
    Draw the perturbation of the magnification coefficients alpha for one realization.

    alpha -> alpha(z) + N(0, mag_sigma), an *absolute* error: 0.1 is a 1-sigma error of
    0.1 on alpha itself, which is the form the measurements come in.
 
    ``realization == 0`` returns zero offsets, so realization 0 is the fiducial
    analysis by construction.

    Returns
    -------
    dict
        ``{'p_offset': {tomo: offset}, 's_offset': {tracer: offset},
        'realization': int, 'mag_perturbation': float}``
    """
    if tomo_bins is None:
        tomo_bins = TOMO_BINS
    if tracers is None:
        tracers = sorted({t for ts in TOMO_TO_TRACER.values() for t in ts})

    meta = {"realization": int(realization), "mag_perturbation": float(mag_sigma)}
    if realization == 0 or not mag_sigma:
        return {
            "p_offset": {int(t): 0.0 for t in tomo_bins},
            "s_offset": {str(t): 0.0 for t in tracers},
            **meta,
        }

    rng = np.random.default_rng([seed, int(realization)])
    return {
        "p_offset": {int(t): float(rng.normal(0.0, mag_sigma)) for t in tomo_bins},
        "s_offset": {str(t): float(rng.normal(0.0, mag_sigma)) for t in tracers},
        **meta,
    }


def alpha_kwargs(scales, tomo, tracer):
    return {
        "alpha_offset_p": scales["p_offset"][tomo],
        "alpha_offset_s": scales["s_offset"][tracer],
    }


def merge_over_tracers(df, names=NAMES, tomo_bins=TOMO_BINS):
    import src.statistics.inference as inference

    merged = {}
    for name in names:
        if name not in df.columns:
            continue
        for tomo in tomo_bins:
            dt = df[df["tomo_bin"] == tomo]
            dt = dt[dt[name].notna()]
            tracers = list(dict.fromkeys(dt["tracer"]))
            if not tracers:
                continue
            zv = [dt["redshift"][dt["tracer"] == t].values for t in tracers]
            vals = [dt[name][dt["tracer"] == t].values.astype(float) for t in tracers]
            errs = [
                dt[name + "_err"][dt["tracer"] == t].values.astype(float)
                for t in tracers
            ]
            zm, nm, em = inference.merge_results(zv, vals, errs)
            merged[f"{tomo}/{name}_z"] = zm
            merged[f"{tomo}/{name}"] = nm
            merged[f"{tomo}/{name}_err"] = em
    return merged


def normalize_merged(merged, names=NAMES, tomo_bins=TOMO_BINS, bounds=BOUNDS):
    norm = {}
    for tomo in tomo_bins:
        lo, hi = bounds[str(tomo)]
        for name in names:
            key = f"{tomo}/{name}"
            if key not in merged:
                continue
            zn = merged[f"{key}_z"]
            m = (zn >= lo) & (zn <= hi)
            vals = merged[key][m]
            errs = merged[f"{key}_err"][m]
            amplitude = np.trapezoid(vals, zn[m])
            norm[f"{key}_z"] = zn[m]
            norm[key] = vals / amplitude
            norm[f"{key}_err"] = errs / amplitude
    return norm


def normalized_samples(spl, z_eval, n_eval_points=200):
    from scipy.integrate import simpson

    samples = spl.get_samples(z_eval=z_eval, n_eval_points=n_eval_points)
    return samples / simpson(samples, z_eval, axis=1)[:, None]


def summarize_samples(samples, z_eval):
    mean_z = np.trapezoid(samples * z_eval, z_eval, axis=1)
    return {
        "z_eval": z_eval,
        "median": np.percentile(samples, 50, axis=0),
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0),
        "lower": np.percentile(samples, 16, axis=0),
        "upper": np.percentile(samples, 84, axis=0),
        "mean_z_samples": mean_z,
        "mean_z": float(np.mean(mean_z)),
        "mean_z_median": float(np.median(mean_z)),
        "mean_z_std": float(np.std(mean_z)),
        "mean_z_percentiles": np.percentile(mean_z, [16, 50, 84]),
    }


def common_grid(splines, n_points=400):
    zmin = max(np.min(s.zv) for s in splines)
    zmax = min(np.max(s.zv) for s in splines)
    return np.linspace(zmin, zmax, n_points)


def pool_realizations(splines, z_eval, n_eval_points=200, max_per_realization=None,
                      seed=0):
    """
    Evaluate the (normalized) posterior samples of several realizations on a common grid.
    Returns
    -------
    pooled : (n_total, n_z) array
    per_realization : list of (n_i, n_z) arrays
    """
    rng = np.random.default_rng(seed)
    per = []
    for spl in splines:
        s = normalized_samples(spl, z_eval, n_eval_points=n_eval_points)
        if max_per_realization is not None and len(s) > max_per_realization:
            s = s[rng.choice(len(s), max_per_realization, replace=False)]
        per.append(s)
    return np.concatenate(per, axis=0), per


def sigma_budget(fiducial_summary, pooled_summary, per_realization_summaries=None):
    """
    Summarise the fiducial and alpha-marginalised mean-redshift posteriors.
    """
    f16, f50, f84 = fiducial_summary["mean_z_percentiles"]
    out = {
        "mean_z_fiducial": fiducial_summary["mean_z"],
        "sigma_fiducial": fiducial_summary["mean_z_std"],
        # asymmetric errors, matching the convention of tab:shifts_results in the paper
        "mean_z_fid_p50": f50,
        "mean_z_fid_lo": f50 - f16,
        "mean_z_fid_hi": f84 - f50,
        # marginalised over alpha
        "mean_z_pooled": pooled_summary["mean_z"],
        "sigma_pooled": pooled_summary["mean_z_std"],
    }
    if per_realization_summaries is not None:
        centres = np.array([s["mean_z"] for s in per_realization_summaries])
        out["sigma_sys"] = float(np.std(centres, ddof=1))
        out["n_realizations"] = len(centres)
    return out
