"""
What the systematics change about the fiducial Delta z prior.

The shear likelihood puts an independent Gaussian on each bin's shift,
``Delta z_i ~ N(0, sigma_i)``, so the prior volume is proportional to the product of the
widths. Each systematic is a re-run of the fiducial B-spline fit, summarised by the same
two numbers: the mean redshift and its posterior width.

magnification
    alpha -> alpha + N(0, sigma_alpha), N realizations. The draws are symmetric, so it
    scatters <z> between realizations without moving it on average, and enters as an
    extra error in quadrature:

        sigma_sys = std(<z>_r),  sigma_tot = sqrt(sigma_fid^2 + sigma_sys^2).

    sigma_sys is measured from the realizations alone. It never divides by the fiducial
    fit, whose own sigma carries ~1% Monte Carlo error, so it cannot come out negative.

polynomial bias
    One re-fit with a degree-2 photometric bias law instead of the power law, compared
    directly against the fiducial fit. It displaces <z> rather than changing the width.

Shifts are quoted in units of sigma_fid, the error they have to be seen against.

Method detail, the four ensembles and the known limitations are in
`src/statistics/nb/nz_systematics_recap.ipynb`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.analysis.plots as plots
import src.statistics.systematics as sy

ROOT = Path(__file__).resolve().parents[3]

SCALE_CUT = [0.3, 3]
VERSION = "v_1p1"
NAME = "npz_bs_bp_mag"      # the measurement the shear priors are built from
MAG_STUDY = "magabs"

C_MAG, C_POLY = "#B3402F", "#2B5D8C"


def _style():
    plots.plot_settings({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10,
                         "axes.grid": False})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.parent / f"{path.name}.{ext}", dpi=300)


def ensembles(root, scale_cut=SCALE_CUT, version=VERSION, mag_study=MAG_STUDY,
              name=NAME):
    """
    Every <z> posterior this analysis compares, per tomographic bin.
    """
    import src.statistics.scripts.magabs_meanz_distribution as mm

    tag = sy.scale_cut_tag(scale_cut)
    alpha = mm.mean_z_draws(root, scale_cut, version, mag_study, name)

    seeds_npz = (Path(root) / "results" / f"splines_fidseeds_{tag}_{version}"
                 / "fidseed_meanz_draws.npz")
    seeds = np.load(seeds_npz) if seeds_npz.exists() else None

    poly_cache = (sy.variant_dir(root, "polybias", scale_cut, version)
                  / f"polybias_meanz_draws_{tag}_{version}.npz")
    if poly_cache.exists():
        poly = np.load(poly_cache)
        poly = {t: poly[str(t)] for t in sy.TOMO_BINS}
    else:
        spl = Path(root) / "results" / f"splines_polybias_{tag}_{version}"
        poly = {t: mm._mean_z_draws(str(spl / f"spl_{name}_{t}")) for t in sy.TOMO_BINS}
        np.savez_compressed(poly_cache, **{str(t): v for t, v in poly.items()})

    out = {}
    for t in sy.TOMO_BINS:
        e = {"r00": alpha[t][0], "alpha": alpha[t][1:], "poly": poly[t]}
        if seeds is not None:
            e["seeds"] = seeds[str(t)]
        out[t] = e
    return out


def nz_bands(root, scale_cut=SCALE_CUT, version=VERSION, mag_study=MAG_STUDY,
             name=NAME, n_sub=200, rebuild=False):
    """
    n(z) posterior bands for the fiducial fit and each systematic, per bin.
    """
    import pickle
    import arviz as az
    from scipy import interpolate
    from scipy.integrate import simpson

    tag = sy.scale_cut_tag(scale_cut)
    cache = (sy.variant_dir(root, mag_study, scale_cut, version)
             / f"{mag_study}_nz_bands_{tag}_{version}.npz")
    if cache.exists() and not rebuild:
        d = np.load(cache)
        return {t: {"z": d[f"{t}/z"],
                    **{k: (d[f"{t}/{k}_med"], d[f"{t}/{k}_lo"], d[f"{t}/{k}_hi"])
                       for k in ("fid", "alpha", "poly")}}
                for t in sy.TOMO_BINS}

    rng = np.random.default_rng(0)

    def draws(stem, z=None):
        m = pickle.load(open(f"{stem}_meta.pkl", "rb"))
        po = az.from_netcdf(f"{stem}.nc").posterior
        c = po["coeffs"].values.reshape(-1, m["n_basis"])
        a = po["amplitude"].values.reshape(-1)
        if z is None:
            z = np.linspace(m["zv"].min(), m["zv"].max(), 200)
        B = np.column_stack([
            interpolate.BSpline(m["knots"], np.eye(m["n_basis"])[i], m["degree"])(z)
            for i in range(m["n_basis"])
        ])
        nz = (c @ B.T) * a[:, None]
        return z, nz / simpson(nz, z, axis=1)[:, None]

    mag_dir = Path(root) / "results" / f"splines_{mag_study}_{tag}_{version}"
    poly_dir = Path(root) / "results" / f"splines_polybias_{tag}_{version}"
    meta = json.loads((sy.variant_dir(root, mag_study, scale_cut, version)
                       / f"{mag_study}_metadata_{tag}_{version}.json").read_text())

    out, flat = {}, {}
    for t in sy.TOMO_BINS:
        z, fid = draws(str(mag_dir / f"spl_{name}_{t}_r00"))
        pooled = []
        for r in range(1, meta["n_realizations"] + 1):
            _, nz = draws(str(mag_dir / f"spl_{name}_{t}_r{r:02d}"), z)
            pooled.append(nz[rng.choice(len(nz), n_sub, replace=False)])
        _, poly = draws(str(poly_dir / f"spl_{name}_{t}"), z)

        entry = {"z": z}
        for k, arr in (("fid", fid), ("alpha", np.concatenate(pooled)), ("poly", poly)):
            q = np.percentile(arr, [50, 16, 84], axis=0)
            entry[k] = (q[0], q[1], q[2])
            flat[f"{t}/{k}_med"], flat[f"{t}/{k}_lo"], flat[f"{t}/{k}_hi"] = q
        flat[f"{t}/z"] = z
        out[t] = entry
        print(f"  bin {t} done")

    np.savez_compressed(cache, **flat)
    print(f"cached -> {cache.name}")
    return out


def load(root, scale_cut=SCALE_CUT, version=VERSION, mag_study=MAG_STUDY, name=NAME):
    """The two systematics CSVs, the magnification metadata, and per-realization <z>."""
    tag = sy.scale_cut_tag(scale_cut)
    mag_dir = sy.variant_dir(root, mag_study, scale_cut, version)

    mag = pd.read_csv(
        mag_dir / f"{mag_study}_sigma_budget_{tag}_{version}.csv"
    ).set_index("tomo_bin")

    poly = pd.read_csv(sy.variant_dir(root, "polybias", scale_cut, version)
                       / f"polybias_sigma_budget_{tag}_{version}.csv")
    poly = poly[poly["name"] == name].set_index("tomo_bin")

    with open(mag_dir / f"{mag_study}_metadata_{tag}_{version}.json") as f:
        meta = json.load(f)

    npz = np.load(mag_dir / f"{mag_study}_summary_{tag}_{version}.npz")
    rmz = {t: npz[f"{t}/realization_mean_z"] for t in mag.index}   # row 0 unperturbed
    return mag, poly, meta, rmz


def budget(mag, poly, realization_mean_z):
    """
    What each systematic changes about the fiducial fit, per tomographic bin.

    Magnification scatters <z> between realizations without moving it, so it enters as
    an extra error in quadrature. The polynomial law is one alternative fit, compared
    directly. `mag_dz_[sig]` is a check, not a result: the alpha draws are symmetric, so
    it must come out ~0.
    """
    b = pd.DataFrame(index=mag.index)
    b["mean_z"] = mag["mean_z_fiducial"]
    b["sigma_fid"] = mag["sigma_fiducial"]

    b["sigma_sys"] = pd.Series(
        {t: v[1:].std(ddof=1) for t, v in realization_mean_z.items()}
    )
    b["sigma_tot"] = np.hypot(b["sigma_fid"], b["sigma_sys"])
    b["mag_dsigma_%"] = 100 * (b["sigma_tot"] / b["sigma_fid"] - 1)
    b["mag_dz_[sig]"] = (
        mag["mean_z_pooled"] - mag["mean_z_fiducial"]
    ) / b["sigma_fid"]

    b["poly_dsigma_%"] = 100 * (poly["sigma_polynomial"] / b["sigma_fid"] - 1)
    b["poly_dz_[sig]"] = (
        poly["mean_z_polynomial"] - poly["mean_z_powerlaw"]
    ) / b["sigma_fid"]
    return b


def measurements(root, scale_cut=SCALE_CUT, version=VERSION, mag_study=MAG_STUDY,
                 name=NAME):
    """The measured n(z) points the splines are fitted to, per bin."""
    tag = sy.scale_cut_tag(scale_cut)
    d = sy.variant_dir(root, mag_study, scale_cut, version)
    meta = json.loads((d / f"{mag_study}_metadata_{tag}_{version}.json").read_text())

    fid = np.load(d / f"merged_res_norm_{tag}_{version}_r00.npz")
    poly = np.load(sy.variant_dir(root, "polybias", scale_cut, version)
                   / f"merged_res_norm_{tag}_{version}.npz")
    alpha = [np.load(d / f"merged_res_norm_{tag}_{version}_r{r:02d}.npz")
             for r in range(1, meta["n_realizations"] + 1)]

    out = {}
    for t in sy.TOMO_BINS:
        k = f"{t}/{name}"
        out[t] = {
            "z": fid[f"{k}_z"],
            "fid": fid[k],
            "fid_err": fid[f"{k}_err"],
            "alpha": np.array([a[k] for a in alpha]),
            "poly": poly[k],
        }
    return out


def figure_nz(bands, meas, path):
    """n(z): the fiducial posterior band, the measured points, and each systematic."""
    _style()
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.0))
    for t, ax in zip(sorted(bands), axs.flat):
        z, m = bands[t]["z"], meas[t]
        med, lo, hi = bands[t]["fid"]

        for row in m["alpha"]:                                   # each realization
            ax.plot(m["z"], row, color=C_MAG, lw=0.4, alpha=0.12, zorder=1)
        ax.fill_between(z, lo, hi, color="0.6", alpha=0.5, lw=0, zorder=2,
                        label="fiducial")
        ax.plot(z, med, color="0.3", lw=1.4, zorder=3)
        ax.plot(z, bands[t]["poly"][0], color=C_POLY, lw=1.4, ls="--", zorder=4,
                label=r"polynomial $b_p(z)$")
        ax.errorbar(m["z"], m["fid"], m["fid_err"], fmt="o", ms=3, lw=1.0, capsize=2,
                    color="k", zorder=5, label="measurement")
        ax.plot([], [], color=C_MAG, lw=1.0, alpha=0.5,
                label=r"$\alpha$ realizations")

        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xlim(z.min(), z.max())
        ax.set_ylim(bottom=min(0, 1.1 * m["fid"].min()), top=1.95 * hi.max())
        ax.text(0.97, 0.93, f"Bin {t}", transform=ax.transAxes, ha="right", va="top",
                fontsize=12)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$n(z)$")
        if t == min(bands):
            ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2,
                      columnspacing=1.0, handlelength=1.4)
    _save(fig, path)
    return fig


def figure_distributions(ens, path, show_seeds=False):
    """<z> posterior of the fiducial fit and of each systematic, with a rug of the
    per-realization means below each panel."""
    _style()
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.0))
    for t, ax in zip(sorted(ens), axs.flat):
        e = ens[t]
        pooled = e["alpha"].ravel()
        lo, hi = np.percentile(np.concatenate([pooled, e["poly"]]), [0.1, 99.9])
        edges = np.linspace(lo, hi, 80)
        step = dict(bins=edges, density=True, histtype="step", lw=1.6)

        ax.hist(e["r00"], bins=edges, density=True, color="0.6", alpha=0.55,
                label="fiducial")
        if show_seeds and "seeds" in e:
            ax.hist(e["seeds"].ravel(), color="k", label="fiducial, 20 seeds", **step)
        ax.hist(pooled, color=C_MAG, label=r"marginalised over $\alpha$", **step)
        ax.hist(e["poly"], color=C_POLY, label=r"polynomial $b_p(z)$", **step)

        top = 1.62 * ax.get_ylim()[1]
        for arr, col in ((e["r00"], "0.35"), (pooled, C_MAG), (e["poly"], C_POLY)):
            ax.axvline(arr.mean(), color=col, lw=1.1, ls=":", zorder=5)
        # rug: <z> of each alpha realization
        centres = e["alpha"].mean(axis=1)
        ax.plot(centres, np.full_like(centres, -0.030 * top), "|", color=C_MAG,
                ms=7, mew=0.9, alpha=0.8, clip_on=False)

        ax.set_ylim(-0.06 * top, top)
        ax.set_xlim(lo, hi)
        ax.text(0.97, 0.93, f"Bin {t}", transform=ax.transAxes, ha="right", va="top",
                fontsize=12)
        ax.set_xlabel(r"$\langle z \rangle$")
        ax.set_ylabel("density")
        if t == min(ens):
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _save(fig, path)
    return fig
