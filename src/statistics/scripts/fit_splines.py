"""
Batch driver for the systematics B-spline fits, one OS process per fit.
"""

import argparse
import os
import subprocess
import sys
import time

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _isolate_pytensor_cache():
    """
    Give this process its own PyTensor C-module cache.

    PyTensor's default cache is ~/.pytensor on shared GPFS. Its inter-process lock is
    unreliable there, and when it fails the forked PyMC chain worker dies and the parent
    reports `ConnectionResetError: [Errno 104]`. Every fit of one batch job failed that
    way. A per-process cache removes the contention entirely; the cost is recompiling
    the model each time, which is seconds against a ~90 s fit.

    Must run before pymc/pytensor are imported.
    """
    if os.environ.get("PYTENSOR_FLAGS"):
        return
    compiledir = REPO / ".pytensor_cache" / f"p{os.getpid()}"
    compiledir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={compiledir}"


_isolate_pytensor_cache()

import numpy as np  # noqa: E402

import src.statistics.corrfiles as cf  # noqa: E402
import src.statistics.systematics as sy  # noqa: E402

STUDIES = ("magabs", "polybias")


def needs_refit(data_file, key, savefile):
    """
    Should this fit be (re)run?
    """
    if not Path(f"{savefile}.nc").exists():
        return True
    meta_path = Path(f"{savefile}_meta.pkl")
    if not meta_path.exists():
        return True
    try:
        import pickle

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        data = np.load(data_file)
        return not (
            np.allclose(meta["nz"], data[key], rtol=1e-10, atol=0)
            and np.allclose(meta["nz_err"], data[f"{key}_err"], rtol=1e-10, atol=0)
        )
    except Exception:  # noqa: BLE001
        return True


def enumerate_jobs(study, scale_cut, version, root):
    """
    List (data_file, key, savefile) triples for a study.
    """
    tag = sy.scale_cut_tag(scale_cut)
    data_dir = sy.variant_dir(root, study, scale_cut, version)
    spl_dir = sy.variant_dir(root, study, scale_cut, version, what="splines")
    spl_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if study == "magabs":
        import json

        with open(data_dir / f"{study}_metadata_{tag}_{version}.json") as f:
            n_real = json.load(f)["n_realizations"]
        name = "npz_bs_bp_mag"
        for r in range(n_real + 1):
            data_file = data_dir / f"merged_res_norm_{tag}_{version}_r{r:02d}.npz"
            for tomo in sy.TOMO_BINS:
                jobs.append(
                    (data_file, f"{tomo}/{name}", spl_dir / f"spl_{name}_{tomo}_r{r:02d}")
                )
    elif study == "polybias":
        data_file = data_dir / f"merged_res_norm_{tag}_{version}.npz"
        for name in sy.NAMES:
            for tomo in sy.TOMO_BINS:
                jobs.append((data_file, f"{tomo}/{name}", spl_dir / f"spl_{name}_{tomo}"))
    else:
        raise ValueError(
            f"Unknown study {study!r}, expected one of {STUDIES}."
        )

    fresh = [j for j in jobs if not needs_refit(j[0], j[1], j[2])]
    todo = [j for j in jobs if needs_refit(j[0], j[1], j[2])]
    stale = [j for j in todo if Path(f"{j[2]}.nc").exists()]
    if stale:
        print(
            f"{len(stale)} existing fit(s) are stale (data changed since they were "
            f"fitted) and will be redone:"
        )
        for j in stale[:5]:
            print(f"    {j[2].name}")
        if len(stale) > 5:
            print(f"    ... and {len(stale) - 5} more")
    print(f"{len(fresh)} fit(s) already up to date")
    return todo


def enumerate_seed_jobs(scale_cut, version, root, n_seeds, base_seed=42,
                        study="magabs", name="npz_bs_bp_mag"):
    """
    Refit the *unperturbed* realization n_seeds times, one sampler seed each.
    """
    tag = sy.scale_cut_tag(scale_cut)
    data_file = (sy.variant_dir(root, study, scale_cut, version)
                 / f"merged_res_norm_{tag}_{version}_r00.npz")
    if not data_file.exists():
        raise FileNotFoundError(f"unperturbed realization not found: {data_file}")

    spl_dir = Path(root) / "results" / f"splines_fidseeds_{tag}_{version}"
    spl_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i in range(n_seeds):
        seed = base_seed + i
        for tomo in sy.TOMO_BINS:
            jobs.append((data_file, f"{tomo}/{name}",
                         spl_dir / f"spl_{name}_{tomo}_s{seed:03d}", seed))
    fresh = [j for j in jobs if not needs_refit(j[0], j[1], j[2])]
    todo = [j for j in jobs if needs_refit(j[0], j[1], j[2])]
    stale = [j for j in todo if Path(f"{j[2]}.nc").exists()]
    if stale:
        print(
            f"{len(stale)} existing fit(s) are stale (data changed since they were "
            f"fitted) and will be redone:"
        )
        for j in stale[:5]:
            print(f"    {j[2].name}")
        if len(stale) > 5:
            print(f"    ... and {len(stale) - 5} more")
    print(f"{len(fresh)} fit(s) already up to date")
    return todo


def fit_one(data_file, key, savefile, cores=None, seed=None):
    """
    Fit and save a single spline. Runs in the worker process.
    """
    import src.statistics.spline as spline

    data = np.load(data_file)
    z = data[f"{key}_z"]
    nz = data[key]
    nz_err = data[f"{key}_err"]

    print(f"fitting {key} from {Path(data_file).name} ({len(z)} points) -> {savefile}")
    spl = spline.BayesianBSpline(zv=z, n_knots=int(len(z) // 2))
    kwargs = dict(sy.SPLINE_FIT_KWARGS)
    if seed is not None:
        kwargs["seed"] = int(seed)
    spl.fit(nz, nz_err, cores=cores, **kwargs)
    spl.save_model(str(savefile))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", choices=list(STUDIES))
    ap.add_argument(
        "--fiducial-seeds", type=int, default=None, metavar="N",
        help="instead of a study, refit the unperturbed realization under N sampler "
        "seeds into results/splines_fidseeds_<tag>_<version>/, to measure and beat "
        "down the Monte Carlo error on sigma_stat",
    )
    ap.add_argument("--scale-cut", nargs=2, type=float, default=[0.3, 3])
    ap.add_argument("--version", default="v_1p1")
    ap.add_argument("--retries", type=int, default=2, help="retries per failed fit")
    ap.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="give up once this many fits in a row fail, rather than burning the whole "
        "allocation on an environment problem (0 disables)",
    )
    ap.add_argument(
        "--cores", type=int, default=None,
        help="chains to sample in parallel per fit; 1 avoids PyMC's forked workers "
        "entirely, which is what dies with ConnectionResetError on a loaded node",
    )
    ap.add_argument(
        "--jobs", type=int, default=1,
        help="number of fits to run concurrently (each still in its own process)",
    )
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="fit in this process instead of spawning one per fit (not recommended)",
    )
    # worker mode
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--data", help=argparse.SUPPRESS)
    ap.add_argument("--key", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        fit_one(args.data, args.key, args.out, cores=args.cores, seed=args.seed)
        return 0

    root = cf.get_base_dir()
    if args.fiducial_seeds is not None:
        jobs = enumerate_seed_jobs(
            args.scale_cut, args.version, root, args.fiducial_seeds
        )
        label = f"fiducial x {args.fiducial_seeds} seeds"
    else:
        if args.study is None:
            ap.error("--study or --fiducial-seeds is required")
        jobs = [j + (None,) for j in
                enumerate_jobs(args.study, args.scale_cut, args.version, root)]
        label = args.study
    print(f"{label}: {len(jobs)} fit(s) to run (finished ones are skipped)\n")

    def run_one(index, data_file, key, savefile, seed=None):
        """One fit with its retries. Returns (name, error) or (name, None)."""
        t0 = time.time()
        print(f"[{index}/{len(jobs)}] {savefile.name}", flush=True)
        env = dict(os.environ)
        env["PYTENSOR_FLAGS"] = (
            f"base_compiledir={REPO / '.pytensor_cache' / savefile.name}"
        )

        for attempt in range(args.retries + 1):
            proc = subprocess.run(
                [
                    sys.executable, __file__, "--worker",
                    "--data", str(data_file), "--key", key, "--out", str(savefile),
                ]
                + ([] if args.cores is None else ["--cores", str(args.cores)])
                + ([] if seed is None else ["--seed", str(seed)]),
                cwd=str(root),
                env=env,
            )
            if proc.returncode == 0 and Path(f"{savefile}.nc").exists():
                print(f"    {savefile.name} ok, {time.time() - t0:.0f}s", flush=True)
                return savefile.name, None
            print(
                f"    {savefile.name} attempt {attempt + 1} failed "
                f"(rc={proc.returncode})",
                flush=True,
            )
        print(f"    {savefile.name} gave up, {time.time() - t0:.0f}s", flush=True)
        return savefile.name, f"rc={proc.returncode}"

    failed = []

    if args.in_process:
        for i, (data_file, key, savefile, seed) in enumerate(jobs, start=1):
            t0 = time.time()
            print(f"[{i}/{len(jobs)}] {savefile.name}", flush=True)
            try:
                fit_one(data_file, key, savefile, cores=args.cores, seed=seed)
            except Exception as exc:  # noqa: BLE001
                failed.append((savefile.name, repr(exc)))
            print(f"    {time.time() - t0:.0f}s", flush=True)

    elif args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor

        print(f"running {args.jobs} fits concurrently\n", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(run_one, i, *job) for i, job in enumerate(jobs, start=1)
            ]
            for fut in futures:
                name, err = fut.result()
                if err:
                    failed.append((name, err))

    else:
        consecutive = 0
        for i, (data_file, key, savefile, seed) in enumerate(jobs, start=1):
            name, err = run_one(i, data_file, key, savefile, seed)
            if err:
                failed.append((name, err))
                consecutive += 1
            else:
                consecutive = 0
            if (
                args.max_consecutive_failures
                and consecutive >= args.max_consecutive_failures
            ):
                print(
                    f"\n{consecutive} fits failed in a row -- stopping. This is an "
                    "environment problem, not a per-fit one; fix it before resubmitting.",
                    flush=True,
                )
                break

    if failed:
        print(f"\n{len(failed)} fit(s) failed:")
        for nm, why in failed:
            print(f"  {nm}: {why}")
        return 1
    print(f"\nAll {len(jobs)} fit(s) complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
