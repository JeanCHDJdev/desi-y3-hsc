# desi-y3-hsc
-------------

Github project for storing code related to the DESI-Y3 x HSC auto and cross correlation analysis. The original HSC Rau+2022 can be found [here](https://arxiv.org/pdf/2211.16516).

This analysis made for 2 distinct papers. 
Paper figures can be found under `paper/figures` and the associated LaTeX folder can be found under `paper/cosmic_shear_tex` and `paper/nz_tex`

One can find the cosmic shear results and code under the `cosmic-shear`, including the fiducial chains as well as reweighted chains for this analysis. The rest of this analysis, mostly found under `src` code, corresponds to the work done in order to calibrate the n($z$) distributions.

Result files can be found in the `results/distributions` directory.

### Installation
----------------

To obtain the required environments, one can find the environment to generate the spline models in `env-splines.yml`. The generic environment used for anything else is `env-desi.yml`, most notably to compute the cross-correlation functions.
One should also run :
```bash
pip install -e .
```
at the root level to install the project through the `pyproject.toml` file.

### Obtaining the HSC dataset
-----------------------------

On NERSC, if the user is a member of DESI, you can compute the auto and cross correlations for the dataset. To do so, one must generate the HSC catalog (and change paths accordingly) : you can find a script to generate the HSC catalog in `src/makecat/make_hscy3.py`. It is possible to add or obtain more columns for debugging or visual inspection through this script.

### Computing cross-correlations
----------------------------

One can compute cross-correlations on NERSC using the `run_corr.py` script and the associated arguments for it. Further improvements could add GPU computations to this, and further optimize the script. It is possible to find example commands to run auto and cross-correlation vectors in `src/statistics/run.md`. Further optimization would be to use GPUs.

### Cosmic shear
----------------

Cosmic shear measurements rely on the nersc job scripts found in the `cosmic_shear` directory. This requires to have `CosmoSIS` [installed](https://cosmosis.readthedocs.io/en/latest/). The default sampling is done with [pocoMC](https://pocomc.readthedocs.io/en/latest/).