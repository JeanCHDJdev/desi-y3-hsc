# results
---------

In this folder, you will find the fiducial distributions. Sacale cuts are indicated by the folder name (0.3-3) and (1-5) in Mpc/h.
The median measurements of the samples are provided. 

You can find all measurements (including removing other corrections, such as bias evolution) under the `distributions_*.npz` file :
- `cross` assumes only dark matter evolution
- `nz_bs` uses spectroscopic bias evolution + dark matter evolution
- `nz_bs_bp` uses spectroscopic bias evolution + photometric bias evolution corrections
- `nz_bs_bp_mag` uses all available corrections, including magnification corrections.
Here, `_vF` simply denotes the fiducial version that is released with the paper. If necessary, a `vP` version (after publication) will also be released on the github,
following the standard review process after journal submission.

You can also find measurements for the expectations of each sample under `expectations_*.npz`
Finally, the measurements themselves used to obtain the model are stored under `merged_res_norm_*.npz`.
The `_metadata` files offer more insights into the measurements, such as which files and tracers were used for the measurement.

One can find all of the samples in the `.sacc` and `.fits` files in the nz_multirank format expected by CosmoSIS for parameter inference.
The samples derived from spline inference are in the fits extensions of the files.

For the photometric galaxy bias correction, one can find the tomographic bins in the `photoz_bias_splines_*` directories, under the `tomo_photoz.npz` file.



