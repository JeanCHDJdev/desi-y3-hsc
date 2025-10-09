# results
---------

In this folder, you will find the fiducial distributions. Sacale cuts are indicated by the folder name (0.3-3) and (1-5) in Mpc/h.
The median measurements of the samples are provided. 

You can find all measurements (including removing other corrections, such as bias evolution) under the distributions.npz file :
-cross assumes only dark matter evolution
-nz_bs uses spectroscopic bias evolution + dark matter evolution
-nz_bs_bp uses spectroscopic bias evolution + photometric bias evolution + dark matter evolution
-nz_bs_bp_mag uses all available corrections, including magnification corrections.

You can also find measurements for the expectations of each sample under expectations.npz
Finally, the measurements themselves used to obtain the model are stored under merged_res_norm_*.npz.
The metadata files offer more insights into the measurements.

One can find all of the samples in the .sacc and .fits files in the nz_multirank format expected by CosmoSIS for parameter inference.


