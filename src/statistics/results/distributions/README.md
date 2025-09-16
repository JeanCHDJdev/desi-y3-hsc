# results
---------

In this folder, you will find the fiducial distributions in fits files. Scales are 0.3-3 h-1Mpc. 
The median measurements of the samples and the mean are provided under the according names. We also provide the distributions
with and without magnification corrections. 

You can find all measurements (including removing other corrections, such as bias evolution) under the distributions.npz file :
-cross assumes only dark matter evolution
-nz_bs uses spectroscopic bias evolution + dark matter evolution
-nz_bs_bp uses spectroscopic bias evolution + photometric bias evolution + dark matter evolution
-nz_bs_bp_mag uses all available corrections, including magnification corrections.

You can also find measurements for the expectations of each sample under expectations.npz
Finally, the measurements themselves used to obtain the model are stored under merged_res_norm_*.npz.


