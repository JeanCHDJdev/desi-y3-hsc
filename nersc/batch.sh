#!/bin/bash
#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 14:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xcorrqso
#SBATCH --output /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/nersc/logs/qso-%j.out
#SBATCH --error /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/ners/logs/qso-%j.err

echo "Loading DESI environment"
conda activate desi

echo "Moving to the working directory"
cdstat

echo "Running cross-correlation script"
srun -n 1 python run_corr.py -t1 QSO -t2 HSC -j -o /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/src/statistics/crosscorr_theta/corr34_use_zbin -z -rh 18 -rd 5