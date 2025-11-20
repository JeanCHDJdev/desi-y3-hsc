#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 4
#SBATCH --time 5:00:00
#SBATCH --constraint cpu
#SBATCH --account m3058
#SBATCH --job-name real-1-5-fiducial
#SBATCH --output /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic_shear/logs/cshear-real-1-5-fiducial-%j.out
#SBATCH --error /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic_shear/logs/cshear-real-1-5-fiducial-%j.err

umask 0002
export TQDM_MININTERVAL=240 # export to four minutes

source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3
cd /global/cfs/projectdirs/desi/users/jchdj/software/cosmosis-standard-library/
command="cosmosis --mpi /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic_shear/config/real-1-5/fiducial/hsc-y3-shear-real.ini"
srun -n 512 $command