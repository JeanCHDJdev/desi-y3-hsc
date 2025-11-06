#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 4
#SBATCH --time 11:00:00
#SBATCH --constraint cpu
#SBATCH --account m3058
#SBATCH --job-name real
#SBATCH --output /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic_shear/logs/cshear-real-%j.out
#SBATCH --error /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic_shear/logs/cshear-real-%j.err

umask 0002
export TQDM_MININTERVAL=120 # export to two minutes

source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3
cd /global/cfs/projectdirs/desi/users/jchdj/software/cosmosis-standard-library/
command="cosmosis --mpi /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic_shear/config/real/hsc-y3-shear-real.ini"
srun -n 512 $command