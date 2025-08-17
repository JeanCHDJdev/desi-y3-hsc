#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 4
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name real
#SBATCH --output /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic-shear/logs/cshear-%j.out
#SBATCH --error /global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/cosmic-shear/logs/cshear-%j.err

umask 0002
export TQDM_MININTERVAL=120 # export to two minutes

source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3
cd /global/cfs/projectdirs/desi/users/jchdj/cosmosis-standard-library/
command="cosmosis --mpi /global/cfs/projectdirs/desi/users/jchdj/cosmosis-standard-library/examples/hsc-y3-shear-real.ini"
srun -n 512 $command
