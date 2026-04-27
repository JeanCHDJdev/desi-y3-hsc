#!/bin/bash
#SBATCH --job-name=auto={BGS_ANY}_areas={1_3}_version=dr2_cat=hscpdr3_i25_imagerr                   # Job name
#SBATCH --constraint=cpu                        # Run on CPU nodes
#SBATCH -q regular                              # Run in regular queue
#SBATCH --nodes=2                        # Number of nodes to use
#SBATCH --ntasks-per-node=1                     # Number of tasks (processes) per node
#SBATCH --cpus-per-task=256                     # Number of CPU cores per task
#SBATCH --mem=128GB                              # Job memory request (per node or total)
#SBATCH --time=02:00:00                         # Wall clock limit (D-HH:MM:SS)
#SBATCH --account=desi                          # Account name (for billing)
#SBATCH --output=../logs/%j.out                 # Standard output file name (%j expands to jobID)
#SBATCH --error=../logs/%j.err                  # Standard error file name (%j expands to jobID)

# Record the start time
START_TIME=$(date +"%Y%m%d_%H%M%S")

# Load python module and activate conda environment
MY_PYTHON="$HOME/miniforge3/envs/desi/bin/python"

# Run programs in parallel with srun
tracers_cross=()
tracers_auto=(BGS_ANY)
areas=(1 3)

# cross-correlation with HSC
for tracer in "${tracers_cross[@]}"; do
    for area in "${areas[@]}"; do
        srun -N 1 -n 1 --exclusive $MY_PYTHON run_1_corr.py \
        -t1 $tracer \
        -t2 HSC \
        -a $area \
        -j \
        -ns 64 \
        -o ../crosscorr/new/ \
        -e davispeebles \
        -re 256 \
        -r1 1 \
        -r2 1 \
        -w nonKP \
         \
         \
        -s 0 \
        -d theta \
         \
         \
        &
    done
done

# auto-correlation
for tracer in "${tracers_auto[@]}"; do
    for area in "${areas[@]}"; do
        srun -N 1 -n 1 --exclusive $MY_PYTHON run_1_corr.py \
        -t1 $tracer \
        -t2 $tracer \
        -a $area \
        -j \
        -ns 64 \
        -o ../crosscorr/new/ \
        -e davispeebles \
        -re 256 \
        -r1 1 \
        -r2 1 \
        -w nonKP \
        -k \
         \
        -s 0 \
        -d theta \
         \
         \
        &
    done
done

# Wait for the end of all processes
wait

# Rename log files with the start time
mv "../logs/${SLURM_JOB_ID}.out" "../logs/${START_TIME}_${SLURM_JOB_ID}.out"
mv "../logs/${SLURM_JOB_ID}.err" "../logs/${START_TIME}_${SLURM_JOB_ID}.err"
