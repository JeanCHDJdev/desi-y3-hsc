#!/bin/bash
#SBATCH --job-name=correlation_computation      # Job name
#SBATCH --constraint=cpu                        # Run on CPU nodes
#SBATCH -q regular                              # Quality of Service (QoS) name
#SBATCH --nodes=2                               # Request one node
#SBATCH --ntasks=2                              # Number of tasks (processes)
#SBATCH --cpus-per-task=256                     # Number of CPU cores per task
#SBATCH --mem=128G                              # Job memory request (per node or total)
#SBATCH --time=03:00:00                         # Wall clock limit (D-HH:MM:SS)
#SBATCH --account=desi                          # Account name (for billing)
#SBATCH --output=logs/%j.out                    # Standard output file name (%j expands to jobID)
#SBATCH --error=logs/%j.err                     # Standard error file name (%j expands to jobID)

# Record the start time
START_TIME=$(date +"%Y%m%d_%H%M%S")

# Load python module and activate conda environment
MY_PYTHON="$HOME/miniforge3/envs/desi/bin/python"

# Run programs in parallel with srun
tracers=("QSO")

for tracer in "${tracers[@]}"; do
    # cross-correlation with HSC
    srun -n 1 --nodes=1 --exclusive $MY_PYTHON run_corr.py -t1 $tracer -t2 HSC -c 256 -j -ns 16 -o outputs/dr2/cross/ &
    # auto-correlation
    srun -n 1 --nodes=1 --exclusive $MY_PYTHON run_corr.py -t1 $tracer -t2 $tracer -c 256 -j -ns 16 -o outputs/dr2/autos_NGC &
done

# Wait for the end of all processes
wait

# Rename log files with the start time
mv "./logs/${SLURM_JOB_ID}.out" "./logs/${START_TIME}_${SLURM_JOB_ID}.out"
mv "./logs/${SLURM_JOB_ID}.err" "./logs/${START_TIME}_${SLURM_JOB_ID}.err"

# Move outputs files in right folder
for tracer in "${tracers[@]}"; do
    mv "./outputs/dr2/autos_NGC/${tracer}x${tracer}/*_moc2.npy" "./outputs/dr2/autos_SGC/${tracer}x${tracer}"
    mv "./outputs/dr2/autos_NGC/${tracer}x${tracer}/*_moc3.npy" "./outputs/dr2/autos_SGC/${tracer}x${tracer}"
done