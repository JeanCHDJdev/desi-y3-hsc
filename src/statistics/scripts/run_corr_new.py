import yaml
from argparse import ArgumentParser
import subprocess
from pathlib import Path

# define a function to parse the command line arguments
def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        default="params.yml", 
        help="Path to input file. Default is params.yml"
        )
    
    return parser.parse_args()

# main function
def main():
    args = parse_args()
    
    # load the input parameter file
    try:
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}.")
        return

    # import parameters from the params.yml file
    tracers_cross = params['tpcf']['cross_correlations'] # desi tracers to cross-correlate with hsc
    tracers_auto = params['tpcf']['auto_correlations'] # targets to auto-correlate
    areas = params['tpcf']['areas'] # patches of the sky
    advanced = params['advanced'] # advanced parameters for correlation

    # define flags for the different options
    k_flag_cross = "-k" if advanced['skip_moc'][0] else ""
    k_flag_auto = "-k" if advanced['skip_moc'][1] else ""
    j_flag = "-j" if params['tpcf']['jackknife'] else ""
    z_bin_flag = "-z" if advanced['z_bin'] else ""
    log_flag = "-l" if advanced['log'] else ""
    c_flag = f"-c {advanced['nproc']}" if advanced['nproc'] != 'None' else ""
    
    # create an identifier for the job based on tracers, areas, bins and input
    identifier_cross = ""
    identifier_bins = ""
    identifier_version = f"version={params['data']['desi_data_release']}"
    identifier_cat = f"cat={params['data']['hsc_catalog']}"
    if len(tracers_cross) > 0:
        identifier_cross = f"cross={{{'_'.join(tracers_cross)}}}_"
        identifier_bins = f"bins={{{'_'.join(params['bins']['hsc'].keys())}}}_"
    identifier_auto = ""
    if len(tracers_auto) > 0:
        identifier_auto = f"auto={{{'_'.join(tracers_auto)}}}_"
    job_name = f"{params['bash']['job_name']}{identifier_cross}{identifier_auto}{identifier_bins}areas={{{'_'.join(map(str, areas))}}}_{identifier_version}_{identifier_cat}"
    
    # calculation of number of jobs
    n_correlations = len(tracers_cross) + len(tracers_auto)
    n_jobs = n_correlations * len(areas)
    
    # assertions on validity of input data
    assert n_correlations > 0, "At least one of cross or auto-correlation lists should be non-empty."
    assert n_jobs > 0, "At least one area should be specified."
    for tracer in tracers_cross:
        assert tracer in ["BGS_ANY", "LRG", "ELGnotqso", "QSO"], f"Tracer {tracer} not found for cross-correlation. Should be among 'BGS_ANY', 'LRG', 'ELGnotqso', 'QSO'."
    for target in tracers_auto:
        assert target in ["BGS_ANY", "LRG", "ELGnotqso", "QSO", "HSC"], f"Target {target} not found for auto-correlation. Should be among 'BGS_ANY', 'LRG', 'ELGnotqso', 'QSO', 'HSC'."
    for area in areas:
        assert area in [1, 2, 3, 4], f"Area {area} not found. Should be among 1, 2, 3, 4."
    assert params['tpcf']['jackknife'] in [True, False], f"Jackknife parameter {params['tpcf']['jackknife']} invalid. Should be a boolean."
    assert params['tpcf']['nsamples'] > 0, f"Number of samples {params['tpcf']['nsamples']} invalid. Should be a positive integer."
    path = advanced['output_dir']
    assert path == "" or (Path(path).exists() and Path(path).is_dir()), f"{path} doesn't exist or is not a directory for output files."
    assert advanced['estimator'] in ["davispeebles", "landyszalay", "peebleshauser"], f"Estimator {advanced['estimator']} invalid. Should be among 'davispeebles', 'landyszalay', 'peebleshauser'."
    assert advanced['resolution'] > 0, f"Resolution {advanced['resolution']} invalid. Should be a positive number."
    assert advanced['sample_rate_1'] >= 0, f"Sample rate 1 {advanced['sample_rate_1']} invalid. Should be a non-negative integer."
    assert advanced['sample_rate_2'] >= 0, f"Sample rate 2 {advanced['sample_rate_2']} invalid. Should be a non-negative integer."
    assert advanced['weight'] in ["nonKP", "PIP", "base"], f"Weight {advanced['weight']} invalid. Should be among 'nonKP', 'PIP', 'base'."
    assert advanced['skip_moc'][0] in [True, False], f"skip_moc for cross-correlation {advanced['skip_moc'][0]} invalid. Should be a boolean."
    assert advanced['skip_moc'][1] in [True, False], f"skip_moc for auto-correlation {advanced['skip_moc'][1]} invalid. Should be a boolean."
    assert advanced['z_bin'] in [True, False], f"z_bin parameter {advanced['z_bin']} invalid. Should be a boolean."
    assert advanced['sims'] in [0, 1, 2, 3, 4, 5], f"Sims parameter {advanced['sims']} invalid. Should be among 0, 1, 2, 3, 4, 5."
    assert advanced['corr_type'] in ["theta", "rp"], f"Correlation type {advanced['corr_type']} invalid. Should be among 'theta', 'rp'."
    log = advanced['log']
    if isinstance(log, str):
        assert path == "" or (Path(path).exists() and Path(path).is_dir()), f"{path} doesn't exist or is not a directory for log file."
    else:
        assert not log, f"Log parameter {log} invalid. Should be the boolean False or a path to a folder."
    assert advanced['nproc'] == 'None' or (isinstance(advanced['nproc'], int) and advanced['nproc'] > 0), f"nproc parameter {advanced['nproc']} invalid. Should be a positive integer or 'None'."
    assert params['data']['hsc_catalog'] in ["hscpdr3_i25_imagerr", "hscy3_cat_withflags"], f"HSC catalog {params['data']['hsc_catalog']} invalid. Should be 'hscpdr3_i25_imagerr' or 'hscy3_cat_withflags'."
    assert params['data']['desi_data_release'] in ["dr1", "dr2"], f"DESI data release {params['data']['desi_data_release']} invalid. Should be 'dr1' or 'dr2'."

    # write the bash script to launch the jobs in parallel with srun
    bash_template = f"""#!/bin/bash
#SBATCH --job-name={job_name}                   # Job name
#SBATCH --constraint={params['bash']['proc']}                        # Run on CPU nodes
#SBATCH -q regular                              # Run in regular queue
#SBATCH --nodes={n_jobs}                        # Number of nodes to use
#SBATCH --ntasks-per-node={params['bash']['ntasks_per_node']}                     # Number of tasks (processes) per node
#SBATCH --cpus-per-task={params['bash']['cpus_per_task']}                     # Number of CPU cores per task
#SBATCH --mem={params['bash']['memory']}                              # Job memory request (per node or total)
#SBATCH --time={params['bash']['time']}                         # Wall clock limit (D-HH:MM:SS)
#SBATCH --account={params['bash']['account']}                          # Account name (for billing)
#SBATCH --output=../logs/%j.out                 # Standard output file name (%j expands to jobID)
#SBATCH --error=../logs/%j.err                  # Standard error file name (%j expands to jobID)

# Record the start time
START_TIME=$(date +"%Y%m%d_%H%M%S")

# Load python module and activate conda environment
MY_PYTHON="$HOME/miniforge3/envs/desi/bin/python"

# Run programs in parallel with srun
tracers_cross=({' '.join(tracers_cross)})
tracers_auto=({' '.join(tracers_auto)})
areas=({' '.join(map(str, areas))})

# cross-correlation with HSC
for tracer in "${{tracers_cross[@]}}"; do
    for area in "${{areas[@]}}"; do
        srun -N 1 -n 1 --exclusive $MY_PYTHON run_1_corr.py \\
        -t1 $tracer \\
        -t2 HSC \\
        -a $area \\
        {j_flag} \\
        -ns {params['tpcf']['nsamples']} \\
        -o {advanced['output_dir']} \\
        -e {advanced['estimator']} \\
        -re {advanced['resolution']} \\
        -r1 {advanced['sample_rate_1']} \\
        -r2 {advanced['sample_rate_2']} \\
        -w {advanced['weight']} \\
        {k_flag_cross} \\
        {z_bin_flag} \\
        -s {advanced['sims']} \\
        -d {advanced['corr_type']} \\
        {log_flag} \\
        {c_flag} \\
        &
    done
done

# auto-correlation
for tracer in "${{tracers_auto[@]}}"; do
    for area in "${{areas[@]}}"; do
        srun -N 1 -n 1 --exclusive $MY_PYTHON run_1_corr.py \\
        -t1 $tracer \\
        -t2 $tracer \\
        -a $area \\
        {j_flag} \\
        -ns {params['tpcf']['nsamples']} \\
        -o {advanced['output_dir']} \\
        -e {advanced['estimator']} \\
        -re {advanced['resolution']} \\
        -r1 {advanced['sample_rate_1']} \\
        -r2 {advanced['sample_rate_2']} \\
        -w {advanced['weight']} \\
        {k_flag_auto} \\
        {z_bin_flag} \\
        -s {advanced['sims']} \\
        -d {advanced['corr_type']} \\
        {log_flag} \\
        {c_flag} \\
        &
    done
done

# Wait for the end of all processes
wait

# Rename log files with the start time
mv "../logs/${{SLURM_JOB_ID}}.out" "../logs/${{START_TIME}}_${{SLURM_JOB_ID}}.out"
mv "../logs/${{SLURM_JOB_ID}}.err" "../logs/${{START_TIME}}_${{SLURM_JOB_ID}}.err"
"""

    # write the new bash content
    with open("launch_corr_jobs.sh", "w") as f:
        f.write(bash_template)
    
    # hsc bins
    bins_hsc = params['bins']['hsc']
    sorted_keys = sorted(bins_hsc.keys())
    assert len(sorted_keys) == sorted_keys[-1] - sorted_keys[0] + 1, "HSC bins should be consecutive bins"
    for key in sorted_keys:
        if key != sorted_keys[-1]: # check that bins are consecutive and valid
            assert bins_hsc[key][1] == bins_hsc[key + 1][0], f"Bin {key} and bin {key + 1} are not consecutive"
        assert len(bins_hsc[key]) == 2, f"Bin {key} should have exactly 2 edges : [start, stop]"
        assert 0 <= bins_hsc[key][0] < bins_hsc[key][1], f"Bin {key} has invalid edges: {bins_hsc[key]}"
    bins_hsc = [bins_hsc[key][0] for key in sorted_keys] + [bins_hsc[sorted_keys[-1]][1]] if len(bins_hsc) > 0 else []
    
    for tracer in ["BGS_ANY", "LRG", "ELGnotqso", "QSO"]:
        assert len(params['bins']['desi'][tracer]) == 3, f"Bins for tracer {tracer} should have 3 parameters : [start, stop, step]"
        assert 0 <= params['bins']['desi'][tracer][0] < params['bins']['desi'][tracer][1], f"Bins for tracer {tracer} have invalid start and stop values : {params['bins']['desi'][tracer]}"
        assert params['bins']['desi'][tracer][2] > 0, f"Step value for tracer {tracer} invalid : should be > 0."
    
    # modification of config_loader.py file
    # this script will load the parameters used in the different python scripts
    config_loader_template = f"""import numpy as np
username = '{params['username']}'
bins_bgs = np.arange({params['bins']['desi']['BGS_ANY'][0]}, {params['bins']['desi']['BGS_ANY'][1] + params['bins']['desi']['BGS_ANY'][2]:.2f}, {params['bins']['desi']['BGS_ANY'][2]})
bins_lrg = np.arange({params['bins']['desi']['LRG'][0]}, {params['bins']['desi']['LRG'][1] + params['bins']['desi']['LRG'][2]:.2f}, {params['bins']['desi']['LRG'][2]})
bins_elg = np.arange({params['bins']['desi']['ELGnotqso'][0]}, {params['bins']['desi']['ELGnotqso'][1] + params['bins']['desi']['ELGnotqso'][2]:.2f}, {params['bins']['desi']['ELGnotqso'][2]})
bins_qso = np.arange({params['bins']['desi']['QSO'][0]}, {params['bins']['desi']['QSO'][1] + params['bins']['desi']['QSO'][2]:.2f}, {params['bins']['desi']['QSO'][2]})
bins_hsc = np.array({bins_hsc})
hsc_catalog = '{params['data']['hsc_catalog']}'
desi_version = 'DR{params['data']['desi_data_release'][-1]}'
"""
    
    # write the new parameter content
    with open("config_loader.py", "w") as f:
        f.write(config_loader_template)
    
    # launch the bash script
    if params['bash']['qos'] == "interactive" and n_correlations == 1: # run in interactive mode with srun if 1 correlation to run and asked by user
        
        # adaptation of the parameters according to the type of correlation to run
        if len(tracers_cross) == 1:
            target1 = tracers_cross[0]
            target2 = "HSC"
            k_flag = k_flag_cross
        else:
            target1 = tracers_auto[0]
            target2 = target1
            k_flag = k_flag_auto

        # command to run the correlation with srun
        cmd = (f"salloc -N 1 -C cpu -t {params['bash']['time']} -q interactive -A desi -J {job_name} "
               f"srun -J {job_name} python run_1_corr.py " 
               f"-t1 {target1} " 
               f"-t2 {target2} " 
               f"-a {' '.join(map(str, areas))} " 
               f"{j_flag} " 
               f"-ns {params['tpcf']['nsamples']} "
               f"-o {advanced['output_dir']} "
               f"-e {advanced['estimator']} "
               f"-re {advanced['resolution']} "
               f"-r1 {advanced['sample_rate_1']} "
               f"-r2 {advanced['sample_rate_2']} "
               f"-w {advanced['weight']} "
               f"{k_flag} "
               f"{z_bin_flag} "
               f"-s {advanced['sims']} "
               f"-d {advanced['corr_type']} "
               f"{log_flag} "
               f"{c_flag} "
        )
        
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing command: code {e.returncode}, message: {e.output}")
            return
    
    elif params['bash']['qos'] == "regular" or n_correlations > 1: # run in regular mode with sbatch if asked by user or more than 1 correlation to run
        
        if params['bash']['qos'] == "interactive":
            print("Warning: More than 1 correlation to run, launching in regular mode instead of interactive mode.")
        
        # launch the bash script with sbatch
        bash_file = "launch_corr_jobs.sh"
        try:
            subprocess.run(f"sbatch {bash_file}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing {bash_file}: code {e.returncode}, message: {e.output}")
            return
    else:
        print(f"Error: Invalid qos parameter {params['bash']['qos']}. Should be 'interactive' or 'regular'.")
        return
    
# run the main function
if __name__ == "__main__":
    main()