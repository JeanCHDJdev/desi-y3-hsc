from argparse import ArgumentParser
import subprocess
import os
from config import CorrelationConfig


class CorrelationJob:
    def __init__(
        self,
        target1,
        target2,
        area,
        job_name,
        output_dir,
        j_flag,
        nsamples,
        estimator,
        resolution,
        sample_rate_1,
        sample_rate_2,
        weight,
        k_flag,
        z_bin_flag,
        sims,
        corr_type,
        log_flag,
        c_flag,
        config_path,
    ):
        self.target1 = target1
        self.target2 = target2
        self.area = area
        self.job_name = job_name
        self.output_dir = output_dir
        self.j_flag = j_flag
        self.nsamples = nsamples
        self.estimator = estimator
        self.resolution = resolution
        self.sample_rate_1 = sample_rate_1
        self.sample_rate_2 = sample_rate_2
        self.weight = weight
        self.k_flag = k_flag
        self.z_bin_flag = z_bin_flag
        self.sims = sims
        self.corr_type = corr_type
        self.log_flag = log_flag
        self.c_flag = c_flag
        self.config_path = config_path

    def build_command(self, python_executable="python"):
        # builds 1 command for 1 config for correlation
        return (
            f"srun -N 1 -n 1 --exclusive -J {self.job_name} {python_executable} run_1_corr.py "
            f"-t1 {self.target1} "
            f"-t2 {self.target2} "
            f"-a {str(self.area)} "
            f"{self.j_flag} "
            f"-ns {self.nsamples} "
            f"-o {self.output_dir} "
            f"-e {self.estimator} "
            f"-re {self.resolution} "
            f"-r1 {self.sample_rate_1} "
            f"-r2 {self.sample_rate_2} "
            f"-w {self.weight} "
            f"{self.k_flag} "
            f"{self.z_bin_flag} "
            f"-s {self.sims} "
            f"-d {self.corr_type} "
            f"{self.log_flag} "
            f"{self.c_flag} "
            f"-C {self.config_path}"
        )


class CorrelationLauncher:
    def __init__(self, params, config, config_path):
        # import some parameters from the yaml config file
        self.params = params
        self.config = config
        self.config_path = config_path
        self.tracers_cross = params['tpcf']['cross_correlations']
        self.tracers_auto = params['tpcf']['auto_correlations']
        self.areas = params['tpcf']['areas']
        self.advanced = params['advanced']

    def create_job_name(self):
        # creates the name of the main job depending on the configuration
        tracer_cross = (
            f"cross={{{'_'.join(self.tracers_cross)}}}_" if self.tracers_cross else ""
        )
        tracer_auto = (
            f"auto={{{'_'.join(self.tracers_auto)}}}_" if self.tracers_auto else ""
        )
        bins = (
            f"bins={{{'_'.join(map(str, self.params['bins']['hsc'].keys()))}}}_"
            if self.tracers_cross
            else ""
        )
        areas_cross = f"areas_cross={{{'_'.join(map(str, self.areas['cross']))}}}_" if self.tracers_cross else "" 
        areas_auto = f"areas_auto={{{'_'.join(map(str, self.areas['auto']))}}}_" if self.tracers_auto else ""
        version = f"version={self.config.desi_data_release}"
        cat = f"cat={self.config.hsc_catalog}"
        custom_name = ""
        if self.params['bash']['job_name'] != "":
            custom_name = f"{self.params['bash']['job_name']}_"
        return f"{custom_name}{tracer_cross}{tracer_auto}{bins}{areas_cross}{areas_auto}{version}_{cat}"

    def make_output_dir(self, default_path):
        # if not specified output_dir, choose the input directory for n(z) plots as output directory for correlations
        if self.advanced.get('output_dir') == 'None':
            return default_path
        return self.advanced['output_dir']

    def build_jobs(self):
        # builds the commands for all the correlation jobs
        j_flag = "-j" if self.params['tpcf']['jackknife'] else ""
        z_bin_flag_cross = "-z" if self.advanced['z_bin'][0] else ""
        z_bin_flag_auto = "-z" if self.advanced['z_bin'][1] else ""
        log_flag = "-l" if self.advanced['log'] else ""
        c_flag = f"-c {self.advanced['nproc']}" if self.advanced['nproc'] != 'None' else ""
        k_flag_cross = "-k" if self.advanced['skip_moc'][0] else ""
        k_flag_auto = "-k" if self.advanced['skip_moc'][1] else ""

        jobs = []
        custom_name = ""
        if self.params['bash']['job_name'] != "":
            custom_name = f"{self.params['bash']['job_name']}_"
        
        for tracer in self.tracers_cross:
            for area in self.areas["cross"]:
                out_dir = self.make_output_dir(f"../outputs/{self.config.desi_data_release}/cross/")
                jobs.append(
                    CorrelationJob(
                        target1=tracer,
                        target2="HSC",
                        area=str(area),
                        job_name=f"{custom_name}{tracer}xHSC_bins={{{'_'.join(map(str, self.params['bins']['hsc'].keys()))}}}_area={area}_version={self.config.desi_data_release}_cat={self.config.hsc_catalog}",
                        output_dir=out_dir,
                        j_flag=j_flag,
                        nsamples=self.params['tpcf']['nsamples'],
                        estimator=self.advanced['estimator'],
                        resolution=self.advanced['resolution'],
                        sample_rate_1=self.advanced['sample_rate_1'],
                        sample_rate_2=self.advanced['sample_rate_2'],
                        weight=self.advanced['weight'],
                        k_flag=k_flag_cross,
                        z_bin_flag=z_bin_flag_cross,
                        sims=self.advanced['sims'],
                        corr_type=self.advanced['corr_type'],
                        log_flag=log_flag,
                        c_flag=c_flag,
                        config_path=self.config_path,
                    )
                )
            
        for tracer in self.tracers_auto:
            for area in self.areas["auto"]:
                if area in [1, 4]:
                    out_dir = self.make_output_dir(f"../outputs/{self.config.desi_data_release}/autos_NGC")
                    jobs.append(
                        CorrelationJob(
                            target1=tracer,
                            target2=tracer,
                            area=str(area),
                            job_name=f"{custom_name}{tracer}x{tracer}_area={area}_version={self.config.desi_data_release}",
                            output_dir=out_dir,
                            j_flag=j_flag,
                            nsamples=self.params['tpcf']['nsamples'],
                            estimator=self.advanced['estimator'],
                            resolution=self.advanced['resolution'],
                            sample_rate_1=self.advanced['sample_rate_1'],
                            sample_rate_2=self.advanced['sample_rate_2'],
                            weight=self.advanced['weight'],
                            k_flag=k_flag_auto,
                            z_bin_flag=z_bin_flag_auto,
                            sims=self.advanced['sims'],
                            corr_type=self.advanced['corr_type'],
                            log_flag=log_flag,
                            c_flag=c_flag,
                            config_path=self.config_path,
                        )
                    )
            
                elif area in [2, 3]:
                    out_dir = self.make_output_dir(f"../outputs/{self.config.desi_data_release}/autos_SGC")
                    jobs.append(
                        CorrelationJob(
                            target1=tracer,
                            target2=tracer,
                            area=str(area),
                            job_name=f"{custom_name}{tracer}x{tracer}_area={area}_version={self.config.desi_data_release}",
                            output_dir=out_dir,
                            j_flag=j_flag,
                            nsamples=self.params['tpcf']['nsamples'],
                            estimator=self.advanced['estimator'],
                            resolution=self.advanced['resolution'],
                            sample_rate_1=self.advanced['sample_rate_1'],
                            sample_rate_2=self.advanced['sample_rate_2'],
                            weight=self.advanced['weight'],
                            k_flag=k_flag_auto,
                            z_bin_flag=z_bin_flag_auto,
                            sims=self.advanced['sims'],
                            corr_type=self.advanced['corr_type'],
                            log_flag=log_flag,
                            c_flag=c_flag,
                            config_path=self.config_path,
                        )
                    )
        return jobs

    def build_interactive_command(self, jobs):
        # builds command to launch in interactive session
        if not jobs:
            return None
        python_exec = "python"
        commands = " & ".join(job.build_command(python_executable=python_exec) for job in jobs)
        return (
            f"salloc -N {len(jobs)} -C {self.params['bash']['proc']} -t {self.params['bash']['time']} "
            f"-q interactive -A {self.params['bash']['account']} -J {self.create_job_name()} "
            f"bash -lc '{commands} & wait'"
        )

    def write_batch_script(self, job_name):
        # builds the bash script for regular session
        bash_template = f"""#!/bin/bash
#SBATCH --job-name={job_name}                   # Job name
#SBATCH --constraint={self.params['bash']['proc']}
#SBATCH -q regular                              # Run in regular queue
#SBATCH --nodes={len(self.build_jobs())}                        # Number of nodes to use
#SBATCH --ntasks-per-node={self.params['bash']['ntasks_per_node']}                     # Number of tasks (processes) per node
#SBATCH --cpus-per-task={self.params['bash']['cpus_per_task']}                     # Number of proc cores per task
#SBATCH --mem={self.params['bash']['memory']}                              # Job memory request (per node or total)
#SBATCH --time={self.params['bash']['time']}                         # Wall clock limit (D-HH:MM:SS)
#SBATCH --account={self.params['bash']['account']}                          # Account name (for billing)
#SBATCH --output=../logs/%j.out                 # Standard output file name (%j expands to jobID)
#SBATCH --error=../logs/%j.err                  # Standard error file name (%j expands to jobID)

# Record the start time
START_TIME=$(date +"%Y%m%d_%H%M%S")

# Load python module and activate conda environment
MY_PYTHON="$HOME/miniforge3/envs/desi/bin/python"

# Run programs in parallel with srun
"""
        jobs = self.build_jobs()
        for job in jobs:
            bash_template += job.build_command(python_executable="$MY_PYTHON") + " &\n"
        bash_template += "\nwait\n\n"
        bash_template += (
            "mv \"../logs/${SLURM_JOB_ID}.out\" \"../logs/${START_TIME}_${SLURM_JOB_ID}.out\"\n"
            "mv \"../logs/${SLURM_JOB_ID}.err\" \"../logs/${START_TIME}_${SLURM_JOB_ID}.err\"\n"
        )
        with open(f"launch_corr_jobs_{self.config_path[:-4]}.sh", "w") as f:
            f.write(bash_template)

    def launch(self):
        # launches either in interactive or regular depending on the config
        jobs = self.build_jobs()
        if self.params['bash']['qos'] == "interactive":
            command = self.build_interactive_command(jobs)
            if not command:
                print("No interactive jobs to launch.")
                return
            subprocess.run(command, shell=True, check=True)
            return
        if self.params['bash']['qos'] == "regular":
            self.write_batch_script(self.create_job_name())
            subprocess.run(f"sbatch launch_corr_jobs_{self.config_path[:-4]}.sh", shell=True, check=True)
            # if os.path.exists(f"./launch_corr_jobs_{self.config_path[:-4]}.sh"):
            #     os.remove(f"./launch_corr_jobs_{self.config_path[:-4]}.sh")
            return
        raise ValueError(f"Invalid qos parameter {self.params['bash']['qos']}. Should be 'interactive' or 'regular'.")

def parse_args():
    # parses the command line arguments
    parser = ArgumentParser()
    
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        default="params.yml", 
        help="Path to input file. Default is params.yml"
        )
    
    return parser.parse_args()

def main():
    # main function
    args = parse_args()
    try:
        config = CorrelationConfig.load(args.config)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return

    launcher = CorrelationLauncher(params=config.params, config=config, config_path=args.config)
    try:
        launcher.launch()
    except subprocess.CalledProcessError as err:
        print(f"Error occurred while launching jobs: code {err.returncode}, message: {err.output}")
    except AssertionError as err:
        print(f"Configuration validation failed: {err}")

if __name__ == "__main__":
    # run the main function
    main()