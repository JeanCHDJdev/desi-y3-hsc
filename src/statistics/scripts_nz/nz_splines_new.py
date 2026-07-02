import numpy as np
import importlib

from pathlib import Path
from typing import Any
from argparse import ArgumentParser
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from config import CorrelationConfig

cf: Any = None
spline: Any = None

class SplinePipeline:
    """Pipeline for regularized n(z) distribution with splines using configuration-driven parameters."""
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.params = config.params
        self.username = config.username
        self.hsc_catalog = config.hsc_catalog
        self.stem = config.desi_data_release
        self.version = self.params['nz_splines']['version']
        self.tomo_to_tracer = self.params['nz_splines']['tomo_to_tracer']
        self.patches = self.params['nz_splines']['areas']
        self.tomo_bins = self.params['nz_splines']['tomo_bins']
        self.scale_cut = self.params['nz_splines']['scale_cut']
        self.names = ['npz_cross', 'npz_bs', 'npz_bs_bp', 'npz_bs_bp_mag']
        self.imag_cut = None if self.params['nz_splines']['imag_cut'] == 'None' else self.params['nz_splines']['imag_cut']
        self.root = Path(f"/global/cfs/projectdirs/desi/users/{self.username}/desi-y3-hsc/")
        # Hyperparameter
        self.n_tune = self.params['nz_splines']['n_tune']
        self.n_samples = self.params['nz_splines']['n_samples']
        self.target_accept = self.params['nz_splines']['target_accept']
        self.prior_concentration = self.params['nz_splines']['prior_concentration']
        self.base_alpha = self.params['nz_splines']['base_alpha']


    def distribution_identifier(self, name=None, tomo=None):
        """Loads the cross-correlation distribution data."""
        tracers = []
        order = ["BGS_ANY", "LRG", "ELGnotqso", "QSO"]
          
        title = self.params['nz_splines']['title']
        if title != "":
            title += "_"
        
        identifier = f'mocs={"".join(map(str, self.patches))}_icut={self.imag_cut}'
        
        if name == None and tomo == None:
            for tomo_bin in self.tomo_bins:
                for tracer in self.tomo_to_tracer[tomo_bin]:
                    if tracer not in tracers:
                        tracers.append(tracer)
            tracers = sorted(tracers, key=order.index)
            
            suffix = f'{title}nz_tracers={",".join(tracers)}_bins={"".join(map(str, self.tomo_bins))}_scale_cut={self.scale_cut[0]}-{self.scale_cut[1]}_{identifier}_{self.stem}_{self.hsc_catalog}_{self.version}.npz'
        
        else:
            for tracer in self.tomo_to_tracer[tomo]:
                if tracer not in tracers:
                    tracers.append(tracer)
            tracers = sorted(tracers, key=order.index)
            
            suffix = f'{title}{name}_tracers={",".join(tracers)}_bin={str(tomo)}_scale_cut={self.scale_cut[0]}-{self.scale_cut[1]}_{identifier}_{self.stem}_{self.hsc_catalog}_{self.version}'
        
        return suffix


    def load_distribution_data(self):
        """Loads the cross-correlation distribution data."""
        
        suffix = self.distribution_identifier()
        data_path = self.root / 'src' / 'statistics' / 'scripts_nz' / 'nz_results' / suffix
        
        print(f"Processing data file: {suffix}")
        
        return np.load(data_path)
   
   
    def run(self):
        """Main execution logic looping over configurations."""
        importlib.reload(spline)
        
        data = self.load_distribution_data()
        dir_splines = self.root / 'src' / 'statistics' / 'scripts_nz' / 'splines_results'
        # if not dir_splines.exists():
        #     dir_splines.mkdir(parents=True)
        current = 0
        for name in self.names:
            for tomo in self.tomo_bins:
                current += 1
                total = len(self.names) * len(self.tomo_bins)
                print(f"Processing correction {name}, bin {tomo} : {100*current/total:.3g}% done.")
                
                savefile = str(dir_splines / self.distribution_identifier(name=name, tomo=tomo))
                npz_arr = data[f"{tomo}/{name}"]
                npz_arr_err = data[f"{tomo}/{name}_err"]
                z = data[f"{tomo}/{name}_z"]

                # if Path(f"{savefile}.nc").exists():
                #     print(f"Skipping {savefile}, already exists")
                #     continue

                if tomo >= 6:
                    spl = spline.BayesianBSpline(zv=z, n_knots=int(len(z)//3))
                else:
                    spl = spline.BayesianBSpline(zv=z, n_knots=int(len(z)//2))
                spl.fit(
                    npz_arr,
                    npz_arr_err,
                    n_tune=self.n_tune,
                    n_samples=self.n_samples,
                    target_accept=self.target_accept,
                    prior_concentration=self.prior_concentration,
                    base_alpha=self.base_alpha,
                )
                spl.save_model(savefile)
  
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='params.yml',
        help='Path to input file. Default is params.yml',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = CorrelationConfig.load(args.config)
    except FileNotFoundError as err:
        print(f'Error: {err}')
        return

    import corrfiles_new as cf
    import spline_new as spline

    globals().update({'cf': cf, 'spline': spline})

    pipeline = SplinePipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()