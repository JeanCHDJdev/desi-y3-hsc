import numpy as np
import importlib
import importlib.util
import matplotlib.pyplot as plt
import pandas as pd
import json
import scipy.interpolate as interp
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import shutil
import math
from typing import Any

from config import CorrelationConfig
from IPython.display import display

cf: Any = None
ct: Any = None
inference: Any = None


class NzPipeline:
    """Pipeline for n(z) production using configuration-driven parameters."""

    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.params = config.params
        self.username = config.username
        self.hsc_catalog = config.hsc_catalog
        self.stem = config.desi_data_release
        self.version = self.params['nz']['version']
        self.tomo_to_tracer = self.params['nz']['tomo_to_tracer']
        self.patches = self.params['nz']['areas']
        self.tomo_bins = self.params['nz']['tomo_bins']
        self.scale_cut = self.params['nz']['scale_cut']
        self.imag_cut = None if self.params['nz']['imag_cut'] == 'None' else self.params['nz']['imag_cut']
        self.root = Path(f"/global/cfs/projectdirs/desi/users/{self.username}/desi-y3-hsc/")
        self.corr_root = self.root / 'src' / 'statistics' / 'outputs'
        self.results_root = self.root / 'results'
        self.bins_bgs = config.bins_bgs
        self.bins_lrg = config.bins_lrg
        self.bins_elg = config.bins_elg
        self.bins_qso = config.bins_qso
        self.bins_hsc = config.bins_hsc
        self.bounds = self.params['nz']['bounds']
        Path(self.results_root).mkdir(parents=True, exist_ok=True)

    def _build_path_dictionary(self, stem=None):
        stem = self.stem if stem is None else stem
        return {
            'HSC': self.corr_root / 'v12_correction' / 'autos_HSC',
            'DESI_NGC': self.corr_root / stem / 'autos_NGC',
            'DESI_SGC': self.corr_root / stem / 'autos_SGC',
            'DESIxHSC': self.corr_root / stem / 'cross',
            'MergedxMerged': self.corr_root / f'merged_{stem}_{self.version}',
            'MergedxHSC': self.corr_root / f'merged_{stem}_{self.version}',
        }

    # def _copy_bins_file(self):
    #     source_file = self._find_source_bins_file()
    #     target_file = self.corr_root / self.stem / 'cross' / 'bins' / 'bins_all.npz'
    #     shutil.copy2(source_file, target_file)

    # def _find_source_bins_file(self):
    #     if len(self.tomo_bins) == 1 and self.tomo_bins[0] in [5, 6, 7]:
    #         return self.corr_root / self.stem / 'cross' / 'bins' / f'bins_all_{self.tomo_bins[0]}.npz'
    #     elif self.tomo_bins == [5, 6, 7]:
    #         return self.corr_root / self.stem / 'cross' / 'bins' / 'bins_all_5-7.npz'
    #     elif self.tomo_bins == [1, 2, 3, 4]:
    #         return self.corr_root / self.stem / 'cross' / 'bins' / 'bins_all_1-4.npz'
    #     raise ValueError(f'Unsupported tomo_bins configuration: {self.tomo_bins}')

    def _save_bins(self):
        """Sauvegarde tous les bins calculés dans un fichier npz."""
        
        output_roots = {
            'cross': f"../outputs/{self.stem}/cross/",
            'autos_NGC': f"../outputs/{self.stem}/autos_NGC/",
            'autos_SGC': f"../outputs/{self.stem}/autos_SGC/",
            'autos_HSC': f"../outputs/v12_correction/autos_HSC/",
            }
        
        # Bins géométriques standards
        bins_rp = np.logspace(math.log(0.1, 10), math.log(10, 10), 33, base=10)
        bins_rppi_s = np.linspace(0.0, 200.0, 51)
        bins_rppi_mu = np.linspace(-100, 100, 21)

        bins_all = {
            "BGS_ANY": self.bins_bgs,
            "LRG": self.bins_lrg,
            "ELGnotqso": self.bins_elg,
            "QSO": self.bins_qso,
            "HSC": self.bins_hsc,
            "rp": bins_rp,
            "rppi_s": bins_rppi_s,
            "rppi_mu": bins_rppi_mu
        }
        
        for corr in output_roots:
            # Construction du chemin
            bin_dir = Path(output_roots[corr]) / "bins"
            bin_dir.mkdir(parents=True, exist_ok=True)
            outfile = bin_dir / "bins_all.npz"

            np.savez(outfile, **bins_all)
        #return outfile

    def _precompute_wdm(self):
        importlib.reload(inference)
        fr = cf.CorrFileReader(self._build_path_dictionary()['DESIxHSC'])
        bins_z_spectro = inference._get_fine_redshift_bins(fr=fr, tracer='Merged')
        vals_z_spectro = (bins_z_spectro[:-1] + bins_z_spectro[1:]) / 2
        vals_z_wdm = np.linspace(0.01, 3, 150)
        rp_wdm = np.linspace(self.scale_cut[0], self.scale_cut[1], 100)
        wdm_values = np.array(
            [
                ct.w_dm(rp_vals=rp_wdm, z=z, integrate=True)
                for z in vals_z_wdm
            ]
        )
        return interp.interp1d(vals_z_wdm, wdm_values, bounds_error=False, fill_value='extrapolate')

    def _run_merge_estimators(self):
        importlib.reload(inference)
        for tomo in self.tomo_bins:
            path_dictionary = self._build_path_dictionary()
            inference.merge_estimators(
                path_dictionary=path_dictionary,
                which_tomo=[tomo],
                which_patches=self.patches,
                outdir=path_dictionary['MergedxMerged'],
                which_tracers=self.tomo_to_tracer[tomo],
                data_release=self.stem,
                hsc_cat=self.hsc_catalog,
            )

    def _build_data_frame(self, wdm_interpolator):
        npzs = [{k: [] for k in self.tomo_bins} for _ in range(4)]
        npz_errs = [{k: [] for k in self.tomo_bins} for _ in range(4)]
        zvals = [{k: [] for k in self.tomo_bins} for _ in range(4)]

        for ic, condition in enumerate(
            [
                (False, False, False),
                (False, True, False),
                (True, True, False),
                (True, True, True),
            ]
        ):
            do_phot_correction, do_spec_correction, do_mag = condition
            for tomo in self.tomo_bins:
                path_dictionary = self._build_path_dictionary()
                fr = cf.CorrFileReader(path_dictionary['DESIxHSC'])

                if condition == (False, False, False):
                    meas = inference.full_npz_tomo(
                        path_dictionary=path_dictionary,
                        do_phot_correction=do_phot_correction,
                        do_spec_correction=do_spec_correction,
                        scale_cuts=self.scale_cut,
                        data_release=self.stem,
                        hsc_cat=self.hsc_catalog,
                        tomo_bin=tomo,
                        tracer=self.tomo_to_tracer[tomo],
                        which_patches=None,
                        precomp_wdm=wdm_interpolator,
                        mode='Merged',
                    )
                    npzs[ic][tomo].append(meas[0])
                    npz_errs[ic][tomo].append(meas[1])
                    zbins = inference._get_fine_redshift_bins(fr, tracer=self.tomo_to_tracer[tomo])
                    _zvals = (zbins[:-1] + zbins[1:]) / 2
                    zvals[ic][tomo].append(_zvals)
                    assert len(meas[0]) == len(_zvals)
                else:
                    for tracer in self.tomo_to_tracer[tomo]:
                        zbins = fr.get_bins(tracer)
                        _zvals = (zbins[:-1] + zbins[1:]) / 2
                        meas = inference.full_npz_tomo(
                            path_dictionary=path_dictionary,
                            do_phot_correction=do_phot_correction,
                            do_spec_correction=do_spec_correction,
                            scale_cuts=self.scale_cut,
                            data_release=self.stem,
                            hsc_cat=self.hsc_catalog,
                            tomo_bin=tomo,
                            tracer=tracer,
                            which_patches=self.patches,
                            precomp_wdm=wdm_interpolator,
                        )
                        if do_mag:
                            _npz, _npz_err, wdm, Mag, dMag = ct.solve_magnification(
                                meas=meas,
                                tracer=tracer,
                                tomo_bin=tomo,
                                scale_cut=self.scale_cut,
                                zvalues=_zvals,
                                imag_cut=self.imag_cut,
                                return_matrices=True,
                            )
                            meas = (_npz, _npz_err)

                        npzs[ic][tomo].append(meas[0])
                        npz_errs[ic][tomo].append(meas[1])
                        zvals[ic][tomo].append(_zvals)
                        assert len(meas[0]) == len(_zvals)

        data_rows = []
        for tomo in self.tomo_bins:
            z_vals_cross = zvals[0][tomo][0]
            nz_cross = npzs[0][tomo][0]
            nz_cross_err = npz_errs[0][tomo][0]
            for j in range(len(z_vals_cross)):
                data_rows.append(
                    {
                        'tomo_bin': tomo,
                        'tracer': 'Merged',
                        'redshift': z_vals_cross[j],
                        'npz_cross': nz_cross[j],
                        'npz_cross_err': nz_cross_err[j],
                        'npz_bs': None,          # copilot 21/05
                        'npz_bs_err': None,      # copilot 21/05
                        'npz_bs_bp': None,
                        'npz_bs_bp_err': None,
                        'npz_bs_bp_mag': None,
                        'npz_bs_bp_mag_err': None,
                    }
                )

            for i, tracer in enumerate(self.tomo_to_tracer[tomo]):
                z_vals_bs = zvals[1][tomo][i]
                nz_bs = npzs[1][tomo][i]
                nz_bs_err = npz_errs[1][tomo][i]
                z_vals_bp = zvals[2][tomo][i]
                nz_bs_bp = npzs[2][tomo][i]
                nz_bs_bp_err = npz_errs[2][tomo][i]
                z_vals_bp_mag = zvals[3][tomo][i]
                nz_bs_bp_mag = npzs[3][tomo][i]
                nz_bs_bp_mag_err = npz_errs[3][tomo][i]
                assert np.allclose(z_vals_bp, z_vals_bp_mag)

                for j in range(len(z_vals_bp)):
                    data_rows.append(
                        {
                            'tomo_bin': tomo,
                            'tracer': tracer,
                            'redshift': z_vals_bp[j],
                            'npz_cross': None,
                            'npz_cross_err': None,
                            'npz_bs': nz_bs[j],
                            'npz_bs_err': nz_bs_err[j],
                            'npz_bs_bp': nz_bs_bp[j],
                            'npz_bs_bp_err': nz_bs_bp_err[j],
                            'npz_bs_bp_mag': nz_bs_bp_mag[j],
                            'npz_bs_bp_mag_err': nz_bs_bp_mag_err[j],
                        }
                    )
        
        df = pd.DataFrame(data_rows)                        # copilot 21/05
        # Ensure all n(z) columns exist and fill NaN values to avoid errors in _merge_results
        for col in ['npz_cross', 'npz_cross_err', 'npz_bs', 'npz_bs_err', 'npz_bs_bp', 'npz_bs_bp_err', 'npz_bs_bp_mag', 'npz_bs_bp_mag_err']:
            if col not in df.columns:
                df[col] = np.nan
        
        return df

        # DataFrame creation moved before column fill (see above for df creation)

    def _save_dataframe(self, df):
        metadata = self._build_metadata()
        output_file = self.results_root / f'nz_res_{self.scale_cut[0]}_{self.scale_cut[1]}_{self.version}.parquet'
        df.to_parquet(output_file, index=False)
        metadata_file = self.results_root / f'nz_res_metadata_{self.scale_cut[0]}_{self.scale_cut[1]}_{self.version}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(df)
        print(f'Scale cut {self.scale_cut[0]} - {self.scale_cut[1]} Mpc/h results saved to {output_file}')
        data = pd.read_parquet(output_file)
        print(data.columns)
        return data

    def _build_metadata(self):
        path_dict_metadata = {}
        for tomo in self.tomo_bins:
            path_dict_metadata[f'tomo_{tomo}'] = {
                'HSC': str(self.corr_root / 'v12_correction' / 'autos_HSC'),
                'DESI_NGC': str(self.corr_root / self.stem / 'autos_NGC'),
                'DESI_SGC': str(self.corr_root / self.stem / 'autos_SGC'),
                'DESIxHSC': str(self.corr_root / self.stem / 'cross'),
                'stem': self.stem,
            }

        return {
            'scale_cuts': self.scale_cut,
            'patches': self.patches,
            'path_dictionaries': path_dict_metadata,
            'creation_date': datetime.now().isoformat(),
            'description': 'Magnification corrected and raw n(z) measurements for DESI tracers across HSC tomographic bins',
            'tracers_by_tomo': self.tomo_to_tracer,
        }

    def _merge_results(self, data):
        importlib.reload(inference)
        merged = {str(tomo): {} for tomo in self.tomo_bins}
        names = ['npz_cross', 'npz_bs', 'npz_bs_bp', 'npz_bs_bp_mag']

        for name in names:
            for tomo in self.tomo_bins:
                data_tomo = data[data['tomo_bin'] == tomo]
                tracers = list(set(data_tomo['tracer']))
                zv = [data_tomo['redshift'][data_tomo['tracer'] == t].values for t in tracers]
                _npz = [data_tomo[name][data_tomo['tracer'] == t].values for t in tracers]
                _npz_err = [
                    data_tomo[name + '_err'][data_tomo['tracer'] == t].values for t in tracers
                ]
                print(name, tomo, tracers, zv, _npz_err)
                zmt_raw, npmt, npmt_err = inference.merge_results(zv, _npz, _npz_err)
                merged[str(tomo)].update({
                    name + '_z': zmt_raw,
                    name: npmt,
                    name + '_err': npmt_err,
                })
            merged[str(tomo)]['z'] = zmt_raw

        save_dict = {f'{tomo}/{k}': np.array(v) for tomo, d in merged.items() for k, v in d.items()}
        np.savez_compressed(
            self.results_root / f'merged_res_{self.scale_cut[0]}_{self.scale_cut[1]}_{self.version}.npz',
            **save_dict,
        )
        return self.results_root / f'merged_res_{self.scale_cut[0]}_{self.scale_cut[1]}_{self.version}.npz'

    def _save_normalized_nz(self):
        tbl = np.load(self.results_root / f'merged_res_{self.scale_cut[0]}_{self.scale_cut[1]}_{self.version}.npz')
        bounds = self.bounds
        names = ['npz_cross', 'npz_bs', 'npz_bs_bp', 'npz_bs_bp_mag']
        
        # names_bis = {'npz_cross' : 'w/o correction', 'npz_bs' : 'w/ spectro. bias', 'npz_bs_bp' : 'w/ spectro. & photo. bias', 'npz_bs_bp_mag' : 'w/ spectro., photo. & mag. corrections'}
        norm_tbl = {}
        
        # fig, ax = plt.subplots(figsize=(8, 6))

        for tomo in self.tomo_bins:
            for i, na in enumerate(names):
                zn = tbl[f'{tomo}/{na}_z']
                z_mask = (zn >= bounds[tomo][0]) & (zn <= bounds[tomo][1])
                np_z = tbl[f'{tomo}/{na}'][z_mask]
                np_z_err = tbl[f'{tomo}/{na}_err'][z_mask]
                amplitude = np.trapz(np_z, zn[z_mask])
                norm_tbl[f'{tomo}/{na}_z'] = zn[z_mask]
                norm_tbl[f'{tomo}/{na}'] = np_z / amplitude
                norm_tbl[f'{tomo}/{na}_err'] = np_z_err / amplitude
                
            #     plt.errorbar(
            #         zn[z_mask],
            #         norm_tbl[f'{tomo}/{na}'],
            #         norm_tbl[f'{tomo}/{na}_err'],
            #         capsize=3,
            #         label=f'{names_bis[names[i]]}',
            #         linestyle='',
            #         marker='s',
            #         ms=4,
            #     )
                
            # plt.axhline(0, color='k', linestyle='--', linewidth=1)
            # plt.grid(True)
            # plt.xlabel('Redshift')
            # plt.ylabel('Normalized n(z)')

        # identifier according to tracers plotted
        tracers = []
        order = ["BGS_ANY", "LRG", "ELGnotqso", "QSO"]
        for tomo in self.tomo_bins:
            for tracer in self.tomo_to_tracer[tomo]:
                if tracer not in tracers:
                    tracers.append(tracer)
        tracers = sorted(tracers, key=order.index)
        
        # lines = []
        # for tomo in self.tomo_bins:
        #     low = self.params['bins']['hsc'][tomo][0]
        #     high = self.params['bins']['hsc'][tomo][1]
        #     lines.append(f"bin {tomo} : ${low} \leq z < {high}$")
        
        # if self.params['nz']['text'] == 'None':
        #     self.params['nz']['text'] = []
        
        # elif len(self.params['nz']['text']) == 0:
        #     text = "\n".join([
        #                 f"tracers : {', '.join(tracers)}",
        #                 *lines,  # Unpacks the list of lines directly into this position
        #                 f"scale cut : {self.scale_cut[0]} - {self.scale_cut[1]} Mpc/h",
        #                 f"data release : DR{self.stem[2:]}",
        #                 f"HSC catalog : {self.hsc_catalog}"
        #             ])
        #     plt.text(1.05, 0.18, text, transform=plt.gca().transAxes, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        # elif len(self.params['nz']['text']) > 0:
        #     text = (
        #         '\n'.join(self.params['nz']['text'])
        #     )
        #     plt.text(1.05, 0.18, text, transform=plt.gca().transAxes, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # if self.params['nz']['legend'] == 'on':
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        title = self.params['nz']['title']
        if title != "":
            title += "_"
        
        np.savez_compressed(
            self.results_root / f'merged_res_norm_{self.scale_cut[0]}_{self.scale_cut[1]}_{self.version}.npz',
            **norm_tbl,
        )

        # identifier according to the corrections plotted
        identifier = f'mocs={"".join(map(str, self.patches))}_icut={self.imag_cut}'
        # if 'npz_cross' in names:
        #     identifier += 'n'
        # if 'npz_bs' in names:
        #     identifier += 's'
        # if 'npz_bs_bp' in names:
        #     identifier += 'p'
        # if 'npz_bs_bp_mag' in names:
        #     identifier += 'm'
        
        suffix = f'{title}nz_tracers={",".join(tracers)}_bins={"".join(map(str, self.tomo_bins))}_scale_cut={self.scale_cut[0]}-{self.scale_cut[1]}_{identifier}_{self.stem}_{self.hsc_catalog}_{self.version}'
        
        Path('nz_results').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            f'nz_results/{suffix}.npz',
            **norm_tbl,
        )
        
        # Path('figs').mkdir(parents=True, exist_ok=True)
        # plt.savefig(f'figs/{suffix}.png', bbox_inches='tight')
        # plt.show()

    def run(self):
        self._save_bins()
        wdm_interpolator = self._precompute_wdm()
        self._run_merge_estimators()
        df = self._build_data_frame(wdm_interpolator)
        data = self._save_dataframe(df)
        self._merge_results(data)
        self._save_normalized_nz()


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
    import cosmotools_new as ct
    import inference_new as inference

    globals().update({'cf': cf, 'ct': ct, 'inference': inference})

    pipeline = NzPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
