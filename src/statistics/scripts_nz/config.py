import os
from pathlib import Path
import yaml
import numpy as np


class CorrelationConfig:
    """Load and expose correlation analysis configuration as an object."""

    _instance = None

    def __init__(self, params, config_path="params.yml"):
        self.params = params
        self.config_path = Path(config_path)
        self.username = params.get("username", os.getenv("USER", "unknown"))
        self.hsc_catalog = params["data"]["hsc_catalog"]
        self.desi_data_release = params["data"]["desi_data_release"]
        self.overwrite_files = params["advanced"].get("overwrite_files", False)
        self.imag_cut = params["nz"]["imag_cut"]
        self.bins_hsc_raw = params["bins"]["hsc"]
        self.bins_hsc = self._make_hsc_bins(params["bins"]["hsc"])
        self.hsc_keys = list(sorted(params["bins"]["hsc"].keys()))
        self.bins_bgs = self._make_tracer_bins(params["bins"]["desi"]["BGS_ANY"], "BGS_ANY")
        self.bins_lrg = self._make_tracer_bins(params["bins"]["desi"]["LRG"], "LRG")
        self.bins_elg = self._make_tracer_bins(params["bins"]["desi"]["ELGnotqso"], "ELGnotqso")
        self.bins_qso = self._make_tracer_bins(params["bins"]["desi"]["QSO"], "QSO")
        self.qso_spectro_bias = params['nz']['qso_spectro_bias']
        self.corr_title_corr = params['advanced']['corr_title']
        self.corr_title_nz = params['nz']['corr_title']
        self.calibration_cut = params['advanced']['calibration_cut']
        CorrelationConfig._instance = self

    @classmethod
    def load(cls, config_path="params.yml"):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            params = yaml.safe_load(f)
        return cls(params, config_path=str(config_path))

    @classmethod
    def default(cls):
        return cls.load("params.yml")
    
    @classmethod
    def get_current(cls):
        if cls._instance is None:
            raise RuntimeError(
                'No configuration loaded. Call CorrelationConfig.load(config_path) first.'
            )
        return cls._instance

    @staticmethod
    def _make_tracer_bins(bounds, tracer):
        if len(bounds) != 3:
            raise ValueError(f"Tracer {tracer} bin definition must be [start, stop, step]")
        start, stop, step = bounds
        assert 0 <= start < stop, (
            f'Bins for tracer {tracer} have invalid start and stop values : {start}, {stop}'
        )
        assert step > 0, (
            f'Step value for tracer {tracer} invalid : should be > 0.'
        )
        return np.arange(start, stop + step/2, step, dtype=np.float64)

    @staticmethod
    def _make_hsc_bins(bins_dict):
        keys = sorted(bins_dict.keys())
        if len(keys) == 0:
            return np.array([], dtype=np.float64)
        if len(keys) > 0:
            assert (
                len(keys) == keys[-1] - keys[0] + 1
            ), 'HSC bins should be consecutive bins'
            for key in keys:
                if key != keys[-1]:
                    assert bins_dict[key][1] == bins_dict[key + 1][0], (
                        f'HSC bin {key} and bin {key + 1} are not consecutive'
                    )
                assert len(bins_dict[key]) == 2, (
                    f'HSC bin {key} should have exactly 2 edges : [start, stop]'
                )
                assert 0 <= bins_dict[key][0] < bins_dict[key][1], (
                    f'HSC bin {key} has invalid edges: {bins_dict[key]}'
                )
        edges = [bins_dict[key][0] for key in keys]
        edges.append(bins_dict[keys[-1]][1])
        return np.array(edges, dtype=np.float64)

    def get_bin_edges(self, tracer):
        tracer_bins = {
            "BGS_ANY": self.bins_bgs,
            "LRG": self.bins_lrg,
            "ELGnotqso": self.bins_elg,
            "QSO": self.bins_qso,
            "HSC": self.bins_hsc,
        }
        return tracer_bins[tracer]
