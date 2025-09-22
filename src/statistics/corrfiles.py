'''
Utils to grab and fetch files for the cross-correlation analysis
and to set up logging.
'''
import numpy as np
import logging
import time
import psutil

from pathlib import Path

class CorrFileReader():
    '''
    Utility class to grab correctly formatted file names for the cross-correlation
    analysis. Provide a ROOT and the directory has to be with the expected shape.

    Parameters
    ----------
    ROOT : str or Path
        The root directory where the correlation files are stored.
    '''
    def __init__(self, ROOT):
        self.ROOT = Path(ROOT)
        assert self.ROOT.exists(), f"Path {self.ROOT} does not exist"

    def get_file(self, b1, b2, tgt1, tgt2, moc):
        """
        Get the file name for given redshift bins and MOC.
        """
        DIR = self.ROOT / f'{tgt1}x{tgt2}'
        if moc == "Merged":
            return f'{DIR}/{tgt1}x{tgt2}_b1x{b1}_b2x{b2}.npy'
        if moc is None:
            moc = [1, 2, 3, 4]
        if isinstance(moc, list):
            assert all(m in [1, 2, 3, 4] for m in moc), f"MOC values should be in [1, 2, 3, 4], not {moc}"
            path_list = []
            list_files = sorted(list(Path(f'{DIR}', f'{tgt1}x{tgt2}_b1x{b1}_b2x{b2}_moc{m}.npy') for m in moc))
            assert len(list_files) == len(moc), f"Expected {len(moc)} files, found {len(list_files)} for {tgt1}x{tgt2} b1x{b1} b2x{b2}"
            path_list.extend(list_files)
            return sorted(path_list)
        if isinstance(moc, int):
            assert moc in [1, 2, 3, 4], f"MOC should be an integer in [1, 2, 3, 4], not {moc}"
            return f'{DIR}/{tgt1}x{tgt2}_b1x{b1}_b2x{b2}_moc{moc}.npy'
    
    def get_auto_file(self, b, tgt, moc):
        """
        Get the file name for given redshift bins and MOC.
        """
        return self.get_file(b, b, tgt, tgt, moc)

    def get_bins(self, name):
        bins = np.load(f'{self.ROOT}/bins/bins_all.npz')
        return bins[name]
    
    def get_cov_results(self, tgt1, tgt2='HSC'):
        """
        Get the covariance result for given redshift bins and MOC.
        """
        covdir = Path(self.ROOT, f'{tgt1}x{tgt2}', 'cov')
        if not covdir.exists():
            raise FileNotFoundError(f"Covariance directory {covdir} does not exist")
        else:
            files = covdir.glob(f'*.npy')
            return list(files)

def setup_crosscorr_logging(log_file='logs/output', log_level=logging.INFO):
    """
    Set up logging with both console and file output
    
    Args:
        log_file (str): Path to the log file
        log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_file = str(Path(log_file).with_suffix(''))
    log_file += f'_{time.strftime("%Y%m%d_%H%M%S")}.log'
    print(f"Logging to {log_file}")
    log_dir = Path(log_file).parent
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    def log_memory_usage():
        """
        Log current process memory usage
        """
        try:
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1e6
            logger.info(f"Memory Usage: {memory_usage:.2f} MB")
        except Exception as e:
            logger.error(f"Could not log memory usage: {e}")

    logger.memory_usage = log_memory_usage

    return logger

def fetch_desi_files(tgt, randoms=False, weight_type='nonKP', sims=False, sims_version=0, cap=None, version='DR2'):
    if cap is None:
        raise ValueError("cap cannot be None. Please provide a value.")
    assert cap in ['NGC', 'SGC'], f"cap should be either NGC or SGC, not {cap}"
    assert tgt in ['ELG_LOPnotqso', 'ELGnotqso', 'LRG', 'QSO', 'BGS_ANY'], f"Unknown target {tgt}"
    assert weight_type in ['PIP', 'nonKP', 'base'], f"Unknown weight type {weight_type}"

    try:
        if sims:
            sims_root = '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/sims/'
            if randoms:
                return Path(
                    sims_root,
                    'randoms',
                    f'{tgt}_ran_hsc_zcorr.fits'
                )
            return Path(
                sims_root,
                f'v{sims_version}',
                f'desi_targets_sim_{tgt}_v{sims_version}.fits'
                )
        else:
            if version not in ['DR1', 'DR2']:
                raise ValueError(f"Unknown version {version}. Available versions are 'DR1' and 'DR2'.")
            
            if version == 'DR2':
                root = Path('/global/cfs/projectdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/')
            elif version == 'DR1':
                root = Path('/global/cfs/projectdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/')

            if weight_type == 'PIP':
                root = Path(root, 'PIP')
                path = f'{tgt}_{cap}{"_[0-9]*_" if randoms else "_"}clustering{".ran" if randoms else ".dat"}.fits'
            elif weight_type == 'nonKP':
                if version == 'DR2':
                    root = Path(root, 'nonKP')
                path = f'{tgt}_{cap}{"_[0-9]*_" if randoms else "_"}clustering{".ran" if randoms else ".dat"}.fits'
            elif weight_type == 'base':
                root = Path(root)
                path = f'{tgt}_{cap}{"_[0-9]*_" if randoms else "_"}full_HPmapcut{".ran" if randoms else ".dat"}.fits'
            else:
                raise ValueError(f"Unknown weight type {weight_type}")

            files = list(root.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found for path: {path}")
            if len(files) == 1:
                files = files[0]
            return files
    except PermissionError:
        logging.error(f"Permission denied accessing DESI files and randoms = {randoms}")
        raise
    except FileNotFoundError:
        logging.error(f"DESI catalog file not found and randoms = {randoms}") 
        raise

def fetch_hsc_files(randoms=False, include_dud=False, sims=False, sims_version=0):
    try:
        if sims:
            sims_root = '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/sims/'
            if randoms:
                return Path(
                    sims_root,
                    'randoms',
                    f'HSC_randoms_zcorr_v{sims_version}.fits'
                )
            return Path(
                sims_root,
                f'v{sims_version}',
                f'hscy3_sim_v{sims_version}.fits'
                )
        elif randoms:
            # this path root currently does not contain D/UD randoms as they
            root = Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/randoms'
                )
            return list(
                root.glob(f'edge_sc_cr_hscr{"*" if include_dud else "[0-9]"}.fits')
                )
        elif not sims and not randoms:
            return Path(
                '/global/cfs/projectdirs/desi/users/jchdj/desi-y3-hsc/data/hsc/cat/hscy3_cat.fits'
                )
    except PermissionError:
        logging.error(f"Permission denied accessing HSC files and randoms = {randoms}")
        raise
    except FileNotFoundError:
        logging.error(f"HSC catalog file not found and randoms = {randoms}") 
        raise