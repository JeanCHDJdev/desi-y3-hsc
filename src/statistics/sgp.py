import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

def gpfit(zval, meas, matern_nu=3/2, constant=1., length_scale=None, return_kernel=False):
    dz = np.mean(np.diff(zval))
    y = meas[0]
    y_err = meas[1]

    X = zval.reshape(-1, 1)
    if length_scale is not None:
        kernel = Matern(nu=matern_nu, length_scale=length_scale, length_scale_bounds='fixed')
    else:
        kernel = Matern(nu=matern_nu, length_scale=dz, length_scale_bounds='fixed')
    if constant is not None:
        kernel *= ConstantKernel(constant, constant_value_bounds='fixed')
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(y_err**2))

    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(y_err))
    assert X.shape == (len(y), 1)
    
    gp.fit(X, y)
    y_mean, y_std = gp.predict(X, return_std=True)
    
    if return_kernel:
        print(f'GP kernel: {gp.kernel_}')
        return y_mean, y_std, gp
    
    return y_mean, y_std

def _suppress(x, damping):
    if x <= 0:
        return 0
    elif x < 1:
        return 1 - (1 - x) ** damping
    else:
        return 1
    
def suppression(zval, gp_n, gp_sigma, SNRthreshold=3, damping=.3):
    dz = np.mean(np.diff(zval))
    kernel_size = int(2 * np.ceil(1 / dz))

    if kernel_size % 2 == 0:
        kernel_size -= 1 

    k_range = np.arange(-kernel_size // 2, kernel_size // 2 + 1)

    gaussian = np.exp(-0.5 * (((k_range+1) * dz) / (0.1)) ** 2)
    gaussian /= np.sum(gaussian)

    x = (1/SNRthreshold) * np.convolve(gp_n/gp_sigma, gaussian, mode='same')
    suppression = np.array([_suppress(xi, damping) for xi in x])
    return suppression

def draw_from_gp(gp_n, gp_sigma, n_draws=100, seed=None):
    """
    Draw samples from the Gaussian Process defined by gp_n and gp_sigma.
    This function assumes that gp_n and gp_sigma are the mean and standard deviation
    of the GP at the given zval points.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=gp_n, scale=gp_sigma, size=(n_draws, len(gp_n))).T

def get_enveloppe(draws, sigma_level=1):
    sigma_to_percent = {1: 68.0, 2: 95.0, 3: 99.7}
    if sigma_level not in sigma_to_percent:
        raise ValueError("Only sigma levels 1, 2, or 3 are supported.")
    p = sigma_to_percent[sigma_level]
    pc = [(100 - p) / 2, 100 - ((100 - p) / 2)]
    mean = np.mean(draws, axis=1)
    lower = np.percentile(draws, pc[0], axis=1)
    upper = np.percentile(draws, pc[1], axis=1)
    
    return mean, lower, upper

def suppress_nz(zval, gp_n, gp_sigma, SNRthreshold=3, damping=.3, n_draws=500):
    """
    Suppress the noise in the Gaussian Process based on the SNR threshold.
    """
    suppression_function = suppression(zval, gp_n, gp_sigma, SNRthreshold=SNRthreshold, damping=damping)
    draws = draw_from_gp(gp_n, gp_sigma, n_draws=n_draws)
    suppressed_draws = draws * suppression_function[:, np.newaxis]
    mean, lower, upper = get_enveloppe(suppressed_draws)
    renormalization = np.trapz(mean, zval)
    if renormalization == 0:
        raise ValueError("Renormalization factor is zero, cannot proceed with suppression.")
    return mean/renormalization , (upper - lower)/(2*renormalization)
