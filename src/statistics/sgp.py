import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

def gpfit(zval, meas, matern_nu=3/2, constant=1.):
    dz = np.mean(np.diff(zval))
    y = meas[0] / np.sum(meas[0] * dz)
    y_err = meas[1] / np.sum(meas[0] * dz)

    X = zval.reshape(-1, 1)
    kernel = ConstantKernel(constant_value=constant) * Matern(nu=matern_nu, length_scale=2*dz, length_scale_bounds='fixed')
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(y_err**2))

    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(y_err))
    assert X.shape == (len(y), 1)
    
    gp.fit(X, y)
    y_mean, y_std = gp.predict(X, return_std=True)
    print(f'GP kernel: {gp.kernel_}')
    return y_mean, y_std

def _suppress(x, damping):
    if x <= 0:
        return 0
    elif x < 1:
        1 - (1 - x) ** damping
    else:
        return 1
    
def apply_suppression(zval, gp_n, gp_sigma, SNRthreshold=3, damping=.3):
    dz = np.mean(np.diff(zval))
    gaussian = np.exp(-0.5 * (zval / dz) ** 2)
    x = (1/SNRthreshold) * np.convolve(gaussian, gp_n/gp_sigma, mode='same')
    print(f'Convolved x: {x}')
    return np.array([_suppress(xi, damping) for xi in x])
