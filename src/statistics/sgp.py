import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

def gpfit(zval, meas, matern_nu=3/2, constant=1.):
    dz = np.diff(zval)[0]
    y = meas[0] / np.sum(meas[0] * dz)
    y_err = meas[1] / np.sum(meas[0] * dz)

    X = zval.reshape(-1, 1)
    kernel = ConstantKernel(constant_value=constant) * Matern(nu=matern_nu, length_scale=dz/2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(y_err**2))

    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(y_err))
    assert X.shape == (len(y), 1)
    
    gp.fit(X, y)
    y_mean, y_std = gp.predict(X, return_std=True)
    print(f'GP kernel: {gp.kernel_}')
    return y_mean, y_std

def suppression_function():
    pass