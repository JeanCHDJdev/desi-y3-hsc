import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from functools import partial

import src.statistics.corrfiles as cf


def hsc_dnnz_error(expect, mids, num_samples=1000):
    """
    Compute the error on the n(z) using a log-normal distribution.

    Parameters
    ----------

    expect : array-like
        The expected n(z) values.
    mids : array-like
        The midpoints of the bins.
    num_samples : int
        The number of samples to draw from the log-normal distribution.
    """
    var = (0.15 * expect) ** 2
    mu = np.log(expect**2 / np.sqrt(var + expect**2))
    sig_2 = np.log(var / expect**2 + 1.0)
    samples = multivariate_normal.rvs(mu, np.diag(sig_2), size=num_samples)
    pz = np.exp(samples)
    pz = np.array([el / np.trapz(el, mids) for el in pz])

    return pz, mu, np.diag(sig_2)


def trapz_weights(x):
    """
    Compute the trapezoidal weights for a given set of x and y values.

    Parameters
    ----------
    x : array-like
        The x values.
    y : array-like
        The y values.

    Returns
    -------
    weights : array-like
        The trapezoidal weights.
    """
    dx = np.diff(x)
    weights = np.zeros_like(x)
    weights[0] = dx[0] / 2  # right interval
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2  # left + right intervals
    weights[-1] = dx[-1] / 2  # left interval

    return weights


def combine_error_bars(x, xerr, y, yerr):
    """
    Combine error bars for two samples with z+/-zerr=(x+/-xerr)/sqrt(y+/-yerr)
    Returns zerr.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    xerr = np.asarray(xerr)
    yerr = np.asarray(yerr)
    return np.sqrt(
        (xerr / np.sqrt(y)) ** 2 + (((x / (2 * np.sqrt(y))) / np.abs(y)) * yerr) ** 2
    )


def combine_error_bars_mult(x, xerr, y, yerr):
    """
    Combine error bars for two samples with z+/-zerr=sqrt((x+/-xerr)*(y+/-yerr))
    Returns zerr.
    """
    x = np.asarray(x).copy()
    y = np.asarray(y).copy()
    xerr = np.asarray(xerr).copy()
    yerr = np.asarray(yerr).copy()
    return np.sqrt(1 / (4 * x * y) * (x * yerr + y * xerr) ** 2)
