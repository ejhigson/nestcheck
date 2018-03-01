#!/usr/bin/env python
"""
Functions for estimating quantities from nested sampling runs.
Each estimator function should have arguments

    def __call__(self, ns_run, logw=None, simulate=False):
        ...

This allows logw to be provided if many estimators are being calculated from
the same run so logw is only calculated once. Otherwise logw is calculated from
the run if required.

Each function should also have a latex_name (str) property.
    latex_name: str
        used for plotting results diagrams.

"""

import functools
import numpy as np
import scipy
import copy
import nestcheck.analyse_run as ar


def name_est(func_in, **kwargs):
    """
    Returns a function based on the input function with an added 'latex_name'
    and with any input kwargs frozen.
    """
    if kwargs:
        estimator = copy.deepcopy(func_in)
    else:
        estimator = functools.partial(copy.deepcopy(func_in), **kwargs)
    estimator.latex_name = get_latex_name(func_in, **kwargs)
    return copy.deepcopy(estimator)


def get_latex_name(func, **kwargs):
    """
    Produce a latex formatted name for each function for use in labelling
    results.
    """
    param_ind = kwargs.pop('param_ind', 0)
    probability = kwargs.pop('probability', 0.5)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    ind_str = str(param_ind + 1)
    if func.__name__ == 'count_samples':
        latex_name = r'# samples'
    elif func.__name__ == 'logz':
        latex_name = r'$\mathrm{log} \mathcal{Z}$'
    elif func.__name__ == 'evidence':
        latex_name = r'$\mathcal{Z}$'
    elif func.__name__ == 'r_mean':
        latex_name = r'$\overline{|\theta|}$'
    elif func.__name__ == 'param_mean':
        latex_name = (r'$\overline{\theta_' + ind_str + '}$')
    elif func.__name__ == 'param_squared_mean':
        latex_name = (r'$\overline{\theta^2_' + ind_str + '}$')
    elif func.__name__ == 'param_cred' or func.__name__ == 'r_cred':
        if probability == 0.5:
            latex_name = r'$\mathrm{median}('
        else:
            # format percent without trailing zeros
            percent_str = ('%f' % (probability * 100)).rstrip('0').rstrip('.')
            latex_name = r'$\mathrm{C.I.}_{' + percent_str + r'\%}('
        if func.__name__ == 'param_cred':
            latex_name += r'\theta_' + ind_str + ')$'
        elif func.__name__ == 'r_cred':
            latex_name += r'|\theta|)$'
    else:
        raise AssertionError('get_latex_name not yet set up for ' +
                             func.__name__)
    return latex_name


# Estimators
# ----------

def count_samples(ns_run, logw=None, simulate=False):
    """Number of samples in run."""
    return ns_run['logl'].shape[0]


def logz(ns_run, logw=None, simulate=False):
    """Natural log of Bayesian evidence."""
    if logw is None:
        logw = ar.get_logw(ns_run, simulate=simulate)
    return scipy.special.logsumexp(logw)


def evidence(ns_run, logw=None, simulate=False):
    """Bayesian evidence."""
    if logw is None:
        logw = ar.get_logw(ns_run, simulate=simulate)
    return np.exp(scipy.special.logsumexp(logw))


def param_mean(ns_run, logw=None, simulate=False, param_ind=0):
    """
    Mean of a single parameter (single component of theta).
    """
    if logw is None:
        logw = ar.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())
    return ((np.sum(w_relative * ns_run['theta'][:, param_ind])
             / np.sum(w_relative)))


def param_cred(ns_run, logw=None, simulate=False, probability=0.5,
               param_ind=0):
    """
    One-tailed credible interval on the value of a single parameter (component
    of theta).
    """
    assert 1 > probability > 0, (
        'credible interval prob= ' + str(probability) + ' not in (0, 1)')
    if logw is None:
        logw = ar.get_logw(ns_run, simulate=simulate)
    # get sorted array of parameter values with their posterior weight
    wp = np.zeros((logw.shape[0], 2))
    wp[:, 0] = np.exp(logw - logw.max())
    wp[:, 1] = ns_run['theta'][:, param_ind]
    wp = wp[np.argsort(wp[:, 1], axis=0)]
    # calculate cumulative distribution function (cdf)
    # Adjust by subtracting 0.5 * weight of first point to correct skew
    # - otherwise we need cdf=1 to return the last value but will return
    # the smallest value if cdf<the fractional weight of the first point.
    # This should not much matter as typically points' relative weights
    # will be very small compared to probability or
    # 1-probability.
    cdf = np.cumsum(wp[:, 0]) - (wp[0, 0] / 2)
    cdf /= np.sum(wp[:, 0])
    # linearly interpolate value
    return np.interp(probability, cdf, wp[:, 1])


def param_squared_mean(ns_run, logw=None, simulate=False, param_ind=0):
    """
    Mean of the square of single parameter (second moment of its posterior
    distribution).
    """
    if logw is None:
        logw = ar.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())  # protect against overflow
    w_relative /= np.sum(w_relative)
    return np.sum(w_relative * (ns_run['theta'][:, param_ind] ** 2))


def r_mean(ns_run, logw=None, simulate=False):
    """Mean of |theta| (the radial distance from the centre)."""
    if logw is None:
        logw = ar.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())
    r = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
    return np.sum(w_relative * r) / np.sum(w_relative)


def r_cred(ns_run, logw=None, simulate=False, probability=0.5):
    """One-tailed credible interval on the value of |theta|."""
    assert 1 > probability > 0, (
        'credible interval prob= ' + str(probability) + ' not in (0, 1)')
    if logw is None:
        logw = ar.get_logw(ns_run, simulate=simulate)
    # get sorted array of r values with their posterior weight
    wr = np.zeros((logw.shape[0], 2))
    wr[:, 0] = np.exp(logw - logw.max())
    wr[:, 1] = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
    wr = wr[np.argsort(wr[:, 1], axis=0)]
    # calculate cumulative distribution function (cdf)
    # Adjust by subtracting 0.5 * weight of first point to correct skew
    # - otherwise we need cdf=1 to return the last value but will return
    # the smallest value if cdf<the fractional weight of the first point.
    # This should not much matter as typically points' relative weights
    # will be very small compared to probability or
    # 1-probability.
    cdf = np.cumsum(wr[:, 0]) - (wr[0, 0] / 2)
    cdf /= np.sum(wr[:, 0])
    # calculate cdf
    # linearly interpolate value
    return np.interp(probability, cdf, wr[:, 1])
