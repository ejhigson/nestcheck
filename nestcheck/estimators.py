#!/usr/bin/env python
"""
Functions for estimating quantities from nested sampling runs.
Each estimator function should have arguments

    def estimator_func(self, ns_run, logw=None, simulate=False):
        ...

This allows logw to be provided if many estimators are being calculated from
the same run so logw is only calculated once. Otherwise logw is calculated from
the run if required.
"""

import functools
import numpy as np
import scipy
import nestcheck.ns_run_utils


def get_latex_name(func_in, **kwargs):
    """
    Produce a latex formatted name for each function for use in labelling
    results.

    Parameters
    ----------
    func_in: function
    kwargs: dict, optional
        Kwargs for function.

    Returns
    -------
    latex_name: str
        Latex formatted name for the function.
    """
    if isinstance(func_in, functools.partial):
        func = func_in.func
        assert not set(func_in.keywords) & set(kwargs), (
            'kwargs={0} and func_in.keywords={1} contain repeated keys'
            .format(kwargs, func_in.keywords))
        kwargs.update(func_in.keywords)
    else:
        func = func_in
    param_ind = kwargs.pop('param_ind', 0)
    probability = kwargs.pop('probability', 0.5)
    kwargs.pop('handle_indexerror', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    ind_str = r'{\hat{' + str(param_ind + 1) + '}}'
    latex_name_dict = {
        'count_samples': r'samples',
        'logz': r'$\mathrm{log} \mathcal{Z}$',
        'evidence': r'$\mathcal{Z}$',
        'r_mean': r'$\overline{|\theta|}$',
        'param_mean': r'$\overline{\theta_' + ind_str + '}$',
        'param_squared_mean': r'$\overline{\theta^2_' + ind_str + '}$'}
    # Add credible interval names
    if probability == 0.5:
        cred_str = r'$\mathrm{median}('
    else:
        # format percent without trailing zeros
        percent_str = ('%f' % (probability * 100)).rstrip('0').rstrip('.')
        cred_str = r'$\mathrm{C.I.}_{' + percent_str + r'\%}('
    latex_name_dict['param_cred'] = cred_str + r'\theta_' + ind_str + ')$'
    latex_name_dict['r_cred'] = cred_str + r'|\theta|)$'
    try:
        return latex_name_dict[func.__name__]
    except KeyError as err:
        err.args = err.args + ('get_latex_name not yet set up for ' +
                               func.__name__,)
        raise


# Estimators
# ----------

def count_samples(ns_run, logw=None, simulate=False):
    """Number of samples in run."""
    return ns_run['logl'].shape[0]


def logz(ns_run, logw=None, simulate=False):
    """Natural log of Bayesian evidence."""
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    return scipy.special.logsumexp(logw)


def evidence(ns_run, logw=None, simulate=False):
    """Bayesian evidence."""
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    return np.exp(scipy.special.logsumexp(logw))


def param_mean(ns_run, logw=None, simulate=False, param_ind=0,
               handle_indexerror=False):
    """
    Mean of a single parameter (single component of theta).

    handle_indexerror means the function returns nan rather than raising an
    IndexError if param_ind >= ndim. This is useful when applying the same
    list of estimators to data sets of different dimensions.
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())
    try:
        return (np.sum(w_relative * ns_run['theta'][:, param_ind])
                / np.sum(w_relative))
    except IndexError:
        if handle_indexerror:
            return np.nan
        else:
            raise


def param_cred(ns_run, logw=None, simulate=False, probability=0.5,
               param_ind=0):
    """
    One-tailed credible interval on the value of a single parameter (component
    of theta).
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())  # protect against overflow
    return weighted_quantile(probability, ns_run['theta'][:, param_ind],
                             w_relative)


def param_squared_mean(ns_run, logw=None, simulate=False, param_ind=0):
    """
    Mean of the square of single parameter (second moment of its posterior
    distribution).
    """
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())  # protect against overflow
    w_relative /= np.sum(w_relative)
    return np.sum(w_relative * (ns_run['theta'][:, param_ind] ** 2))


def r_mean(ns_run, logw=None, simulate=False):
    """Mean of the radial coordinate (magnitude of theta vector)."""
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())
    r = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
    return np.sum(w_relative * r) / np.sum(w_relative)


def r_cred(ns_run, logw=None, simulate=False, probability=0.5):
    """One-tailed credible interval on the value of the radial coordinate
    (magnitude of theta vector)."""
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(ns_run, simulate=simulate)
    w_relative = np.exp(logw - logw.max())  # protect against overflow
    r = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
    return weighted_quantile(probability, r, w_relative)


# Helper functions
# ----------------


def weighted_quantile(probability, values, weights):
    """
    Get quantile estimate for input probability given weighted samples using
    linear interpolation.

    Parameters
    ----------
    probability: float
        Quantile to estimate - must be in open interval (0, 1).
    values: 1d numpy array
        Sample values.
    weights: 1d numpy array
        Corresponding sample weights (same shape as values).

    Returns
    -------
    quantile: float
    """
    assert 1 > probability > 0, (
        'credible interval prob= ' + str(probability) + ' not in (0, 1)')
    assert values.shape == weights.shape
    assert values.ndim == 1
    assert weights.ndim == 1
    sorted_inds = np.argsort(values)
    quantiles = np.cumsum(weights[sorted_inds]) - (0.5 * weights[sorted_inds])
    quantiles /= np.sum(weights)
    return np.interp(probability, quantiles, values[sorted_inds])
