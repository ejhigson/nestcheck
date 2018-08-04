#!/usr/bin/env python
"""
Sampling error estimates and diagnostic tests for nested sampling runs.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats
import nestcheck.ns_run_utils


# Bootstrap resampling
# --------------------


def bootstrap_resample_run(ns_run, threads=None, ninit_sep=False,
                           random_seed=False):
    """
    Bootstrap resamples threads of nested sampling run, returning a new
    (resampled) nested sampling run.

    Get the individual threads for a nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    threads: None or list of numpy arrays, optional
    ninit_sep: bool
        For dynamic runs: resample initial threads and dynamically added
        threads separately. Useful when there are only a few threads which
        start by sampling the whole prior, as errors occur if none of these are
        included in the bootstrap resample.
    random_seed: None, bool or int, optional
        Set numpy random seed. Default is to use None (so a random seed is
        chosen from the computer's internal state) to ensure reliable results
        when multiprocessing. Can set to an integer or to False to not edit the
        seed.


    Returns
    -------
    ns_run_temp: dict
        Nested sampling run dictionary.
    """
    if random_seed is not False:
        # save the random state so we don't affect other bits of the code
        state = np.random.get_state()
        np.random.seed(random_seed)
    if threads is None:
        threads = nestcheck.ns_run_utils.get_run_threads(ns_run)
    n_threads = len(threads)
    if ninit_sep:
        try:
            ninit = ns_run['settings']['ninit']
            assert np.all(ns_run['thread_min_max'][:ninit, 0] == -np.inf), (
                'ninit_sep assumes the initial threads are labeled '
                '(0,...,ninit-1), so these should start by sampling the whole '
                'prior.')
            inds = np.random.randint(0, ninit, ninit)
            inds = np.append(inds, np.random.randint(ninit, n_threads,
                                                     n_threads - ninit))
        except KeyError:
            warnings.warn((
                'bootstrap_resample_run has kwarg ninit_sep=True but '
                'ns_run["settings"]["ninit"] does not exist. Doing bootstrap '
                'with ninit_sep=False'), UserWarning)
            ninit_sep = False
    if not ninit_sep:
        inds = np.random.randint(0, n_threads, n_threads)
    threads_temp = [threads[i] for i in inds]
    resampled_run = nestcheck.ns_run_utils.combine_threads(threads_temp)
    try:
        resampled_run['settings'] = ns_run['settings']
    except KeyError:
        pass
    if random_seed is not False:
        # if we have used a random seed then return to the original state
        np.random.set_state(state)
    return resampled_run


def run_std_bootstrap(ns_run, estimator_list, **kwargs):
    """
    Uses bootstrap resampling to calculate an estimate of the
    standard deviation of the distribution of sampling errors (the
    uncertainty on the calculation) for a single nested sampling run.

    For more details about bootstrap resampling for estimating sampling
    errors see 'Sampling errors in nested sampling parameter estimation'
    (Higson et al. 2017).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of functions for estimating quantities (such as the
        Bayesian evidence or mean of parameters) from nested sampling runs.
        Example functions can be found in estimators.py. Each should have
        arguments: func(ns_run, logw=None)
    kwargs: dict
        kwargs for run_bootstrap_values

    Returns
    -------
    output: 1d numpy array
        Sampling error on calculation result for each estimator in
        estimator_list.
    """
    bs_values = run_bootstrap_values(ns_run, estimator_list, **kwargs)
    stds = np.zeros(bs_values.shape[0])
    for j, _ in enumerate(stds):
        stds[j] = np.std(bs_values[j, :], ddof=1)
    return stds


def run_bootstrap_values(ns_run, estimator_list, **kwargs):
    """
    Uses bootstrap resampling to calculate an estimate of the
    standard deviation of the distribution of sampling errors (the
    uncertainty on the calculation) for a single nested sampling run.

    For more details about bootstrap resampling for estimating sampling
    errors see 'Sampling errors in nested sampling parameter estimation'
    (Higson et al. 2017).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of functions for estimating quantities (such as the
        Bayesian evidence or mean of parameters) from nested sampling runs.
        Example functions can be found in estimators.py. Each should have
        arguments: func(ns_run, logw=None)
    n_simulate: int
    ninit_sep: bool, optional
        For dynamic runs: resample initial threads and dynamically added
        threads separately. Useful when there are only a few threads which
        start by sampling the whole prior, as errors occur if none of these are
        included in the bootstrap resample.
    flip_skew: bool, optional
        Determine if distribution of bootstrap values should be flipped about
        its mean to better represent our probability distribution on the true
        value - see "Bayesian astrostatistics: a backward look to the future"
        (Loredo, 2012) Figure 2 for an explanation.
        If true, the samples {X} are mapped to (2 mu - {X}), where mu is X's
        mean. This leaves the mean and standard deviation unchanged.
    random_seeds: list, optional
        list of random_seed arguments for bootstrap_resample_run.
        Defaults to range(n_simulate) in order to give reproducible results.

    Returns
    -------
    output: 1d numpy array
        Sampling error on calculation result for each estimator in
        estimator_list.
    """
    ninit_sep = kwargs.pop('ninit_sep', False)
    flip_skew = kwargs.pop('flip_skew', True)
    n_simulate = kwargs.pop('n_simulate')  # No default, must specify
    random_seeds = kwargs.pop('random_seeds', range(n_simulate))
    assert len(random_seeds) == n_simulate
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    threads = nestcheck.ns_run_utils.get_run_threads(ns_run)
    bs_values = np.zeros((len(estimator_list), n_simulate))
    for i, random_seed in enumerate(random_seeds):
        ns_run_temp = bootstrap_resample_run(
            ns_run, threads=threads, ninit_sep=ninit_sep,
            random_seed=random_seed)
        bs_values[:, i] = nestcheck.ns_run_utils.run_estimators(
            ns_run_temp, estimator_list)
        del ns_run_temp
    if flip_skew:
        estimator_means = np.mean(bs_values, axis=1)
        for i, mu in enumerate(estimator_means):
            bs_values[i, :] = (2 * mu) - bs_values[i, :]
    return bs_values


def run_ci_bootstrap(ns_run, estimator_list, **kwargs):
    """
    Uses bootstrap resampling to calculate credible intervals on the
    distribution of sampling errors (the uncertainty on the calculation)
    for a single nested sampling run.

    For more details about bootstrap resampling for estimating sampling
    errors see 'Sampling errors in nested sampling parameter estimation'
    (Higson et al. 2017).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of functions for estimating quantities (such as the
        Bayesian evidence or mean of parameters) from nested sampling runs.
        Example functions can be found in estimators.py. Each should have
        arguments: func(ns_run, logw=None)
    cred_int: float
    n_simulate: int
    ninit_sep: bool, optional

    Returns
    -------
    output: 1d numpy array
        Credible interval on sampling error on calculation result for each
        estimator in estimator_list.
    """
    cred_int = kwargs.pop('cred_int')   # No default, must specify
    bs_values = run_bootstrap_values(ns_run, estimator_list, **kwargs)
    # estimate specific confidence intervals
    # formulae for alpha CI on estimator T = 2 T(x) - G^{-1}(T(x*))
    # where G is the CDF of the bootstrap resamples
    expected_estimators = nestcheck.ns_run_utils.run_estimators(
        ns_run, estimator_list)
    cdf = ((np.asarray(range(bs_values.shape[1])) + 0.5) /
           bs_values.shape[1])
    ci_output = expected_estimators * 2
    for i, _ in enumerate(ci_output):
        ci_output[i] -= np.interp(
            1. - cred_int, cdf, np.sort(bs_values[i, :]))
    return ci_output


def run_std_simulate(ns_run, estimator_list, n_simulate=None):
    """
    Uses the 'simulated weights' method to calculate an estimate of the
    standard deviation of the distribution of sampling errors (the
    uncertainty on the calculation) for a single nested sampling run.

    Note that the simulated weights method is not accurate for parameter
    estimation calculations.

    For more details about the simulated weights method for estimating sampling
    errors see 'Sampling errors in nested sampling parameter estimation'
    (Higson et al. 2017).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of functions for estimating quantities (such as the
        bayesian evidence or mean of parameters) from nested sampling runs.
        Example functions can be found in estimators.py. Each should have
        arguments: func(ns_run, logw=None)
    n_simulate: int

    Returns
    -------
    output: 1d numpy array
        Sampling error on calculation result for each estimator in
        estimator_list.
    """
    assert n_simulate is not None, 'run_std_simulate: must specify n_simulate'
    all_values = np.zeros((len(estimator_list), n_simulate))
    for i in range(n_simulate):
        all_values[:, i] = nestcheck.ns_run_utils.run_estimators(
            ns_run, estimator_list, simulate=True)
    stds = np.zeros(all_values.shape[0])
    for i, _ in enumerate(stds):
        stds[i] = np.std(all_values[i, :], ddof=1)
    return stds


# Implementation error diagnostics
# --------------------------------


def implementation_std(vals_std, vals_std_u, bs_std, bs_std_u, **kwargs):
    """
    Estimates implementation errors from the standard deviations of results
    and of bootstrap values. See "Diagnostic tests for nested sampling
    calculations" (Higson et al. 2018) for more details.

    Simulate errors dirstributions using the fact that (from central limit
    theorem) our uncertainties on vals_std and bs_std are (approximately)
    normally distributed.
    """
    nsim = kwargs.pop('nsim', 1000000)
    random_seed = kwargs.pop('random_seed', 0)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # if the implementation errors are uncorrelated with the
    # sampling errrors: var results = var imp + var sampling
    # so std imp = sqrt(var results - var sampling)
    imp_var = (vals_std ** 2) - (bs_std ** 2)
    imp_std = np.sqrt(np.abs(imp_var)) * np.sign(imp_var)
    ind = np.where(imp_std <= 0)[0]
    imp_std[ind] = 0
    imp_std_u = np.zeros(imp_std.shape)
    imp_frac = imp_std / vals_std
    imp_frac_u = np.zeros(imp_frac.shape)
    # Simulate errors distributions
    for i, _ in enumerate(imp_std_u):
        state = np.random.get_state()
        np.random.seed(random_seed)
        sim_vals_std = np.random.normal(vals_std[i], vals_std_u[i], size=nsim)
        sim_bs_std = np.random.normal(bs_std[i], bs_std_u[i], size=nsim)
        sim_imp_var = (sim_vals_std ** 2) - (sim_bs_std ** 2)
        sim_imp_std = np.sqrt(np.abs(sim_imp_var)) * np.sign(sim_imp_var)
        imp_std_u[i] = np.std(sim_imp_std, ddof=1)
        imp_frac_u[i] = np.std((sim_imp_std / sim_vals_std), ddof=1)
        np.random.set_state(state)
    return imp_std, imp_std_u, imp_frac, imp_frac_u


def run_thread_values(run, estimator_list):
    """Helper function for parallelising thread_values_df."""
    threads = nestcheck.ns_run_utils.get_run_threads(run)
    vals_list = [nestcheck.ns_run_utils.run_estimators(th, estimator_list)
                 for th in threads]
    vals_array = np.stack(vals_list, axis=1)
    return vals_array


def pairwise_distances(dist_list, earth_mover_dist=True, energy_dist=True):
    """
    Applies statistical_distances to each unique pair of distributions in
    dist_list.
    """
    out = []
    index = []
    for i, samp_i in enumerate(dist_list):
        for j, samp_j in enumerate(dist_list):
            if j < i:
                index.append(str((i, j)))
                out.append(statistical_distances(
                    samp_i, samp_j, earth_mover_dist=earth_mover_dist,
                    energy_dist=energy_dist))
    columns = ['ks pvalue', 'ks distance']
    if earth_mover_dist:
        columns.append('earth mover distance')
    if energy_dist:
        columns.append('energy distance')
    ser = pd.DataFrame(out, index=index, columns=columns).unstack()
    ser.index.names = ['calculation type', 'run']
    return ser


def statistical_distances(samples1, samples2, earth_mover_dist=True,
                          energy_dist=True):
    """
    Gets 4 measures of the statistical distance between samples.
    """
    out = []
    temp = scipy.stats.ks_2samp(samples1, samples2)
    out.append(temp.pvalue)
    out.append(temp.statistic)
    if earth_mover_dist:
        out.append(scipy.stats.wasserstein_distance(samples1, samples2))
    if energy_dist:
        out.append(scipy.stats.energy_distance(samples1, samples2))
    return np.asarray(out)
