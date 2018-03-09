#!/usr/bin/env python
"""
Functions used to analyse nested sampling runs, perform calculations and
estimate sampling errors.
"""

import copy
import numpy as np
import scipy.special
import nestcheck.data_processing as dp


def rel_posterior_mass(logx, logl):
    """
    Calculate the relative poterior mass for some array of logx values
    given the likelihood, prior and number of dimensions.
    The posterior mass at each logX value is proportional to L(X)X, where L(X)
    is the likelihood.
    The weight is returned normalized so that the integral of the weight with
    respect to logX is 1.

    Parameters
    ----------
    logx: 1d numpy array
        logx values at which to calculate posterior mass.
    logl: 1d numpy array
        logl values corresponding to each logx (same shape as logx).

    Returns
    -------
    w_rel: 1d numpy array
        Relative posterior mass at each input logx value
    """
    logw = logx + logl
    w_rel = np.exp(logw - logw.max())
    w_rel /= np.abs(np.trapz(w_rel, x=logx))
    return w_rel


def run_estimators(ns_run, estimator_list, simulate=False):
    """
    Calculates values of list of estimators for a single nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of estimator classes, each containing class method
        estimator(self, logw, ns_run)
    simulate: bool, optional

    Returns
    -------
    output: 1d numpy array
        Calculation result for each estimator in estimator_list.
    """
    logw = get_logw(ns_run, simulate=simulate)
    output = np.zeros(len(estimator_list))
    for i, est in enumerate(estimator_list):
        output[i] = est(ns_run, logw=logw)
    return output


def array_given_run(ns_run):
    """
    Converts information on samples in a nested sampling run dictionary into a
    numpy array representation. This allows fast addition of more samples and
    recalculation of nlive.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
        Contains keys: 'logl', 'thread_label', 'nlive_array', 'theta'

    Returns
    -------
    samples: numpy array
        Numpy array containing columns
        [logl, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    """
    samples = np.zeros((ns_run['logl'].shape[0], 3 + ns_run['theta'].shape[1]))
    samples[:, 0] = ns_run['logl']
    samples[:, 1] = ns_run['thread_labels']
    # Calculate 'change in nlive' after each step
    samples[:-1, 2] = np.diff(ns_run['nlive_array'])
    samples[-1, 2] = -1  # nlive drops to zero after final point
    samples[:, 3:] = ns_run['theta']
    return samples


def dict_given_run_array(samples, thread_min_max):
    """
    Converts an array of information about samples back into a dictionary.

    Parameters
    ----------
    samples: numpy array
        Numpy array containing columns
        [logl, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    thread_min_max': numpy array, optional
        2d array with a row for each thread containing the likelihoods at which
        it begins and ends.
        Needed to calculate nlive_array (otherwise this is set to None).

    Returns
    -------
    ns_run: dict
        Nested sampling run dictionary corresponding to the samples array.
        Contains keys: 'logl', 'thread_label', 'nlive_array',
        'theta'
        N.B. this does not contain a record of the run's settings.
    """
    ns_run = {'logl': samples[:, 0],
              'thread_labels': samples[:, 1].astype(int),
              'thread_min_max': thread_min_max,
              'theta': samples[:, 3:]}
    assert np.array_equal(samples[:, 1], ns_run['thread_labels']), \
        ('Casting thread labels from samples array to int has changed ' +
         'their values!')
    nlive_0 = (thread_min_max[:, 0] < ns_run['logl'].min()).sum()
    nlive_array = np.zeros(samples.shape[0]) + nlive_0
    nlive_array[1:] += np.cumsum(samples[:-1, 2])
    assert nlive_array.min() > 0, \
        ('nlive contains 0s or negative values!' +
         '\nnlive_array = ' + str(nlive_array) +
         '\nthread_min_max=' + str(thread_min_max))
    assert nlive_array[-1] == 1, 'final point in nlive_array != 1!' \
        '\nnlive_array = ' + str(nlive_array)
    ns_run['nlive_array'] = nlive_array
    return ns_run


def get_run_threads(ns_run):
    """
    Get the individual threads for a nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
        Contains keys: 'logl', 'r', 'logx', 'thread_label', 'nlive_array',
        'theta'

    Returns
    -------
    threads: list of numpy array
        Each thread (list element) is a samples array containing columns
        [logl, r, logx, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    """
    samples = array_given_run(ns_run)
    # assert np.array_equal(
    #     np.asarray(range(ns_run['thread_labels'].min(),
    #                      ns_run['thread_labels'].max() + 1)),
    #     np.unique(ns_run['thread_labels'])), \
    #     str(np.unique(ns_run['thread_labels']))
    unique_threads = np.unique(ns_run['thread_labels'])
    assert ns_run['thread_min_max'].shape[0] == unique_threads.shape[0], \
        ('some threads have no points! ' + str(unique_threads.shape[0]) +
         '!=' + str(ns_run['thread_min_max'].shape[0]))
    threads = []
    for i, th_lab in enumerate(unique_threads):
        thread_array = samples[np.where(samples[:, 1] == th_lab)]
        # delete changes in nlive due to other threads in the run
        thread_array[:, 2] = 0
        thread_array[-1, 2] = -1
        min_max = np.reshape(ns_run['thread_min_max'][i, :], (1, 2))
        assert min_max[0, 1] == thread_array[-1, 0], \
            'thread max logl should equal logl of its final point!'
        threads.append(dict_given_run_array(thread_array, min_max))
    return threads


# Functions for estimating sampling errors
# ----------------------------------------


def bootstrap_resample_run(ns_run, threads=None, ninit_sep=False):
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
        threads seperately. Useful when there are only a few threads which
        start by sampling the whole prior, as errors occur if none of these are
        included in the bootstrap resample.

    Returns
    -------
    ns_run_temp: dict
        Nested sampling run dictionary.
    """
    if threads is None:
        threads = get_run_threads(ns_run)
    n_threads = len(threads)
    if ninit_sep:
        try:
            ninit = ns_run['settings']['ninit']
            inds = np.random.randint(0, ninit, ninit)
            inds = np.append(inds, np.random.randint(ninit, n_threads,
                                                     n_threads - ninit))
        except KeyError:
            print('WARNING: bootrap_resample_run: ninit_sep=True but ' +
                  'ns_run["settings"]["ninit"] does not exist.')
    else:
        inds = np.random.randint(0, n_threads, n_threads)
    threads_temp = [threads[i] for i in inds]
    resampled_run = combine_threads(threads_temp)
    try:
        resampled_run['settings'] = ns_run['settings']
    except KeyError:
        pass
    return resampled_run


def combine_ns_runs(run_list_in, logl_warn_only=True):
    """
    Combine a list of complete ns runs (each without any repeated threads)
    into a single ns run.
    """
    run_list = copy.deepcopy(run_list_in)
    nthread_tot = 0
    for i, _ in enumerate(run_list):
        dp.check_ns_run(run_list[i], logl_warn_only=logl_warn_only)
        run_list[i]['thread_labels'] += nthread_tot
        nthread_tot += run_list[i]['thread_min_max'].shape[0]
    thread_min_max = np.vstack([run['thread_min_max'] for run in run_list])
    # construct samples array from the threads, including an updated nlive
    samples_temp = np.vstack([array_given_run(run) for run in run_list])
    samples_temp = samples_temp[np.argsort(samples_temp[:, 0])]
    # Make combined run
    run = dict_given_run_array(samples_temp, thread_min_max)
    # # Now we need to reorder the thread labels and thread_min_max values so
    # # they go in order
    # thread_labels_new = np.zeros(run['thread_labels'].shape).astype(int)
    # thread_min_max_new = np.zeros(run['thread_min_max'].shape)
    # for i, th_lab in enumerate(np.unique(run['thread_labels'])):
    #     thread_labels_new[np.where(run['thread_labels'] == th_lab)[0]] = i
    #     thread_min_max_new[i, :] = run['thread_min_max'][th_lab, :]
    # run['thread_labels'] = thread_labels_new
    # run['thread_min_max'] = thread_min_max_new
    dp.check_ns_run(run, logl_warn_only=logl_warn_only)
    return run


def combine_threads(threads, assert_birth_point=False):
    """
    Combine list of threads into a single ns run.
    This is different to combining runs as repeated threads are allowed, and as
    some threads can start from loglikelihood contours on which no dead
    point in the run is present.

    Note that if all the thread labels are not unique and in ascending order,
    the output will fail check_ns_run. However provided the thread labels are
    not used it will work ok for calculations based on nlive, logl and theta.
    """
    thread_min_max = np.vstack([td['thread_min_max'] for td in threads])
    assert len(threads) == thread_min_max.shape[0]
    # construct samples array from the threads, including an updated nlive
    samples_temp = np.vstack([array_given_run(thread) for thread in threads])
    samples_temp = samples_temp[np.argsort(samples_temp[:, 0])]
    # update the changes in live points column for threads which start part way
    # through the run. These are only present in dynamic nested sampling.
    logl_starts = thread_min_max[:, 0]
    for logl_start in logl_starts[logl_starts != -np.inf]:
        ind = np.where(samples_temp[:, 0] == logl_start)[0]
        if assert_birth_point:
            assert ind.shape == (1,), \
                'No unique birth point! ' + str(ind.shape)
        if ind.shape == (1,):
            # If the point at which this thread started is present exactly
            # once in this bootstrap replication:
            samples_temp[ind[0], 2] += 1
        elif ind.shape == (0,):
            # If the point with the likelihood at which the thread started
            # is not present in this particular bootstrap replication,
            # approximate it with the point with the nearest likelihood.
            ind_closest = np.argmin(np.abs(samples_temp[:, 0] - logl_start))
            samples_temp[ind_closest, 2] += 1
        else:
            # If the point at which this thread started is present multiple
            # times in this bootstrap replication, select one at random to
            # increment nlive on. This avoids any systematic bias from e.g.
            # always choosing the first point.
            samples_temp[np.random.choice(ind), 2] += 1
    # make run
    ns_run = dict_given_run_array(samples_temp, thread_min_max)
    try:
        dp.check_ns_run_threads(ns_run)
    except AssertionError:
        # If the threads are not valid (e.g. for bootstrap resamples) then
        # set them to None so they can't be accidentally used
        ns_run['thread_labels'] = None
        ns_run['thread_min_max'] = None
    return ns_run


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
    estimator_list: list of estimator classes, each containing class method
        estimator(self, logw, ns_run)
    n_simulate: int
    ninit_sep: bool, optional

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
    estimator_list: list of estimator classes, each containing class method
        estimator(self, logw, ns_run)
    n_simulate: int
    ninit_sep: bool, optional
        For dynamic runs: resample initial threads and dynamically added
        threads seperately. Useful when there are only a few threads which
        start by sampling the whole prior, as errors occur if none of these are
        included in the bootstrap resample.
    flip_skew: bool, optional
        Determine if distribution of bootstrap values should be flipped about
        its mean to better represent our probability distribution on the true
        value - see "Bayesian astrostatistics: a backward look to the future"
        (Loredo, 2012) Figure 2 for an explanation.
        If true, the samples {X} are mapped to (2 mu - {X}), where mu is X's
        mean. This leaves the mean and standard deviation unchanged.

    Returns
    -------
    output: 1d numpy array
        Sampling error on calculation result for each estimator in
        estimator_list.
    """
    ninit_sep = kwargs.pop('ninit_sep', False)
    flip_skew = kwargs.pop('flip_skew', True)
    n_simulate = kwargs.pop('n_simulate')  # No default, must specify
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    threads = get_run_threads(ns_run)
    bs_values = np.zeros((len(estimator_list), n_simulate))
    for i in range(n_simulate):
        ns_run_temp = bootstrap_resample_run(ns_run, threads=threads,
                                             ninit_sep=ninit_sep)
        bs_values[:, i] = run_estimators(ns_run_temp, estimator_list)
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
    estimator_list: list of estimator classes, each containing class method
        estimator(self, logw, ns_run)
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
    expected_estimators = run_estimators(ns_run, estimator_list)
    cdf = ((np.asarray(range(bs_values.shape[1])) + 0.5) /
           bs_values.shape[1])
    ci_output = expected_estimators * 2
    for i, _ in enumerate(ci_output):
        ci_output[i] -= np.interp(1. - cred_int, cdf,
                                  np.sort(bs_values[i, :]))
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
    estimator_list: list of estimator classes, each containing class method
        estimator(self, logw, ns_run)
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
        all_values[:, i] = run_estimators(ns_run, estimator_list,
                                          simulate=True)
    stds = np.zeros(all_values.shape[0])
    for i, _ in enumerate(stds):
        stds[i] = np.std(all_values[i, :], ddof=1)
    return stds


# Helper functions
# ----------------

def get_logw(ns_run, simulate=False):
    """
    Calculates the log posterior weights of the samples (using logarithms to
    avoid overflow errors with very large or small values).

    Uses the trapezium rule such that the weight of point i is
    w_i = l_i (X_{i-1} - X_{i+1}) / 2

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary containing keys:
        logl: 1d numpy array
            Ordered loglikelihood values of points.
        nlive_array: 1d numpy array
            Ordered local number of live points present at each point's
            iso-likelihood contour.
    simulate: bool, optional
        Should log prior volumes logx be simulated from their distribution (if
        false their expected values are used)

    Returns
    -------
    logw: 1d numpy array
        Log posterior masses of points
    """
    try:
        # find logX value for each point
        logx = get_logx(ns_run['nlive_array'], simulate=simulate)
        logw = np.zeros(ns_run['logl'].shape[0])
        # Vectorized trapezium rule: w_i prop to (X_{i-1} - X_{i+1}) / 2
        logw[1:-1] = log_subtract(logx[:-2], logx[2:]) - np.log(2)
        # Assign all prior volume closest to first point X_first to that point:
        # that is from logx=0 to logx=log((X_first + X_second) / 2)
        logw[0] = log_subtract(0, scipy.special.logsumexp([logx[0], logx[1]]) -
                               np.log(2))
        # Assign all prior volume closest to final point X_last to that point:
        # that is from logx=log((X_penultimate + X_last) / 2) to logx=-inf
        logw[-1] = scipy.special.logsumexp([logx[-2], logx[-1]]) - np.log(2)
        # multiply by likelihood (add in log space)
        logw += ns_run['logl']
        return logw
    except IndexError:
        if ns_run['logl'].shape[0] == 1:
            # If there is only one point in the run then assign all prior
            # volume X \in (0, 1) to that point, so the weight is just
            # 1 * logl_0 = logl_0
            return copy.deepcopy(ns_run['logl'])
        else:
            raise


def get_logx(nlive, simulate=False):
    """
    Returns a logx vector showing the expected or simulated logx positions of
    points.

    The shrinkage factor between two points
        t_i = X_{i-1} / X_{i}
    is distributed as the largest of n_i uniform random variables between 1 and
    0, where n_i is the local number of live points.

    We are interested in
        log(t_i) = logX_{i-1} - logX_{i}
    which has expected value -1/n_i.

    Parameters
    ----------
    nlive_array: 1d numpy array
        Ordered local number of live points present at each point's
        isolikelihood contour.
    simulate: bool, optional
        Should log prior volumes logx be simulated from their distribution (if
        False their expected values are used)

    Returns
    -------
    logw: 1d numpy array
        Log posterior masses of points
    """
    assert nlive.min() > 0, 'nlive contains zeros or negative values!' \
        'nlive = ' + str(nlive)
    if simulate:
        logx_steps = np.log(np.random.random(nlive.shape)) / nlive
    else:
        logx_steps = -1 * (nlive ** -1)
    return np.cumsum(logx_steps)


def log_subtract(loga, logb):
    """
    Returns log(a-b) given loga and logb, where loga > logb.
    See https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
    for more details.
    """
    # assert loga >= logb, 'log_subtract: a-b is negative for loga=' + \
    #                       str(loga) + ' and logb=' + str(logb)
    return loga + np.log(1 - np.exp(logb - loga))
