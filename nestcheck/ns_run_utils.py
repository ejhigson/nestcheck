#!/usr/bin/env python
"""
Functions for performing basic operations on nested sampling runs; such as
working out point weights and splitting and combining runs.

Nested sampling runs are stored in a standard format as python dictionaries
(see data_processing module docstring for more details).
"""
import copy
import warnings
import numpy as np
import scipy.special
import nestcheck.data_processing as dp


def run_estimators(ns_run, estimator_list, simulate=False):
    """
    Calculates values of list of quantities (such as the Bayesian evidence or
    mean of parameters) for a single nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    estimator_list: list of functions for estimating quantities from nested
        sampling runs. Example functions can be found in estimators.py. Each
        should have arguments: func(ns_run, logw=None).
    simulate: bool, optional
        See get_logw docstring.

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
        Nested sampling run dict (see data_processing module docstring for more
        details).

    Returns
    -------
    samples: 2d numpy array
        Array containing columns
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
    Converts an array of information about samples back into a nested sampling
    run dictionary (see data_processing module docstring for more details).

    N.B. the output dict only keys: 'logl', 'thread_label', 'nlive_array',
    'theta'. Any other keys giving additional information about the run
    output cannot be reproduced from the function arguments and are
    therefore ommitted.

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
        Nested sampling run dict (see data_processing module docstring for more
        details).
    """
    ns_run = {'logl': samples[:, 0],
              'thread_labels': samples[:, 1],
              'thread_min_max': thread_min_max,
              'theta': samples[:, 3:]}
    if np.all(~np.isnan(ns_run['thread_labels'])):
        ns_run['thread_labels'] = ns_run['thread_labels'].astype(int)
        assert np.array_equal(samples[:, 1], ns_run['thread_labels']), ((
            'Casting thread labels from samples array to int has changed '
            'their values!\nsamples[:, 1]={}\nthread_labels={}').format(
                samples[:, 1], ns_run['thread_labels']))
    nlive_0 = (thread_min_max[:, 0] <= ns_run['logl'].min()).sum()
    assert nlive_0 > 0, 'nlive_0={}'.format(nlive_0)
    nlive_array = np.zeros(samples.shape[0]) + nlive_0
    nlive_array[1:] += np.cumsum(samples[:-1, 2])
    # Check if there are multiple threads starting on the first logl point
    dup_th_starts = (thread_min_max[:, 0] == ns_run['logl'].min()).sum()
    if dup_th_starts > 1:
        # In this case we approximate the true nlive (which we dont really
        # know) by making sure the array's final point is 1 and setting all
        # points with logl = logl.min() to have the same nlive
        nlive_array += (1 - nlive_array[-1])
        n_logl_min = (ns_run['logl'] == ns_run['logl'].min()).sum()
        nlive_array[:n_logl_min] = nlive_0
        warnings.warn((
            'duplicate starting logls: {} threads start at logl.min()={}, '
            'and {} points have logl=logl.min(). nlive_array may only be '
            'approximately correct.').format(
                dup_th_starts, ns_run['logl'].min(), n_logl_min), UserWarning)
    assert nlive_array.min() > 0, ((
        'nlive contains 0s or negative values. nlive_0={}'
        '\nnlive_array = {}\nthread_min_max={}').format(
            nlive_0, nlive_array, thread_min_max))
    assert nlive_array[-1] == 1, (
        'final point in nlive_array != 1.\nnlive_array = ' + str(nlive_array))
    ns_run['nlive_array'] = nlive_array
    return ns_run


def get_run_threads(ns_run):
    """
    Get the individual threads from a nested sampling run.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).

    Returns
    -------
    threads: list of numpy array
        Each thread (list element) is a samples array containing columns
        [logl, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    """
    samples = array_given_run(ns_run)
    unique_threads = np.unique(ns_run['thread_labels'])
    assert ns_run['thread_min_max'].shape[0] == unique_threads.shape[0], (
        'some threads have no points! {0} != {1}'.format(
            unique_threads.shape[0], ns_run['thread_min_max'].shape[0]))
    threads = []
    for i, th_lab in enumerate(unique_threads):
        thread_array = samples[np.where(samples[:, 1] == th_lab)]
        # delete changes in nlive due to other threads in the run
        thread_array[:, 2] = 0
        thread_array[-1, 2] = -1
        min_max = np.reshape(ns_run['thread_min_max'][i, :], (1, 2))
        assert min_max[0, 1] == thread_array[-1, 0], (
            'thread max logl should equal logl of its final point!')
        threads.append(dict_given_run_array(thread_array, min_max))
    return threads


def combine_ns_runs(run_list_in, **kwargs):
    """
    Combine a list of complete nested sampling run dictionaries into a single
    ns run.

    Input runs must contain any repeated threads.

    Parameters
    ----------
    run_list_in: list of dicts
        List of nested sampling runs in dict format (see data_processing module
        docstring for more details).
    kwargs: dict, optional
        Options for check_ns_run.

    Returns
    -------
    run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    """
    run_list = copy.deepcopy(run_list_in)
    if len(run_list) == 1:
        run = run_list[0]
    else:
        nthread_tot = 0
        for i, _ in enumerate(run_list):
            dp.check_ns_run(run_list[i], **kwargs)
            run_list[i]['thread_labels'] += nthread_tot
            nthread_tot += run_list[i]['thread_min_max'].shape[0]
        thread_min_max = np.vstack([run['thread_min_max'] for run in run_list])
        # construct samples array from the threads, including an updated nlive
        samples_temp = np.vstack([array_given_run(run) for run in run_list])
        samples_temp = samples_temp[np.argsort(samples_temp[:, 0])]
        # Make combined run
        run = dict_given_run_array(samples_temp, thread_min_max)
        # Combine only the additive properties stored in run['output']
        run['output'] = {}
        for key in ['nlike', 'ndead']:
            try:
                run['output'][key] = sum([temp['output'][key] for temp in
                                          run_list_in])
            except KeyError:
                pass
    dp.check_ns_run(run, **kwargs)
    return run


def combine_threads(threads, assert_birth_point=False):
    """
    Combine list of threads into a single ns run.
    This is different to combining runs as repeated threads are allowed, and as
    some threads can start from log-likelihood contours on which no dead
    point in the run is present.

    Note that if all the thread labels are not unique and in ascending order,
    the output will fail check_ns_run. However provided the thread labels are
    not used it will work ok for calculations based on nlive, logl and theta.

    Parameters
    ----------
    threads: list of dicts
        List of nested sampling run dicts, each representing a single thread.
    assert_birth_point: bool, optional
        Whether or not to assert there is exactly one point present in the run
        with the log-likelihood at which each point was born. This is not true
        for bootstrap resamples of runs, where birth points may be repeated or
        not present at all.

    Returns
    -------
    run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    """
    thread_min_max = np.vstack([td['thread_min_max'] for td in threads])
    assert len(threads) == thread_min_max.shape[0]
    # construct samples array from the threads, including an updated nlive
    samples_temp = np.vstack([array_given_run(thread) for thread in threads])
    samples_temp = samples_temp[np.argsort(samples_temp[:, 0])]
    # update the changes in live points column for threads which start part way
    # through the run. These are only present in dynamic nested sampling.
    logl_starts = thread_min_max[:, 0]
    state = np.random.get_state()  # save random state
    np.random.seed(0)  # seed to make sure any random assignment is repoducable
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
    np.random.set_state(state)
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


def get_logw(ns_run, simulate=False):
    """
    Calculates the log posterior weights of the samples (using logarithms to
    avoid overflow errors with very large or small values).

    Uses the trapezium rule such that the weight of point i is

    .. math:: w_i = \\mathcal{L}_i (X_{i-1} - X_{i+1}) / 2

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    simulate: bool, optional
        Should log prior volumes logx be simulated from their distribution (if
        false their expected values are used).

    Returns
    -------
    logw: 1d numpy array
        Log posterior masses of points.
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


def get_w_rel(ns_run, simulate=False):
    """
    Get the relative posterior weights of the samples, normalised so the
    maximum sample weight is 1. This is calculated from get_logw with
    protection against numerical overflows.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    simulate: bool, optional
        See the get_logw docstring for more details.

    Returns
    -------
    w_rel: 1d numpy array
        Relative posterior masses of points.
    """
    logw = get_logw(ns_run, simulate=simulate)
    return np.exp(logw - logw.max())


def get_logx(nlive, simulate=False):
    """
    Returns a logx vector showing the expected or simulated logx positions of
    points.

    The shrinkage factor between two points

    .. math:: t_i = X_{i-1} / X_{i}

    is distributed as the largest of :math:`n_i` uniform random variables
    between 1 and 0, where :math:`n_i` is the local number of live points.

    We are interested in

    .. math:: \\log(t_i) = \\log X_{i-1} - \\logX_{i}

    which has expected value :math:`-1/n_i`.

    Parameters
    ----------
    nlive_array: 1d numpy array
        Ordered local number of live points present at each point's
        iso-likelihood contour.
    simulate: bool, optional
        Should log prior volumes logx be simulated from their distribution (if
        False their expected values are used).

    Returns
    -------
    logx: 1d numpy array
        log X values for points.
    """
    assert nlive.min() > 0, (
        'nlive contains zeros or negative values! nlive = ' + str(nlive))
    if simulate:
        logx_steps = np.log(np.random.random(nlive.shape)) / nlive
    else:
        logx_steps = -1 * (nlive.astype(float) ** -1)
    return np.cumsum(logx_steps)


def log_subtract(loga, logb):
    """
    Numerically stable way to calculate log(a-b) given loga and logb (where
    loga > logb) while avoiding overflow errors.
    See https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
    for more details.

    Parameters
    ----------
    loga: float
    logb: float
        Must be less than loga.

    Returns
    -------
    log(a - b): float
    """
    return loga + np.log(1 - np.exp(logb - loga))
