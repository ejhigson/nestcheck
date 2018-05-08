#!/usr/bin/env python
"""
Create dummy nested sampling run data for testing.
"""

import numpy as np
import nestcheck.ns_run_utils
import nestcheck.write_polychord_output


def get_dummy_thread(nsamples, **kwargs):
    """
    Generate dummy data for a single nested sampling thread.

    Log-likelihood values of points are generated from a uniform distribution
    in (0, 1), sorted, scaled by logl_range and shifted by logl_start (if it is
    not -np.inf). Theta values of each point are each generated from a uniform
    distribution in (0, 1).

    Parameters
    ----------
    nsamples: int
        Number of samples in thread.
    ndim: int, optional
        Number of dimensions.
    seed: int, optional
        If not False, the seed is set with np.random.seed(seed).
    logl_start: float, optional
        logl at which thread starts.
    logl_range: float, optional
        Scale factor applied to logl values.
    """
    seed = kwargs.pop('seed', False)
    ndim = kwargs.pop('ndim', 2)
    logl_start = kwargs.pop('logl_start', -np.inf)
    logl_range = kwargs.pop('logl_range', 1)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if seed is not False:
        np.random.seed(seed)
    thread = {'logl': np.sort(np.random.random(nsamples)) * logl_range,
              'nlive_array': np.full(nsamples, 1.),
              'theta': np.random.random((nsamples, ndim)),
              'thread_labels': np.zeros(nsamples).astype(int)}
    if logl_start != -np.inf:
        thread['logl'] += logl_start
    thread['thread_min_max'] = np.asarray([[logl_start, thread['logl'][-1]]])
    return thread


def get_dummy_run(nthread, nsamples, **kwargs):
    """
    Generate dummy data for a nested sampling run.

    Log-likelihood values of points are generated from a uniform distribution
    in (0, 1), sorted, scaled by logl_range and shifted by logl_start (if it is
    not -np.inf). Theta values of each point are each generated from a uniform
    distribution in (0, 1).

    Parameters
    ----------
    nthreads: int
        Number of threads in the run.
    nsamples: int
        Number of samples in thread.
    ndim: int, optional
        Number of dimensions.
    seed: int, optional
        If not False, the seed is set with np.random.seed(seed).
    logl_start: float, optional
        logl at which thread starts.
    logl_range: float, optional
        Scale factor applied to logl values.
    """
    seed = kwargs.pop('seed', False)
    ndim = kwargs.pop('ndim', 2)
    logl_start = kwargs.pop('logl_start', -np.inf)
    logl_range = kwargs.pop('logl_range', 1)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    threads = []
    # set seed before generating any threads and do not reset for each thread
    if seed is not False:
        np.random.seed(seed)
    threads = []
    for _ in range(nthread):
        threads.append(get_dummy_thread(
            nsamples, ndim=ndim, seed=False, logl_start=logl_start,
            logl_range=logl_range))
    # Sort threads in order of starting logl so labels match labels that would
    # have been given processing a dead points array. N.B. this only works when
    # all threads have same start_logl
    threads = sorted(threads, key=lambda th: th['logl'][0])
    for i, _ in enumerate(threads):
        threads[i]['thread_labels'] = np.full(nsamples, i)
    # Use combine_ns_runs rather than combine threads as this relabels the
    # threads according to their order
    return nestcheck.ns_run_utils.combine_threads(threads)


def get_dummy_dynamic_run(nsamples, **kwargs):
    """
    Generate dummy data for a dynamic nested sampling run.

    Loglikelihood values of points are generated from a uniform distribution
    in (0, 1), sorted, scaled by logl_range and shifted by logl_start (if it is
    not -np.inf). Theta values of each point are each generated from a uniform
    distribution in (0, 1).

    Parameters
    ----------
    nsamples: int
        Number of samples in thread.
    nthread_init: int
        Number of threads in the inital run (starting at logl=-np.inf).
    nthread_dyn: int
        Number of threads in the inital run (starting at randomly chosen points
        in the initial run).
    ndim: int, optional
        Number of dimensions.
    seed: int, optional
        If not False, the seed is set with np.random.seed(seed).
    logl_start: float, optional
        logl at which thread starts.
    logl_range: float, optional
        Scale factor applied to logl values.
    """
    seed = kwargs.pop('seed', False)
    ndim = kwargs.pop('ndim', 2)
    nthread_init = kwargs.pop('nthread_init', 2)
    nthread_dyn = kwargs.pop('nthread_dyn', 3)
    logl_range = kwargs.pop('logl_range', 1)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    init = get_dummy_run(nthread_init, nsamples, ndim=ndim, seed=seed,
                         logl_start=-np.inf, logl_range=logl_range)
    dyn_starts = list(np.random.choice(
        init['logl'], nthread_dyn, replace=True))
    threads = nestcheck.ns_run_utils.get_run_threads(init)
    # Seed must be False here so it is not set again for each thread
    threads += [get_dummy_thread(
        nsamples, ndim=ndim, seed=False, logl_start=start,
        logl_range=logl_range) for start in dyn_starts]
    # make sure the threads have unique labels and combine them
    for i, _ in enumerate(threads):
        threads[i]['thread_labels'] = np.full(nsamples, i)
    run = nestcheck.ns_run_utils.combine_threads(threads)
    # To make sure the thread labelling is same way it would when
    # processing a dead points file, tranform into dead points
    samples = nestcheck.write_polychord_output.run_dead_birth_array(run)
    return nestcheck.data_processing.process_samples_array(samples)
