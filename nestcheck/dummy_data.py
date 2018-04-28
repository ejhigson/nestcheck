#!/usr/bin/env python
"""
Create dummy nested sampling run data for testing.
"""

import os
import numpy as np
import nestcheck.ns_run_utils


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
    samples = run_dead_points_array(run)
    return nestcheck.data_processing.process_samples_array(samples)


def run_dead_points_array(run):
    """
    Converts input run into an array of the format of a PolyChord
    <root>_dead-birth.txt file.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).

    Returns
    -------
    samples: 2d numpy array
        Array of dead points and any remaining live points at termination.
        Has #parameters + 2 columns:
        param_1, param_2, ... , logl, birth_logl
    """
    nestcheck.data_processing.check_ns_run(run)
    threads = nestcheck.ns_run_utils.get_run_threads(run)
    samp_arrays = []
    ndim = run['theta'].shape[1]
    for th in threads:
        samp_arr = np.zeros((th['theta'].shape[0], ndim + 2))
        samp_arr[:, :ndim] = th['theta']
        samp_arr[:, ndim] = th['logl']
        samp_arr[1:, ndim + 1] = th['logl'][:-1]
        if th['thread_min_max'][0, 0] == -np.inf:
            samp_arr[0, ndim + 1] = -1e30
        else:
            samp_arr[0, ndim + 1] = th['thread_min_max'][0, 0]
        samp_arrays.append(samp_arr)
    samples = np.vstack(samp_arrays)
    samples = samples[np.argsort(samples[:, ndim]), :]
    return samples


def write_dummy_polychord_stats(file_root, base_dir, **kwargs):
    """
    Writes a dummy PolyChord format .stats file for tests functions for
    processing stats files. This is written to:

    base_dir/file_root.stats

    Also returns the data in the file as a dict for comparison.

    Parameters
    ----------
    file_root: str
        Root for run output file names (PolyChord file_root setting).
    base_dir: str
        Directory containing data (PolyChord base_dir setting).
    kwargs: dict
        Output information to write to .stats file - if not specified,
        default values are used.

    Returns
    -------
    output: dict
        The expected output of
        nestcheck.process_polychord_stats(file_root, base_dir)
    """
    default_output = {'file_root': file_root,
                      'base_dir': base_dir,
                      'logZ': -9.99,
                      'logZerr': 1.11,
                      'logZs': [-8.88, -7.77, -6.66, -5.55],
                      'logZerrs': [1.88, 1.77, 1.66, 1.55],
                      'nposterior': 0,
                      'nequals': 0,
                      'ndead': 1234,
                      'nlike': 123456,
                      'nlive': 0,
                      'avnlike': 100.0,
                      'avnlikeslice': 10.0,
                      'param_means': [0.11, 0.01, -0.09],
                      'param_mean_errs': [0.04, 0.03, 0.04]}
    output = {}
    for key, value in default_output.items():
        output[key] = kwargs.pop(key, value)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    output['ncluster'] = len(output['logZs'])
    # Make a PolyChord format .stats file corresponding to output
    file_lines = [
        'Evidence estimates:',
        '===================',
        '  - The evidence Z is a log-normally distributed ...',
        '  - We denote this as log(Z) = mu +/- sigma.',
        '',
        'Global evidence:',
        '----------------',
        '',
        'log(Z)       =  {0} +/-   {1}'.format(
            output['logZ'], output['logZerr']),
        '',
        '',
        'Local evidences:',
        '----------------',
        '']
    for i, (lz, lzerr) in enumerate(zip(output['logZs'], output['logZerrs'])):
        file_lines.append('log(Z_ {0})  =  {1} +/-   {2}'.format(
            str(i + 1).rjust(2), lz, lzerr))
    file_lines += [
        '',
        '',
        'Run-time information:',
        '---------------------',
        '',
        ' ncluster:          0 /       1',
        ' nposterior:        {0}'.format(output['nposterior']),
        ' nequals:           {0}'.format(output['nequals']),
        ' ndead:          {0}'.format(output['ndead']),
        ' nlive:             {0}'.format(output['nlive']),
        ' nlike:         {0}'.format(output['nlike']),
        ' <nlike>:       {0}   (    {1} per slice )'.format(
            output['avnlike'], output['avnlikeslice']),
        '',
        '',
        'Dim No.       Mean        Sigma']
    for i, (mean, meanerr) in enumerate(zip(output['param_means'],
                                            output['param_mean_errs'])):
        file_lines.append('{0}  {1} +/-   {2}'.format(
            str(i + 1).ljust(3), mean, meanerr))
    with open(os.path.join(base_dir, file_root) + '.stats', 'w') as stats_file:
        stats_file.writelines('{}\n'.format(line) for line in file_lines)
    return output
