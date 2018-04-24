#!/usr/bin/env python
"""
Create dummy nested sampling run data for testing.
"""

import os
import numpy as np
import nestcheck.ns_run_utils


def write_dummy_polychord_stats(file_root, base_dir):
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

    Returns
    -------
    output: dict
        The expected output of
        nestcheck.process_polychord_stats(file_root, base_dir)
    """
    output = {'file_root': file_root,
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


def get_dummy_run(nthread, nsamples, ndim, seed=False, logl_start=-np.inf):
    """Generate template ns runs for quick testing without loading test
    data."""
    threads = []
    if seed is not False:
        np.random.seed(seed)
    threads = []
    for _ in range(nthread):
        threads.append(get_dummy_thread(nsamples, ndim, seed=False,
                                        logl_start=logl_start))
    # Sort threads in order of starting logl so labels match labels that would
    # have been given processing a dead points array. N.B. this only works when
    # all threads have same start_logl
    threads = sorted(threads, key=lambda th: th['logl'][0])
    for i, _ in enumerate(threads):
        threads[i]['thread_labels'] = np.full(nsamples, i)
    # Use combine_ns_runs rather than combine threads as this relabels the
    # threads according to their order
    return nestcheck.ns_run_utils.combine_threads(threads)


def get_dummy_dynamic_run(nsamples, ndim, seed=False, nthread_init=2,
                          nthread_dyn=3):
    """Get a dummy dynamic run."""
    init = get_dummy_run(nthread_init, nsamples, ndim, seed=seed,
                         logl_start=-np.inf)
    dyn_starts = list(np.random.choice(
        init['logl'], nthread_dyn, replace=False))
    threads = nestcheck.ns_run_utils.get_run_threads(init)
    threads += [get_dummy_thread(nsamples, ndim, seed=False, logl_start=start)
                for start in dyn_starts]
    # make sure the threads have unique labels and combine them
    for i, _ in enumerate(threads):
        threads[i]['thread_labels'] = np.full(nsamples, i)
    run = nestcheck.ns_run_utils.combine_threads(threads)
    # To make sure the thread labelling is same way it would when
    # processing a dead points file, tranform into dead points
    samples = run_dead_points_array(run)
    return nestcheck.data_processing.process_samples_array(samples)


def get_dummy_thread(nsamples, ndim, seed=False, logl_start=-np.inf):
    """Generate a single ns thread for quick testing without loading test
    data."""
    if seed is not False:
        np.random.seed(seed)
    thread = {'logl': np.sort(np.random.random(nsamples)),
              'nlive_array': np.full(nsamples, 1.),
              'theta': np.random.random((nsamples, ndim)),
              'thread_labels': np.zeros(nsamples).astype(int)}
    if logl_start != -np.inf:
        thread['logl'] += logl_start
    thread['thread_min_max'] = np.asarray([[logl_start, thread['logl'][-1]]])
    return thread


def run_dead_points_array(run):
    """
    Makes a PolyChord-stype dead points array corresponding to the input
    run.
    """
    nestcheck.data_processing.check_ns_run(run)
    threads = nestcheck.ns_run_utils.get_run_threads(run)
    dead_arrs = []
    ndim = run['theta'].shape[1]
    for th in threads:
        dead = np.zeros((th['theta'].shape[0], ndim + 2))
        dead[:, :ndim] = th['theta']
        dead[:, ndim] = th['logl']
        dead[1:, ndim + 1] = th['logl'][:-1]
        if th['thread_min_max'][0, 0] == -np.inf:
            dead[0, ndim + 1] = -1e30
        else:
            dead[0, ndim + 1] = th['thread_min_max'][0, 0]
        dead_arrs.append(dead)
    dead = np.vstack(dead_arrs)
    dead = dead[np.argsort(dead[:, ndim]), :]
    return dead
