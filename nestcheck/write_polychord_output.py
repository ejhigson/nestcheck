#!/usr/bin/env python
"""
Functions for writing PolyChord-format output files given a nested sampling run
dictionary stored in the nestcheck format.
"""

import copy
import functools
import os
import numpy as np
import nestcheck.estimators as e
import nestcheck.error_analysis
import nestcheck.ns_run_utils


def write_run_output(run, **kwargs):
    """
    Writes PolyChord output files corresponding to the input nested sampling
    run. The file root is

    root = os.path.join(run['output']['base_dir'], run['output']['file_root'])

    Output files which can be made with this function (see the PolyChord
    documentation for more information about what each contains):

    [root].stats
    [root].txt
    [root]_equal_weights.txt
    [root]_dead-birth.txt
    [root]_dead.txt

    Files produced by PolyChord which are not made by this function:

    [root].resume: for resuming runs part way through (not relevant for a
    completed run).
    [root]_phys_live.txt and [root]phys_live-birth.txt: for checking runtime
    progress (not relevant for a completed run).
    [root].paramnames: for use with getdist (not needed when calling getdist
    from within python).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    write_dead: bool, optional
        Whether or not to write [root]_dead.txt and [root]_dead-birth.txt.
    write_stats: bool, optional
        Whether or not to write [root].stats.
    posteriors: bool, optional
        Whether or not to write [root].txt.
    equals: bool, optional
        Whether or not to write [root]_equal_weights.txt.
    stats_means_errs: bool, optional
        Whether or not to calculate mean values of logZ and each parameter and
        their uncertainties.
    fmt: str, optional
        Formatting for numbers written by np.savetxt. Default value is set to
        make output files look like the ones produced by PolyChord.
    n_simulate: int, optional
        Number of bootstrap replications to use when estimating uncertainty on
        evidence and parameter means.
    """
    write_dead = kwargs.pop('write_dead', True)
    write_stats = kwargs.pop('write_stats', True)
    posteriors = kwargs.pop('posteriors', False)
    equals = kwargs.pop('equals', False)
    stats_means_errs = kwargs.pop('stats_means_errs', True)
    fmt = kwargs.pop('fmt', '% .14E')
    n_simulate = kwargs.pop('n_simulate', 100)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    mandatory_keys = ['file_root', 'base_dir']
    for key in mandatory_keys:
        assert key in run['output'], key + ' not in run["output"]'
    root = os.path.join(run['output']['base_dir'], run['output']['file_root'])
    if write_dead:
        samples = run_dead_birth_array(run)
        np.savetxt(root + '_dead-birth.txt', samples, fmt=fmt)
        np.savetxt(root + '_dead.txt', samples[:, :-1], fmt=fmt)
    if equals or posteriors:
        w_rel = nestcheck.ns_run_utils.get_w_rel(run)
        post_arr = np.zeros((run['theta'].shape[0], run['theta'].shape[1] + 2))
        post_arr[:, 0] = w_rel
        post_arr[:, 1] = -2 * run['logl']
        post_arr[:, 2:] = run['theta']
    if posteriors:
        np.savetxt(root + '.txt', post_arr, fmt=fmt)
        run['output']['nposterior'] = post_arr.shape[0]
    else:
        run['output']['nposterior'] = 0
    if equals:
        inds = np.where(w_rel > np.random.random(w_rel.shape[0]))[0]
        np.savetxt(root + '_equal_weights.txt', post_arr[inds, 1:],
                   fmt=fmt)
        run['output']['nequals'] = inds.shape[0]
    else:
        run['output']['nequals'] = 0
    if write_stats:
        run['output']['ndead'] = run['logl'].shape[0]
        if stats_means_errs:
            # Get logZ and param estimates and errors
            estimators = [e.logz]
            for i in range(run['theta'].shape[1]):
                estimators.append(functools.partial(e.param_mean, param_ind=i))
            values = nestcheck.ns_run_utils.run_estimators(run, estimators)
            stds = nestcheck.error_analysis.run_std_bootstrap(
                run, estimators, n_simulate=n_simulate)
            run['output']['logZ'] = values[0]
            run['output']['logZerr'] = stds[0]
            run['output']['param_means'] = list(values[1:])
            run['output']['param_mean_errs'] = list(stds[1:])
        write_stats_file(run['output'])


def run_dead_birth_array(run, **kwargs):
    """
    Converts input run into an array of the format of a PolyChord
    <root>_dead-birth.txt file. Note that this in fact includes live points
    remaining at termination as well as dead points.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    kwargs: dict, optional
        Options for check_ns_run.

    Returns
    -------
    samples: 2d numpy array
        Array of dead points and any remaining live points at termination.
        Has #parameters + 2 columns:
        param_1, param_2, ... , logl, birth_logl
    """
    nestcheck.data_processing.check_ns_run(run, **kwargs)
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


def write_stats_file(run_output_dict):
    """
    Writes a dummy PolyChord format .stats file for tests functions for
    processing stats files. This is written to:

    base_dir/file_root.stats

    Also returns the data in the file as a dict for comparison.

    Parameters
    ----------
    run_output_dict: dict
        Output information to write to .stats file. Must contain file_root and
        base_dir. If other settings are not specified, default values are used.

    Returns
    -------
    output: dict
        The expected output of
        nestcheck.process_polychord_stats(file_root, base_dir)
    """
    mandatory_keys = ['file_root', 'base_dir']
    for key in mandatory_keys:
        assert key in run_output_dict, key + ' not in run_output_dict'
    default_output = {'logZ': 0.0,
                      'logZerr': 0.0,
                      'logZs': [0.0],
                      'logZerrs': [0.0],
                      'ncluster': 1,
                      'nposterior': 0,
                      'nequals': 0,
                      'ndead': 0,
                      'nlike': 0,
                      'nlive': 0,
                      'avnlike': 0.0,
                      'avnlikeslice': 0.0,
                      'param_means': [0.0, 0.0, 0.0],
                      'param_mean_errs': [0.0, 0.0, 0.0]}
    allowed_keys = set(mandatory_keys) | set(default_output.keys())
    assert set(run_output_dict.keys()).issubset(allowed_keys), (
        'Input dict contains unexpected keys: {}'.format(
            set(run_output_dict.keys()) - allowed_keys))
    output = copy.deepcopy(run_output_dict)
    for key, value in default_output.items():
        if key not in output:
            output[key] = value
    # Make a PolyChord format .stats file corresponding to output
    file_lines = [
        'Evidence estimates:',
        '===================',
        ('  - The evidence Z is a log-normally distributed, with location and '
         'scale parameters mu and sigma.'),
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
    file_path = os.path.join(output['base_dir'],
                             output['file_root'] + '.stats')
    with open(file_path, 'w') as stats_file:
        stats_file.writelines('{}\n'.format(line) for line in file_lines)
    return output
