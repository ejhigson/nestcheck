#!/usr/bin/env python
"""
Functions for processing output files produced by nested sampling software.
Currently compatable with MultiNest and PolyChord output.

Nestcheck's diagnostics require infomation about the steps at which points were
sampled in order to split nested sampling runs into their constituent threads
(single live point runs). See "Sampling Errors In Nested Sampling Parameter
Estimation" (Higson et al. 2017) for more details. *Producing these requires
MultiNest >= v3.11 and PolyChord >= v1.13.*


**nestested sampling run format**

nestcheck stored nested sampling runs in a standard format as python
dictionaries. For a run with nsamp samples, the keys are:

    logl: 1d numpy array
        Log-likelihood values (floats) for each sample.
        Shape is (nsamp,).
    thread_labels: 1d numpy array
        Int representing which thread each point belongs to.
        For some thread label k, the thread's start (birth) log-likelihood
        and end log-likelihood are given by thread_min_max[k, :].
        Shape is (nsamp,).
    thread_min_max: 2d numpy array
        Shape is (# threads, 2).
        Each row k contains min logl (birth contour) and max logl for
        thread with thread label i.
    theta: 2d numpy array
        Parameter values for samples - each row represents a sample.
        Shape is (nsamp, d) where d is number of dimensions.
    nlive_array: 1d numpy array
        Number of live points present between the previous point and
        this point.
    output: dict (optional)
        Dict containing extra information about the run.

Samples are arranged in ascending order of logl.
"""

import os
import re
import warnings
import copy
import numpy as np
import nestcheck.io_utils
import nestcheck.parallel_utils


@nestcheck.io_utils.save_load_result
def batch_process_data(file_roots, **kwargs):
    """
    Process output from many nested sampling runs in parallel with optional
    error handling and caching.

    The result can be cached usin the 'save_name', 'save' and 'load' kwargs (by
    default this is not done). See save_load_result docstring for more details.

    Remaining kwargs passed to parallel_utils.parallel_apply (see its
    docstring for more details).

    Parameters
    ----------
    file_roots: list of strs
        file_roots for the runs to load.
    base_dir: str, optional
        path to directory containing files.
    process_func: function, optional
        function to use to process the data.
    func_kwargs: dict, optional
        additional keyword arguments for process_func.
    errors_to_handle: error or tuple of errors, optional
        which errors to catch when they occur in processing rather than
        raising.
    save_name: str or None, optional
        See nestcheck.io_utils.save_load_result.
    save: bool, optional
        See nestcheck.io_utils.save_load_result.
    load: bool, optional
        See nestcheck.io_utils.save_load_result.
    overwrite_existing: bool, optional
        See nestcheck.io_utils.save_load_result.

    Returns
    -------
    list of ns_run dicts
        List of nested sampling runs in dict format (see the module
        docstring for more details).
    """
    base_dir = kwargs.pop('base_dir', 'chains')
    process_func = kwargs.pop('process_func', process_polychord_run)
    func_kwargs = kwargs.pop('func_kwargs', {})
    func_kwargs['errors_to_handle'] = kwargs.pop('errors_to_handle', ())
    data = nestcheck.parallel_utils.parallel_apply(
        process_error_helper, file_roots, func_args=(base_dir, process_func),
        func_kwargs=func_kwargs, **kwargs)
    # Sort processed runs into the same order as file_roots (as parallel_apply
    # does not preserve order)
    data = sorted(data,
                  key=lambda x: file_roots.index(x['output']['file_root']))
    # Extract error information and print
    errors = {}
    for i, run in enumerate(data):
        if 'error' in run:
            try:
                errors[run['error']].append(i)
            except KeyError:
                errors[run['error']] = [i]
    for error_name, index_list in errors.items():
        message = (error_name + ' processing ' + str(len(index_list)) + ' / '
                   + str(len(file_roots)) + ' files')
        if len(index_list) != len(file_roots):
            message += '. Roots with errors have indexes: ' + str(index_list)
        print(message)
    # Return runs which did not have errors
    return [run for run in data if 'error' not in run]


def process_error_helper(root, base_dir, process_func, errors_to_handle=(),
                         **func_kwargs):
    """
    Wrapper which applies process_func and handles some common errors so one
    bad run does not spoil the whole batch.

    Useful errors to handle include:

    OSError: if you are not sure if all the files exist
    AssertionError: if some of the many assertions fail for known reasons;
    for example is there are occasional problems decomposing runs into threads
    due to limited numerical precision in logls.

    Parameters
    ----------
    root: str
        File root.
    base_dir: str
        Directory containing file.
    process_func: func
        Function for processing file.
    errors_to_handle: error type or tuple of error types
        Errors to catch without throwing an exception.
    func_kwargs: dict
        Kwargs to pass to process_func.

    Returns
    -------
    run: dict
        Nested sampling run dict (see the module docstring for more
        details) or, if an error occured, a dict containing its type
        and the file root.
    """
    try:
        return process_func(root, base_dir, **func_kwargs)
    except errors_to_handle as err:
        run = {'error': type(err).__name__,
               'output': {'file_root': root}}
        return run


def process_polychord_run(file_root, base_dir, process_stats_file=True,
                          **kwargs):
    """
    Loads data from a PolyChord run into the nestcheck dictionary format for
    analysis.

    N.B. producing required output file containing information about the
    iso-likelihood contours within which points were sampled (where they were
    "born") requies PolyChord version v1.13 or later and the setting
    write_dead=True.

    Parameters
    ----------
    file_root: str
        Root for run output file names (PolyChord file_root setting).
    base_dir: str
        Directory containing data (PolyChord base_dir setting).
    process_stats_file: bool, optional
        Should PolyChord's <root>.stats file be processed? Set to False if you
        don't have the <root>.stats file (such as if PolyChord was run with
        write_stats=False).
    kwargs: dict, optional
        Options passed to check_ns_run

    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more details).
    """
    # N.B. PolyChord dead points files also contains remaining live points at
    # termination
    samples = np.loadtxt(os.path.join(base_dir, file_root) + '_dead-birth.txt')
    ns_run = process_samples_array(samples, **kwargs)
    ns_run['output'] = {'base_dir': base_dir, 'file_root': file_root}
    if process_stats_file:
        try:
            ns_run['output'] = process_polychord_stats(file_root, base_dir)
        except (OSError, IOError, ValueError) as err:
            warnings.warn(
                ('process_polychord_stats raised {} processing {}.stats file. '
                 ' Proceeding without stats.').format(
                     type(err).__name__, os.path.join(base_dir, file_root)),
                UserWarning)
    return ns_run


def process_multinest_run(file_root, base_dir, **kwargs):
    """
    Loads data from a MultiNest run into the nestcheck dictionary format for
    analysis.

    N.B. producing required output file containing information about the
    iso-likelihood contours within which points were sampled (where they were
    "born") requies MultiNest version 3.11 or later.

    Parameters
    ----------
    file_root: str
        Root name for output files. When running MultiNest, this is determined
        by the nest_root parameter.
    base_dir: str
        Directory containing output files. When running MultiNest, this is
        determined by the nest_root parameter.
    kwargs: dict, optional
        Passed to check_ns_run (via process_samples_array)


    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more details).
    """
    # Load dead and live points
    dead = np.loadtxt(os.path.join(base_dir, file_root) + '-dead-birth.txt')
    live = np.loadtxt(os.path.join(base_dir, file_root)
                      + '-phys_live-birth.txt')
    # Remove unnessesary final columns
    dead = dead[:, :-2]
    live = live[:, :-1]
    assert dead[:, -2].max() < live[:, -2].min(), (
        'final live points should have greater logls than any dead point!',
        dead, live)
    ns_run = process_samples_array(np.vstack((dead, live)), **kwargs)
    assert np.all(ns_run['thread_min_max'][:, 0] == -np.inf), (
        'As MultiNest does not currently perform dynamic nested sampling, all '
        'threads should start by sampling the whole prior.')
    ns_run['output'] = {}
    ns_run['output']['file_root'] = file_root
    ns_run['output']['base_dir'] = base_dir
    return ns_run


def process_dynesty_run(results):
    """
    Transforms results from a dynesty run into the nestcheck dictionary format
    for analysis. This function has been tested with dynesty v9.2.0.

    Note that the nestcheck point weights and evidence will not be exactly
    the same as the dynesty ones as nestcheck calculates logX volumes more
    precicely (using the trapizium rule).

    Parameters
    ----------
    results: dynesty results object
        N.B. the remaining live points at termination must be included in the
        results (dynesty samplers' run_nested method does this if
        add_live_points=True - its default value).

    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more details).
    """
    samples = np.zeros((results.samples.shape[0],
                        results.samples.shape[1] + 3))
    samples[:, 0] = results.logl
    samples[:, 1] = results.samples_id
    samples[:, 3:] = results.samples
    unique_th, first_inds = np.unique(results.samples_id, return_index=True)
    assert np.array_equal(unique_th, np.asarray(range(unique_th.shape[0])))
    thread_min_max = np.full((unique_th.shape[0], 2), np.nan)
    try:
        # Try processing standard nested sampling results
        assert unique_th.shape[0] == results.nlive
        assert np.array_equal(
            np.unique(results.samples_id[-results.nlive:]),
            np.asarray(range(results.nlive))), (
                'perhaps the final live points are not included?')
        thread_min_max[:, 0] = -np.inf
    except AttributeError:
        # If results has no nlive attribute, it must be dynamic nested sampling
        assert unique_th.shape[0] == sum(results.batch_nlive)
        for th_lab, ind in zip(unique_th, first_inds):
            thread_min_max[th_lab, 0] = (
                results.batch_bounds[results.samples_batch[ind], 0])
    for th_lab in unique_th:
        final_ind = np.where(results.samples_id == th_lab)[0][-1]
        thread_min_max[th_lab, 1] = results.logl[final_ind]
        samples[final_ind, 2] = -1
    assert np.all(~np.isnan(thread_min_max))
    run = nestcheck.ns_run_utils.dict_given_run_array(samples, thread_min_max)
    nestcheck.data_processing.check_ns_run(run)
    return run


def process_polychord_stats(file_root, base_dir):
    """
    Reads a PolyChord <root>.stats output file and returns the information
    contained in a dictionary.

    Parameters
    ----------
    file_root: str
        Root for run output file names (PolyChord file_root setting).
    base_dir: str
        Directory containing data (PolyChord base_dir setting).

    Returns
    -------
    output: dict
        See PolyChord documentation for more details.
    """
    filename = os.path.join(base_dir, file_root) + '.stats'
    output = {'base_dir': base_dir,
              'file_root': file_root}
    with open(filename, 'r') as stats_file:
        lines = stats_file.readlines()
    output['logZ'] = float(lines[8].split()[2])
    output['logZerr'] = float(lines[8].split()[4])
    # Cluster logZs and errors
    output['logZs'] = []
    output['logZerrs'] = []
    for line in lines[14:]:
        if line[:5] != 'log(Z':
            break
        output['logZs'].append(float(
            re.findall(r'=(.*)', line)[0].split()[0]))
        output['logZerrs'].append(float(
            re.findall(r'=(.*)', line)[0].split()[2]))
    # Other output info
    nclust = len(output['logZs'])
    output['ncluster'] = nclust
    output['nposterior'] = int(lines[20 + nclust].split()[1])
    output['nequals'] = int(lines[21 + nclust].split()[1])
    output['ndead'] = int(lines[22 + nclust].split()[1])
    output['nlive'] = int(lines[23 + nclust].split()[1])
    try:
        output['nlike'] = int(lines[24 + nclust].split()[1])
    except ValueError:
        # if nlike has too many digits, PolyChord just writes ***** to .stats
        # file. This causes a ValueError
        output['nlike'] = np.nan
    output['avnlike'] = float(lines[25 + nclust].split()[1])
    output['avnlikeslice'] = float(lines[25 + nclust].split()[3])
    # Means and stds of dimensions (not produced by PolyChord<=1.13)
    if len(lines) > 29 + nclust:
        output['param_means'] = []
        output['param_mean_errs'] = []
        for line in lines[29 + nclust:]:
            output['param_means'].append(float(line.split()[1]))
            output['param_mean_errs'].append(float(line.split()[3]))
    return output


def process_samples_array(samples, **kwargs):
    """
    Convert an array of nested sampling dead and live points of the type
    produced by PolyChord and MultiNest into a nestcheck nested sampling run
    dictionary.

    Parameters
    ----------
    samples: 2d numpy array
        Array of dead points and any remaining live points at termination.
        Has #parameters + 2 columns:
        param_1, param_2, ... , logl, birth_logl
    kwargs: dict, optional
        Options passed to get_birth_inds

    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more
        details). Only contains information in samples (not additional
        optional output key).
    """
    samples = samples[np.argsort(samples[:, -2])]
    ns_run = {}
    ns_run['logl'] = samples[:, -2]
    ns_run['theta'] = samples[:, :-2]
    birth_contours = samples[:, -1]
    # birth_contours, ns_run['theta'] = check_logls_unique(
    #     samples[:, -2], samples[:, -1], samples[:, :-2])
    birth_inds = get_birth_inds(birth_contours, ns_run['logl'], **kwargs)
    ns_run['thread_labels'] = threads_given_birth_contours(birth_inds)
    unique_threads = np.unique(ns_run['thread_labels'])
    assert np.array_equal(unique_threads,
                          np.asarray(range(unique_threads.shape[0])))
    # Work out nlive_array and thread_min_max logls from thread labels and
    # birth contours
    thread_min_max = np.zeros((unique_threads.shape[0], 2))
    # NB delta_nlive indexes are offset from points' indexes by 1 as we need an
    # element to represent the initial sampling of live points before any dead
    # points are created.
    # I.E. birth on step 1 corresponds to replacing dead point zero
    delta_nlive = np.zeros(samples.shape[0] + 1)
    for label in unique_threads:
        thread_inds = np.where(ns_run['thread_labels'] == label)[0]
        # Max is final logl in thread
        thread_min_max[label, 1] = ns_run['logl'][thread_inds[-1]]
        thread_start_birth_ind = birth_inds[thread_inds[0]]
        # delta nlive indexes are +1 from logl indexes to allow for initial
        # nlive (before first dead point)
        delta_nlive[thread_inds[-1] + 1] -= 1
        if thread_start_birth_ind == birth_inds[0]:
            # thread minimum is -inf as it starts by sampling from whole prior
            thread_min_max[label, 0] = -np.inf
            delta_nlive[0] += 1
        else:
            assert thread_start_birth_ind >= 0
            thread_min_max[label, 0] = ns_run['logl'][thread_start_birth_ind]
            delta_nlive[thread_start_birth_ind + 1] += 1
    ns_run['thread_min_max'] = thread_min_max
    ns_run['nlive_array'] = np.cumsum(delta_nlive)[:-1]
    return ns_run


def get_birth_inds(birth_logl_arr, logl_arr, **kwargs):
    """
    Maps the iso-likelihood contours on which points were born to the index of
    the dead point on this contour.

    MultiNest and PolyChord use different values to identify the inital live
    points which were sampled from the whole prior (PolyChord uses -1e+30
    and MultiNest -0.179769313486231571E+309). However in each case the first
    dead point must have been sampled from the whole prior, so for either
    package we can use

    init_birth = birth_logl_arr[0]

    If there are many points with the same logl_arr and dup_assert is False,
    these points are randomly assigned an order (to ensure results are
    consistent, random seeding is used).

    Parameters
    ----------
    logl_arr: 1d numpy array
        logl values of each point.
    birth_logl_arr: 1d numpy array
        logl values of the iso-likelihood contour from within each point was
        sampled (on which it was born).
    dup_assert: bool, optional
        See check_ns_run_logls docstring.
    dup_warn: bool, optional
        See check_ns_run_logls docstring.

    Returns
    -------
    birth_inds: 1d numpy array of ints
        Step at which each element of logl_arr was sampled. Points sampled from
        the whole prior are assigned value -1.
    """
    dup_assert = kwargs.pop('dup_assert', False)
    dup_warn = kwargs.pop('dup_warn', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert logl_arr.ndim == 1, logl_arr.ndim
    assert birth_logl_arr.ndim == 1, birth_logl_arr.ndim
    # Check for duplicate logl values (if specified by dup_assert or dup_warn)
    check_ns_run_logls({'logl': logl_arr}, dup_assert=dup_assert,
                       dup_warn=dup_warn)
    # Random seed so results are consistent if there are duplicate logls
    state = np.random.get_state()  # Save random state before seeding
    np.random.seed(0)
    # Calculate birth inds
    init_birth = birth_logl_arr[0]
    assert np.all(birth_logl_arr <= logl_arr), (
        logl_arr[birth_logl_arr > logl_arr])
    birth_inds = np.full(birth_logl_arr.shape, np.nan)
    birth_inds[birth_logl_arr == init_birth] = -1
    for i, birth_logl in enumerate(birth_logl_arr):
        if not np.isnan(birth_inds[i]):
            # birth ind has already been assigned
            continue
        dup_deaths = np.where(logl_arr == birth_logl)[0]
        if dup_deaths.shape == (1,):
            # death index is unique
            birth_inds[i] = dup_deaths[0]
            continue
        # The remainder of this loop deals with the case that multiple points
        # have the same logl value (=birth_logl). This can occur due to limited
        # precision, or for likelihoods with contant regions. In this case we
        # randomly assign the duplicates birth steps in a manner
        # that provides a valid division into nested sampling runs
        dup_births = np.where(birth_logl_arr == birth_logl)[0]
        assert dup_deaths.shape[0] > 1, dup_deaths
        if np.all(birth_logl_arr[dup_deaths] != birth_logl):
            # If no points both are born and die on this contour, we can just
            # randomly assign an order
            np.random.shuffle(dup_deaths)
            inds_to_use = dup_deaths
        else:
            # If some points are both born and die on the contour, we need to
            # take care that the assigned birth inds do not result in some
            # points dying before they are born
            try:
                inds_to_use = sample_less_than_condition(
                    dup_deaths, dup_births)
            except ValueError:
                raise ValueError((
                    'There is no way to allocate indexes dup_deaths={} such '
                    'that each is less than dup_births={}.').format(
                        dup_deaths, dup_births))
        try:
            # Add our selected inds_to_use values to the birth_inds array
            # Note that dup_deaths (and hence inds to use) may have more
            # members than dup_births, because one of the duplicates may be
            # the final point in a thread. We therefore include only the first
            # dup_births.shape[0] elements
            birth_inds[dup_births] = inds_to_use[:dup_births.shape[0]]
        except ValueError:
            warnings.warn((
                'for logl={}, the number of points born (indexes='
                '{}) is bigger than the number of points dying '
                '(indexes={}). This indicates a problem with your '
                'nested sampling software - it may be caused by '
                'a bug in PolyChord which was fixed in PolyChord '
                'v1.14, so try upgrading. I will try to give an '
                'approximate allocation of threads but this may '
                'fail.').format(
                    birth_logl, dup_births, inds_to_use), UserWarning)
            extra_inds = np.random.choice(
                inds_to_use, size=dup_births.shape[0] - inds_to_use.shape[0])
            inds_to_use = np.concatenate((inds_to_use, extra_inds))
            np.random.shuffle(inds_to_use)
            birth_inds[dup_births] = inds_to_use[:dup_births.shape[0]]
    assert np.all(~np.isnan(birth_inds)), np.isnan(birth_inds).sum()
    np.random.set_state(state)  # Reset random state
    return birth_inds.astype(int)


def sample_less_than_condition(choices_in, condition):
    """Creates a random sample from choices without replacement, subject to the
    condition that each element of the output is greater than the corresponding
    element of the condition array.

    condition should be in ascending order."""
    output = np.zeros(min(condition.shape[0], choices_in.shape[0]))
    choices = copy.deepcopy(choices_in)
    for i, _ in enumerate(output):
        # randomly select one of the choices which meets condition
        avail_inds = np.where(choices < condition[i])[0]
        selected_ind = np.random.choice(avail_inds)
        output[i] = choices[selected_ind]
        # remove the chosen value
        choices = np.delete(choices, selected_ind)
    return output


def threads_given_birth_contours(birth_inds):
    """
    Divides a nested sampling run into threads, using info on the contours at
    which points were sampled. See "Sampling errors in nested sampling
    parameter estimation" (Higson et al. 2017) for more information.

    Parameters
    ----------
    birth_inds: 1d numpy array
        Indexes of the iso-likelihood contours from within which each point was
        sampled ("born").

    Returns
    -------
    thread_labels: 1d numpy array of ints
        labels of the thread each point belongs to.
    """
    unique, counts = np.unique(birth_inds, return_counts=True)
    # First get a list of all the indexes on which threads start and their
    # counts. This is every point initially sampled from the prior, plus any
    # indexes where more than one point is sampled.
    thread_start_inds = np.concatenate((
        unique[:1], unique[1:][counts[1:] > 1]))
    thread_start_counts = np.concatenate((
        counts[:1], counts[1:][counts[1:] > 1] - 1))
    thread_labels = np.full(birth_inds.shape, np.nan)
    thread_num = 0
    for nmulti, multi in enumerate(thread_start_inds):
        for i, start_ind in enumerate(np.where(birth_inds == multi)[0]):
            # unless nmulti=0 the first point born on the contour (i=0) is
            # already assigned to a thread
            if i != 0 or nmulti == 0:
                # check point has not already been assigned
                assert np.isnan(thread_labels[start_ind])
                thread_labels[start_ind] = thread_num
                # find the point which replaced it
                next_ind = np.where(birth_inds == start_ind)[0]
                while next_ind.shape != (0,):
                    # check point has not already been assigned
                    assert np.isnan(thread_labels[next_ind[0]])
                    thread_labels[next_ind[0]] = thread_num
                    # find the point which replaced it
                    next_ind = np.where(birth_inds == next_ind[0])[0]
                thread_num += 1
    assert np.all(~np.isnan(thread_labels)), (
        ('Some points were not given a thread label! Indexes without labels '
         'are {} (out of a total of {} samples).\nlogls on which threads '
         'start are: {} with {} threads starting on each.').format(
             np.where(np.isnan(thread_labels))[0], birth_inds.shape[0],
             thread_start_inds, thread_start_counts))
    assert np.array_equal(thread_labels, thread_labels.astype(int)), (
        'Thread labels should all be ints!')
    thread_labels = thread_labels.astype(int)
    # Check unique thread labels are a sequence from 0 to nthreads-1
    assert np.array_equal(
        np.unique(thread_labels),
        np.asarray(range(sum(thread_start_counts)))), (
            str(np.unique(thread_labels)) + ' is not equal to range('
            + str(sum(thread_start_counts)) + ')')
    return thread_labels


# Functions for checking nestcheck format nested sampling run dictionaries to
# ensure they have the expected properties.


def check_ns_run(run, dup_assert=False, dup_warn=False):
    """
    Checks a nestcheck format nested sampling run dictionary has the expected
    properties (see the module docstring for more details).

    Parameters
    ----------
    run: dict
        nested sampling run to check.
    dup_assert: bool, optional
        See check_ns_run_logls docstring.
    dup_warn: bool, optional
        See check_ns_run_logls docstring.


    Raises
    ------
    AssertionError
        if run does not have expected properties.
    """
    assert isinstance(run, dict)
    check_ns_run_members(run)
    check_ns_run_logls(run, dup_assert=dup_assert, dup_warn=dup_warn)
    check_ns_run_threads(run)


def check_ns_run_members(run):
    """
    Check nested sampling run member keys and values.

    Parameters
    ----------
    run: dict
        nested sampling run to check.

    Raises
    ------
    AssertionError
        if run does not have expected properties.
    """
    run_keys = list(run.keys())
    # Mandatory keys
    for key in ['logl', 'nlive_array', 'theta', 'thread_labels',
                'thread_min_max']:
        assert key in run_keys
        run_keys.remove(key)
    # Optional keys
    for key in ['output']:
        try:
            run_keys.remove(key)
        except ValueError:
            pass
    # Check for unexpected keys
    assert not run_keys, 'Unexpected keys in ns_run: ' + str(run_keys)
    # Check type of mandatory members
    for key in ['logl', 'nlive_array', 'theta', 'thread_labels',
                'thread_min_max']:
        assert isinstance(run[key], np.ndarray), (
            key + ' is type ' + type(run[key]).__name__)
    # check shapes of keys
    assert run['logl'].ndim == 1
    assert run['logl'].shape == run['nlive_array'].shape
    assert run['logl'].shape == run['thread_labels'].shape
    assert run['theta'].ndim == 2
    assert run['logl'].shape[0] == run['theta'].shape[0]


def check_ns_run_logls(run, dup_assert=False, dup_warn=False):
    """
    Check run logls are unique and in the correct order.

    Parameters
    ----------
    run: dict
        nested sampling run to check.
    dup_assert: bool, optional
        Whether to raise and AssertionError if there are duplicate logl values.
    dup_warn: bool, optional
        Whether to give a UserWarning if there are duplicate logl values (only
        used if dup_assert is False).

    Raises
    ------
    AssertionError
        if run does not have expected properties.
    """
    assert np.array_equal(run['logl'], run['logl'][np.argsort(run['logl'])])
    if dup_assert or dup_warn:
        unique_logls, counts = np.unique(run['logl'], return_counts=True)
        repeat_logls = run['logl'].shape[0] - unique_logls.shape[0]
        msg = ('{} duplicate logl values (out of a total of {}). This may be '
               'caused by limited numerical precision in the output files.'
               '\nrepeated logls = {}\ncounts = {}\npositions in list of {}'
               ' unique logls = {}').format(
                   repeat_logls, run['logl'].shape[0],
                   unique_logls[counts != 1], counts[counts != 1],
                   unique_logls.shape[0], np.where(counts != 1)[0])
        if dup_assert:
            assert repeat_logls == 0, msg
        elif dup_warn:
            if repeat_logls != 0:
                warnings.warn(msg, UserWarning)


def check_ns_run_threads(run):
    """
    Check thread labels and thread_min_max have expected properties.

    Parameters
    ----------
    run: dict
        Nested sampling run to check.

    Raises
    ------
    AssertionError
        If run does not have expected properties.
    """
    assert run['thread_labels'].dtype == int
    uniq_th = np.unique(run['thread_labels'])
    assert np.array_equal(
        np.asarray(range(run['thread_min_max'].shape[0])), uniq_th), \
        str(uniq_th)
    # Check thread_min_max
    assert np.any(run['thread_min_max'][:, 0] == -np.inf), (
        'Run should have at least one thread which starts by sampling the ' +
        'whole prior')
    for th_lab in uniq_th:
        inds = np.where(run['thread_labels'] == th_lab)[0]
        th_info = 'thread label={}, first_logl={}, thread_min_max={}'.format(
            th_lab, run['logl'][inds[0]], run['thread_min_max'][th_lab, :])
        assert run['thread_min_max'][th_lab, 0] <= run['logl'][inds[0]], (
            'First point in thread has logl less than thread min logl! ' +
            th_info + ', difference={}'.format(
                run['logl'][inds[0]] - run['thread_min_max'][th_lab, 0]))
        assert run['thread_min_max'][th_lab, 1] == run['logl'][inds[-1]], (
            'Last point in thread logl != thread end logl! ' + th_info)
