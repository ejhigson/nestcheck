#!/usr/bin/env python
"""
Functions for processing nested sampling samples.
"""

import numpy as np
import nestcheck.io_utils as iou


def get_polychord_data(file_root, n_runs, **kwargs):
    """
    Load and process polychord chains
    """
    data_dir = kwargs.pop('data_dir', 'cache/')
    chains_dir = kwargs.pop('chains_dir', 'chains/')
    load = kwargs.pop('load', True)
    save = kwargs.pop('save', True)
    overwrite_existing = kwargs.pop('overwrite_existing', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    save_name = file_root + '_' + str(n_runs) + 'runs'
    if load:
        try:
            return iou.pickle_load(data_dir + save_name)
        except OSError:  # FileNotFoundError is a subclass of OSError
            pass
    data = []
    # load and process chains
    for i in range(1, n_runs + 1):
        try:
            root = chains_dir + file_root + '_' + str(i)
            data.append(process_polychord_run(root))
        except (OSError, AssertionError) as err:
            print(type(err).__name__ + ' processing file ' + root)
            save = False  # only save if every file is processed ok
    if save:
        print('Processed new chains: saving to ' + save_name)
        iou.pickle_save(data, data_dir + save_name, print_time=False,
                        overwrite_existing=overwrite_existing)
    return data


def process_polychord_run(root):
    dead_points = np.loadtxt(root + '_dead.txt')
    info = iou.pickle_load(root + '_info')
    ns_run = process_polychord_dead_points(dead_points)
    for key in ['output', 'settings']:
        assert key not in ns_run
        ns_run[key] = info.pop(key)
    assert not info
    # # for compatibility with PerfectNestedSampling functions
    # ns_run_dict['settings'] = {'dynamic_goal': None,
    #                            'nlive_const': nlive}
    # ns_run_dict['r'] = np.zeros(samples.shape[0])
    # ns_run_dict['logx'] = np.zeros(samples.shape[0])
    # Run some tests
    # --------------
    # For the standard ns case
    if not ns_run['settings']['nlives']:
        standard_nlive_array = np.zeros(ns_run['logl'].shape)
        standard_nlive_array += ns_run['settings']['nlive']
        for i in range(1, ns_run['settings']['nlive']):
            standard_nlive_array[-i] = i
        assert np.array_equal(ns_run['nlive_array'], standard_nlive_array)
    return ns_run


def process_polychord_dead_points(dead_points):
    """
    tbc
    """
    dead_points = dead_points[np.argsort(dead_points[:, 0])]
    # Treat dead points
    ns_run = {}
    ns_run['logl'] = dead_points[:, 0]
    ns_run['birth_step'] = dead_points[:, 1].astype(int)
    assert np.array_equal(ns_run['birth_step'], dead_points[:, 1]), \
        'birth_step values should all be integers!'
    ns_run['theta'] = dead_points[:, 2:]
    ns_run['thread_labels'] = threads_given_birth_order(ns_run['birth_step'])
    # perform some checks
    # nlive = dead_points.shape[0] - np.count_nonzero(dead_points[:, 1])
    # assert np.unique(ns_run['thread_labels'][-nlive:]).shape[0] == nlive, \
    #     'The final nlive=' + str(nlive) + ' points of the run are not all ' \
    #     'from different threads!'
    # get thread min max
    unique_threads = np.unique(ns_run['thread_labels'])
    # Work out nlive_array and thread_min_max logls from thread labels and
    # birth contours
    thread_min_max = np.zeros((unique_threads.shape[0], 2))
    # NB delta_nlive indexes are offset from points' indexes by 1 as we need an
    # element to repesent the initial sampling of live points before any dead
    # points are created.
    # I.E. birth on step 1 corresponds to replacing dead point zero
    delta_nlive = np.zeros(dead_points.shape[0] + 1)
    for i, label in enumerate(unique_threads):
        inds = np.where(ns_run['thread_labels'] == label)[0]
        birth = ns_run['birth_step'][inds[0]]
        death = inds[-1] + 1
        delta_nlive[birth] += 1
        delta_nlive[death] -= 1
        if birth == 0:
            # thread minimum is -inf it starts by sampling from whole prior
            thread_min_max[i, 0] = -np.inf
        else:
            thread_min_max[i, 0] = ns_run['logl'][birth - 1]
        # Max is final logl in thread
        thread_min_max[i, 1] = ns_run['logl'][death - 1]
    ns_run['thread_min_max'] = thread_min_max
    ns_run['nlive_array'] = np.cumsum(delta_nlive)[:-1]
    return ns_run


def threads_given_birth_order(birth_order):
    """
    Divides a nested sampling run into threads, using info on the contours at
    which points were sampled.
    """
    unique, counts = np.unique(birth_order, return_counts=True)
    multi_birth_steps = unique[np.where(counts > 1)]
    # # check that a point is born on every step
    # assert np.array_equal(unique, np.asarray(range(birth_order.max()+1)))
    thread_labels = np.zeros(birth_order.shape)
    thread_num = 0
    for nstep, step in enumerate(multi_birth_steps):
        for i, start_ind in enumerate(np.where(birth_order == step)[0]):
            # unless nstep=0 the first point is assigned to the continuation of
            # the original thread
            if i != 0 or nstep == 0:
                thread_num += 1
                # check point has not already been assigned
                assert thread_labels[start_ind] == 0
                thread_labels[start_ind] = thread_num
                # find the point which replaced it
                next_ind = np.where(birth_order == (start_ind + 1))[0]
                while next_ind.shape != (0,):
                    # check point has not already been assigned
                    assert thread_labels[next_ind[0]] == 0
                    thread_labels[next_ind[0]] = thread_num
                    # find the point which replaced it
                    next_ind = np.where(birth_order == (next_ind[0] + 1))[0]
    assert thread_labels.min() > 0
    n_threads = (np.sum(counts[np.where(counts > 1)]) -
                 (multi_birth_steps.shape[0] - 1))
    assert np.unique(thread_labels).shape[0] == n_threads
    if thread_labels.min() == 0:
        ind = np.where(thread_labels == 0)[0]
        print('WARNING: ' + str(ind.shape[0]) + ' points without thread' +
              ' label:', ind)
        # If there are not many points missing labels then given them random
        # labels. Otherwise throw assertion error.
        if ind.shape[0] <= 10:
            nlive = np.unique(thread_labels).shape[0] - 1
            for ind_i in ind:
                thread_labels[ind_i] = np.random.randint(1, nlive + 1)
        else:
            assert thread_labels.min() > 0, \
                'ERROR: some points did not get a thread label!\n' \
                'Indexes without labels are: ' + \
                str(np.where(thread_labels == 0)) + \
                ' out of ' + str(birth_order.shape) + ' points.\n' \
                'These points were born at steps ' + \
                str(birth_order[np.where(thread_labels == 0)[0]]) + '.\n' \
                'N_threads = ' + str(np.unique(thread_labels).shape[0] - 1)
    assert np.array_equal(thread_labels, thread_labels.astype(int)), \
        'Thread labels should all be ints!'
    return thread_labels.astype(int)
