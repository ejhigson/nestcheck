#!/usr/bin/env python
"""
Functions for processing nested sampling samples.
"""

import numpy as np
import PerfectNS.save_load_utils as slu


def get_polychord_data(file_root, n_runs, **kwargs):
    """
    Load and process polychord chains
    """
    data_dir = kwargs.pop('data_dir', 'processed_data/')
    chains_dir = kwargs.pop('chains_dir', 'chains/')
    load = kwargs.pop('load', True)
    save = kwargs.pop('save', True)
    overwrite_existing = kwargs.pop('overwrite_existing', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    save_name = file_root + '_' + str(n_runs) + 'runs'
    if load:
        # print('get_run_data: ' + save_name)
        try:
            data = slu.pickle_load(data_dir + save_name, print_time=False)
        except OSError:  # FileNotFoundError is a subclass of OSError
            print('File not found - try generating new data')
            load = False
        except EOFError:
            print('EOFError loading file - try generating new data and '
                  'overwriting current file')
            load = False
            overwrite_existing = True
    if not load:
        data = []
        # load and process chains
        for i in range(1, n_runs + 1):
            try:
                temp_root = chains_dir + file_root + '_' + str(i)
                temp_dead = np.loadtxt(temp_root + '_dead.txt')
                data.append(process_chains(temp_dead))
                try:
                    stats_file = file_root + '_' + str(i) + '_stats_pickle'
                    temp_stats = slu.pickle_load(chains_dir + stats_file,
                                                 print_time=False)
                    data[-1]['stats'] = temp_stats
                except OSError:  # FileNotFoundError is a subclass of OSError
                    pass
            except OSError:  # FileNotFoundError is a subclass of OSError
                print('File not found:')
                print(temp_root + '_dead.txt')
                save = False  # only save if every file is found
            except AssertionError as err:
                print('Error processing file:')
                print(temp_root + '_dead.txt')
                print(type(err), str(err.args))
                save = False  # only save if every file is processed ok
        if save:
            print('Processed new chains: saving to ' + save_name)
            slu.pickle_save(data, data_dir + save_name, print_time=False,
                            overwrite_existing=overwrite_existing)
    return data


def process_chains(dead_points):
    """
    tbc
    """
    dead_points = dead_points[np.argsort(dead_points[:, 0])]
    # Treat dead points
    samples = np.zeros(dead_points.shape)
    samples[:, 0] = dead_points[:, 0]
    samples[:, 1] = thread_labels_given_birth_order(dead_points[:, 1])
    samples[:, 2:] = dead_points[:, 2:]
    # perform some checks
    nlive = samples.shape[0] - np.count_nonzero(dead_points[:, 1])
    assert np.unique(samples[-nlive:, 1]).shape[0] == nlive, \
        'The final nlive=' + str(nlive) + ' points of the run are not all ' \
        'from different threads!'
    # make run dictionary
    ns_run_dict = {'logl': samples[:, 0],
                   'thread_labels': samples[:, 1],
                   'theta': samples[:, 2:]}
    nlive_array = np.zeros(samples.shape[0]) + nlive
    for i in range(1, nlive):
        nlive_array[-i] = i
    ns_run_dict['nlive_array'] = nlive_array
    # for compatibility with PerfectNestedSampling functions
    ns_run_dict['settings'] = {'dynamic_goal': None,
                               'nlive_const': nlive}
    ns_run_dict['r'] = np.zeros(samples.shape[0])
    ns_run_dict['logx'] = np.zeros(samples.shape[0])
    thread_min_max = np.zeros((nlive, 2))
    thread_min_max[:, 0] = np.nan
    for i in range(nlive):
        ind = np.where(ns_run_dict['thread_labels'] == (i + 1))[0][-1]
        thread_min_max[i, 1] = ns_run_dict['logl'][ind]
    ns_run_dict['thread_min_max'] = thread_min_max
    return ns_run_dict


def thread_labels_given_birth_order(birth_order):
    """
    Divides a nested sampling run into threads, using info on the contours at
    which points were sampled.
    """
    thread_labels = np.zeros(birth_order.shape)
    for i, start_ind in enumerate(np.where(birth_order == 0)[0]):
        # label the first point in this thread
        thread_labels[start_ind] = i + 1
        # find the point which replaced it
        next_ind = np.where(birth_order == (start_ind + 1))[0]
        while next_ind.shape == (1,):
            # label new point
            thread_labels[next_ind[0]] = i + 1
            # find the point which replaced it
            next_ind = np.where(birth_order == (next_ind[0] + 1))[0]
        assert next_ind.shape == (0,), \
            'ERROR: multiple points with same birth step'
    if thread_labels.min() <= 0:
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
    return thread_labels
