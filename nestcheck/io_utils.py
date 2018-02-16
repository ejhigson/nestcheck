#!/usr/bin/env python
"""
Contains helper functions for saving, loading and input/output.
"""


import time
import pickle
import errno
from functools import wraps
import os.path


def timing_decorator(func):
    """
    Prints the time a function takes to execute.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper for printing execution time.
        """
        print_time = kwargs.pop('print_time', False)
        start_time = time.time()
        result = func(*args, **kwargs)
        if print_time:
            end_time = time.time()
            print(func.__name__ + ' took %.3f seconds' %
                  (end_time - start_time))
        return result
    return wrapper


def data_save_name(settings, n_repeats, extra_root=None, include_dg=True):
    """
    Make a standard save name format for a given settings configoration.
    """
    save_name = ''
    if extra_root is not None:
        save_name += str(extra_root) + '_'
    if include_dg:
        save_name += 'dg' + str(settings.dynamic_goal) + '_'
    save_name += str(settings.n_dim) + 'd'
    # add likelihood and prior info
    save_name += '_' + type(settings.likelihood).__name__
    if type(settings.likelihood).__name__ == 'exp_power':
        save_name += '_' + str(settings.likelihood.power)
    save_name += '_' + str(settings.likelihood.likelihood_scale)
    save_name += '_' + type(settings.prior).__name__
    save_name += '_' + str(settings.prior.prior_scale)
    save_name += '_' + str(settings.termination_fraction) + 'term'
    save_name += '_' + str(n_repeats) + 'reps'
    save_name += '_' + str(settings.nlive_const) + 'nlive'
    if settings.dynamic_goal is not None or include_dg is False:
        save_name += '_' + str(settings.ninit) + 'ninit'
        if settings.nbatch != 1:
            save_name += '_' + str(settings.nbatch) + 'nbatch'
    if settings.n_samples_max is not None and settings.nlive_const is None:
        save_name += '_' + str(settings.n_samples_max) + 'sampmax'
    if settings.tuned_dynamic_p is True and settings.dynamic_goal is not None:
        save_name += '_tuned'
    save_name = save_name.replace('.', '_')
    save_name = save_name.replace('-', '_')
    return save_name


@timing_decorator
def pickle_save(data, name, **kwargs):
    """Saves object with pickle,  appending name with the time file exists."""
    extension = kwargs.pop('extension', '.pkl')
    overwrite_existing = kwargs.pop('overwrite_existing', False)
    print_filename = kwargs.pop('print_filename', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    filename = name + extension
    # Check if the target directory exists and if not make it
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname) and dirname != '':
        try:
            os.makedirs(dirname)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
    if os.path.isfile(filename) and not overwrite_existing:
        filename = name + '_' + time.asctime().replace(' ', '_')
        filename += extension
        print('File already exists! Saving with time appended')
    if print_filename:
        print(filename)
    try:
        outfile = open(filename, 'wb')
        pickle.dump(data, outfile)
        outfile.close()
    except MemoryError:
        print('pickle_save could not save data due to memory error: exiting ' +
              'without saving')


@timing_decorator
def pickle_load(name, extension='.pkl'):
    """Load data with pickle."""
    filename = name + extension
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data
