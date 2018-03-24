#!/usr/bin/env python
"""
Contains helper functions for saving, loading and input/output.
"""


import time
import pickle
import functools
import os.path


def timing_decorator(func):
    """
    Prints the time a function takes to execute.
    """
    @functools.wraps(func)
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


def save_load_result(func):
    """
    Saves and/or loads function results.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper for saving and loading"""
        save_name = kwargs.pop('save_name', None)
        save_dir = kwargs.pop('save_dir', 'cache')
        save = kwargs.pop('save', save_name is not None)
        load = kwargs.pop('load', save_name is not None)
        if load:
            if save_name is None:
                print('WARNING: ' + func.__name__ + ' cannot load:',
                      'save_name=None')
            else:
                try:
                    return pickle_load(save_dir + '/' + save_name)
                except OSError:
                    pass
        result = func(*args, **kwargs)
        if save:
            if save_name is None:
                print('WARNING: ' + func.__name__ + ' cannot save:',
                      'save_name=None')
            else:
                pickle_save(result, save_dir + '/' + save_name)
        return result
    return wrapper


@timing_decorator
def pickle_save(data, name, **kwargs):
    """Saves object with pickle,  appending name with the time file exists."""
    extension = kwargs.pop('extension', '.pkl')
    overwrite_existing = kwargs.pop('overwrite_existing', True)
    print_filename = kwargs.pop('print_filename', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    filename = name + extension
    # Check if the target directory exists and if not make it
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname)
    if os.path.isfile(filename) and not overwrite_existing:
        print(filename + ' already exists! Saving with time appended')
        filename = name + '_' + time.asctime().replace(' ', '_')
        filename += extension
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
