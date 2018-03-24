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
    Prints the time func takes to execute.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper for printing execution time.

        Parameters
        ----------
        print_time: bool, optional
            whether or not to save
        """
        print_time = kwargs.pop('print_time', False)
        if not print_time:
            return func(*args, **kwargs)
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(func.__name__ + ' took %.3f seconds' %
                  (end_time - start_time))
            return result
    return wrapper


def save_load_result(func):
    """
    Saves and/or loads func output (must be picklable).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Default behavior is no saving and loading. Specify save_name to save
        and load.

        Parameters
        ----------
        save_name: str, optional
            file name including directory and excluding extension.
        save: bool, optional
            whether or not to save
        load: bool, optional
            whether or not to load

        Returns
        -------
        result:
            func output
        """
        save_name = kwargs.pop('save_name', None)
        save = kwargs.pop('save', save_name is not None)
        load = kwargs.pop('load', save_name is not None)
        if load:
            if save_name is None:
                print('WARNING: ' + func.__name__ + ' cannot load:',
                      'save_name=None')
            else:
                try:
                    return pickle_load(save_name)
                except OSError:
                    pass
        result = func(*args, **kwargs)
        if save:
            if save_name is None:
                print('WARNING: ' + func.__name__ + ' cannot save:',
                      'save_name=None')
            else:
                pickle_save(result, save_name)
        return result
    return wrapper


@timing_decorator
def pickle_save(data, name, **kwargs):
    """
    Saves object with pickle.

    Parameters
    ----------
    data: anything picklable
        object to save
    name: str
        path to save to (includes dir, excludes extension).
    extension: str, optional
        file extension.
    overwrite existing: bool, optional
        When the save path already contains file: if True, file will be
        overwritten, if False the data will be saved with the system time
        appended to the file name
    """
    extension = kwargs.pop('extension', '.pkl')
    overwrite_existing = kwargs.pop('overwrite_existing', True)
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
    try:
        outfile = open(filename, 'wb')
        pickle.dump(data, outfile)
        outfile.close()
    except MemoryError:
        print('pickle_save could not save data due to memory error: exiting ' +
              'without saving')


@timing_decorator
def pickle_load(name, extension='.pkl'):
    """
    Load data with pickle.

    Parameters
    ----------
    name: str
        path to save to (includes dir, excludes extension).
    extension: str, optional
        file extension.

    Returns
    -------
    contents of file path
    """
    filename = name + extension
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data
