#!/usr/bin/env python
"""
Helper functions for saving, loading and input/output.
"""


import functools
import os.path
import pickle
import time
import warnings


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
            whether or not to print time function takes.
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
            File name including directory and excluding extension.
        save: bool, optional
            Whether or not to save.
        load: bool, optional
            Whether or not to load.
        overwrite existing: bool, optional
            When the save path already contains file: if True, file will be
            overwritten, if False the data will be saved with the system time
            appended to the file name.
        warn_if_error: bool, optional
            Whether or not to issue UserWarning if load=True and save_name
            is not None but there is an error loading.

        Returns
        -------
        Result
            func output.
        """
        save_name = kwargs.pop('save_name', None)
        save = kwargs.pop('save', save_name is not None)
        load = kwargs.pop('load', save_name is not None)
        overwrite_existing = kwargs.pop('overwrite_existing', True)
        warn_if_error = kwargs.pop('warn_if_error', False)
        if load:
            if save_name is None:
                warnings.warn(
                    ('{} has load=True but cannot load because '
                     'save_name=None'.format(func.__name__)),
                    UserWarning)
            else:
                try:
                    return pickle_load(save_name)
                except (OSError, IOError) as err:
                    if warn_if_error:
                        msg = ('{} had {} loading file {}.'.format(
                            func.__name__, type(err).__name__, save_name))
                        msg = ' Continuing without loading.'
                        warnings.warn(msg, UserWarning)
        result = func(*args, **kwargs)
        if save:
            if save_name is None:
                warnings.warn((func.__name__ + ' has save=True but cannot ' +
                               'save because save_name=None'), UserWarning)
            else:
                pickle_save(result, save_name,
                            overwrite_existing=overwrite_existing)
        return result
    return wrapper


@timing_decorator
def pickle_save(data, name, **kwargs):
    """
    Saves object with pickle.

    Parameters
    ----------
    data: anything picklable
        Object to save.
    name: str
        Path to save to (includes dir, excludes extension).
    extension: str, optional
        File extension.
    overwrite existing: bool, optional
        When the save path already contains file: if True, file will be
        overwritten, if False the data will be saved with the system time
        appended to the file name.
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
    # check if permission error is defined (was not before python 3.3)
    # and otherwise use IOError
    try:
        PermissionError
    except NameError:
        PermissionError = IOError
    try:
        outfile = open(filename, 'wb')
        pickle.dump(data, outfile)
        outfile.close()
    except (MemoryError, PermissionError) as err:
        warnings.warn((type(err).__name__ + ' in pickle_save: continue without'
                       ' saving.'), UserWarning)


@timing_decorator
def pickle_load(name, extension='.pkl'):
    """
    Load data with pickle.

    Parameters
    ----------
    name: str
        Path to save to (includes dir, excludes extension).
    extension: str, optional
        File extension.

    Returns
    -------
    Contents of file path.
    """
    filename = name + extension
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data
