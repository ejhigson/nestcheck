#!/usr/bin/env python
"""
Parallel wrapper functions using the concurrent.futures module.
"""

import concurrent.futures
import tqdm


def parallel_map(func, *iterables, chunksize=1, max_workers=None,
                 parallelise=True):
    """
    map function parallelised with concurrent.futures.ProcessPoolExecutor.
    """
    if parallelise:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        return pool.map(func, *iterables, chunksize=chunksize)
    else:
        return map(func, *iterables)


def parallel_apply(func, arg_iterable, **kwargs):
    """
    Apply function to iterable with paralleisation and a tqdm progress bar.
    Returns results in order.

    Equivalent to

    [func(*func_pre_args, x, *func_args, **func_kwargs) for x in arg_iterable]

    Parameters
    ----------
    func: function
        Function to apply to list of args.
    arg_iterable: iterable
        argument to iterate over.
    func_args: tuple
        Additional positional arguments for func.
    func_pre_args: tuple
        Positional arguments to place before the iterable argument in func.
    func_kwargs:
        Additional keyword argumnets for func.
    parallelise: bool, optional
        To turn off parallelisation if needed.
    max_workers: int or None, optional
        Number of processes.
        If max_workers is None then concurrent.futures.ProcessPoolExecutor
        defaults to using the number of processors of the machine.
        N.B. If max_workers=None and running on supercomputer clusters with
        multiple nodes, this may default to the number of processors on a
        single node.
    Returns
    -------
    results_list: list of function outputs
    """
    max_workers = kwargs.pop('max_workers', None)
    parallelise = kwargs.pop('parallelise', True)
    func_args = kwargs.pop('func_args', ())
    func_pre_args = kwargs.pop('func_pre_args', ())
    func_kwargs = kwargs.pop('func_kwargs', {})
    tqdm_desc = kwargs.pop('tqdm_desc', None)
    tqdm_leave = kwargs.pop('tqdm_leave', False)
    tqdm_disable = kwargs.pop('tqdm_disable', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # If running in a jupyter notebook then use tqdm_notebook. Otherwise use
    # regular tqdm progress bar
    try:
        ip = get_ipython()
        assert ip.has_trait('kernel')
        progress = tqdm.tqdm_notebook
    except (NameError, AssertionError):
        progress = tqdm.tqdm
    if not parallelise:
        print('Warning: parallel_apply not parallelised!')
        return [func(*func_pre_args, x, *func_args, **func_kwargs) for x in
                progress(arg_iterable, desc=tqdm_desc, leave=tqdm_leave,
                         disable=tqdm_disable)]
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        futures = []
        for element in arg_iterable:
            futures.append(pool.submit(func, *func_pre_args, element,
                                       *func_args, **func_kwargs))
        # Progress bar for completion of tasks
        for _ in progress(concurrent.futures.as_completed(futures),
                          desc=tqdm_desc, leave=tqdm_leave,
                          disable=tqdm_disable, total=len(futures)):
            pass
        # Use concurrent.futures.wait to return results in order
        return [fut.result() for fut in concurrent.futures.wait(futures)]
