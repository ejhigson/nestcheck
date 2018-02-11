#!/usr/bin/env python
"""
Parallel wrapper functions using the concurrent.futures module.
"""

import concurrent.futures
import tqdm
import PerfectNS.save_load_utils as slu


@slu.timing_decorator
def parallel_mapper(func, list_to_map, *args, **kwargs):
    """
    Performs function on a list first arguments with paralleisation and a nice
    progress bar.

    Parameters
    ----------
    func: function
        Function acting each list element with arguments
            func(list_to_map[i], *args, **kwarfs)
    list_to_map: list
        Arguments for func
    parallelise: bool, optional
        To turn off parallelisation if needed.
    max_workers: int or None, optional
        Number of processes.
        If max_workers is None then concurrent.futures.ProcessPoolExecutor
        defaults to using the number of processors of the machine.
        N.B. If max_workers=None and running on supercomputer clusters with
        multiple nodes, this may default to the number of processors on a
        single node and therefore there will be no speedup from multiple
        nodes (must specify manually in this case).
    Returns
    -------
    results_list: list of function outputs
    """
    # Use pop instead of get so parallel_mapper's kwargs are not passed to func
    max_workers = kwargs.pop('max_workers', None)
    parallelise = kwargs.pop('parallelise', True)
    use_tqdm = kwargs.pop('use_tqdm', True)
    tqdm_desc = kwargs.pop('tqdm_desc', None)
    results_list = []
    if parallelise:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        futures = []
        for element in list_to_map:
            futures.append(pool.submit(func, element, *args, **kwargs))
        if use_tqdm:
            for fut in tqdm.tqdm_notebook(concurrent.futures
                                          .as_completed(futures),
                                          desc=tqdm_desc,
                                          leave=False, total=len(futures)):
                results_list.append(fut.result())
        else:
            for fut in concurrent.futures.as_completed(futures):
                results_list.append(fut.result())
        del futures
        del pool
    else:
        print('Warning: func_on_runs not parallelised!')
        if use_tqdm:
            for element in tqdm.tqdm_notebook(list_to_map, desc=tqdm_desc,
                                              leave=False):
                results_list.append(func(element, *args, **kwargs))
        else:
            for element in list_to_map:
                results_list.append(func(element, *args, **kwargs))
    return results_list
