#!/usr/bin/env python
"""
High-level functions for calculating results of error analysis and diagnostic
tests for batches of nested sampling runs.
"""

import numpy as np
import pandas as pd
import nestcheck.error_analysis
import nestcheck.io_utils
import nestcheck.ns_run_utils
import nestcheck.parallel_utils as pu
import nestcheck.pandas_functions as pf


@nestcheck.io_utils.save_load_result
def run_list_error_values(run_list, estimator_list, estimator_names,
                          n_simulate=100, **kwargs):
    """
    Gets a data frame with calculation values and error diagnostics for each
    run in the input run list.

    NB when parallelised the results will not be produced in order (so results
    from some run number will not nessesarily correspond to that number run in
    run_list).

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs
        Name of each func in estimator_list.
    n_simulate: int, optional
        Number of bootstrap replications to use on each run.
    thread_pvalue: bool, optional
        Whether or not to compute KS test diaganostic for correlations between
        threads within a run.
    bs_stat_dist: bool, optional
        Whether or not to compute statistical distance between bootstrap error
        distributions diaganostic.
    parallel: bool, optional
        Whether or not to parallelise - see parallel_utils.parallel_apply.
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
    df: pandas DataFrame
        Results table showing calculation values and diagnostics. Rows
        show different runs (or pairs of runs for pairwise comparisons).
        Columns have titles given by estimator_names and show results for the
        different functions in estimators_list.
    """
    thread_pvalue = kwargs.pop('thread_pvalue', False)
    bs_stat_dist = kwargs.pop('bs_stat_dist', False)
    parallel = kwargs.pop('parallel', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    # Calculation results
    # -------------------
    df = estimator_values_df(run_list, estimator_list, parallel=parallel,
                             estimator_names=estimator_names)
    df.index = df.index.map(str)
    df['calculation type'] = 'values'
    df.set_index('calculation type', drop=True, append=True, inplace=True)
    df = df.reorder_levels(['calculation type', 'run'])
    # Bootstrap stds
    # --------------
    # Create bs_vals_df then convert to stds so bs_vals_df does not need to be
    # recomputed if bs_stat_dist is True
    bs_vals_df = bs_values_df(run_list, estimator_list, estimator_names,
                              n_simulate, parallel=parallel)
    bs_std_df = bs_vals_df.applymap(lambda x: np.std(x, ddof=1))
    bs_std_df.index.name = 'run'
    bs_std_df['calculation type'] = 'bootstrap std'
    bs_std_df.set_index('calculation type', drop=True, append=True,
                        inplace=True)
    bs_std_df = bs_std_df.reorder_levels(['calculation type', 'run'])
    df = pd.concat([df, bs_std_df])
    # Pairwise KS p-values on threads
    # -------------------------------
    if thread_pvalue:
        t_vals_df = thread_values_df(
            run_list, estimator_list, estimator_names, parallel=parallel)
        t_d_df = pairwise_dists_on_cols(t_vals_df, earth_mover_dist=False,
                                        energy_dist=False)
        # Keep only the p value not the distance measures
        t_d_df = t_d_df.xs('ks pvalue', level='calculation type',
                           drop_level=False)
        # Append 'thread ' to caclulcation type
        t_d_df.index.set_levels(['thread ks pvalue'], level='calculation type',
                                inplace=True)
        df = pd.concat([df, t_d_df])
    # Pairwise distances on BS distributions
    # --------------------------------------
    if bs_stat_dist:
        b_d_df = pairwise_dists_on_cols(bs_vals_df)
        # Select only statistical distances - not KS pvalue as this is not
        # useful for the bootstrap resample distributions (see Higson 2018 for
        # more details).
        dists = ['ks distance', 'earth mover distance', 'energy distance']
        b_d_df = b_d_df.loc[pd.IndexSlice[dists, :], :]
        # Append 'bootstrap ' to caclulcation type
        new_ind = ['bootstrap ' +
                   b_d_df.index.get_level_values('calculation type'),
                   b_d_df.index.get_level_values('run')]
        b_d_df.set_index(new_ind, inplace=True)
        df = pd.concat([df, b_d_df])
    return df


@nestcheck.io_utils.save_load_result
def estimator_values_df(run_list, estimator_list, **kwargs):
    """
    Get a dataframe of estimator values.

    NB when parallelised the results will not be produced in order (so results
    from some run number will not nessesarily correspond to that number run in
    run_list).

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs, optional
        Name of each func in estimator_list.
    parallel: bool, optional
        Whether or not to parallelise - see parallel_utils.parallel_apply.
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
    df: pandas DataFrame
        Results table showing calculation values and diagnostics. Rows
        show different runs.
        Columns have titles given by estimator_names and show results for the
        different functions in estimators_list.
    """
    estimator_names = kwargs.pop(
        'estimator_names',
        ['est_' + str(i) for i in range(len(estimator_list))])
    parallel = kwargs.pop('parallel', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    values_list = pu.parallel_apply(
        nestcheck.ns_run_utils.run_estimators, run_list,
        func_args=(estimator_list,), parallel=parallel)
    df = pd.DataFrame(np.stack(values_list, axis=0))
    df.columns = estimator_names
    df.index.name = 'run'
    return df


def error_values_summary(error_values, **summary_df_kwargs):
    """
    Get summary statistics about calculation errors, including estimated
    implementation errors.

    Parameters
    ----------
    error_values: pandas DataFrame
        Of format output by run_list_error_values (look at it for more
        details).
    summary_df_kwargs: dict, optional
        See pandas_functions.summary_df docstring for more details.

    Returns
    -------
    df: pandas DataFrame
        Table showing means and standard deviations of results and diagnostics
        for the different runs. Also contains estimated numerical uncertainties
        on results.
    """
    df = pf.summary_df_from_multi(error_values, **summary_df_kwargs)
    # get implementation stds
    imp_std, imp_std_unc, imp_frac, imp_frac_unc = \
        nestcheck.error_analysis.implementation_std(
            df.loc[('values std', 'value')],
            df.loc[('values std', 'uncertainty')],
            df.loc[('bootstrap std mean', 'value')],
            df.loc[('bootstrap std mean', 'uncertainty')])
    df.loc[('implementation std', 'value'), df.columns] = imp_std
    df.loc[('implementation std', 'uncertainty'), df.columns] = imp_std_unc
    df.loc[('implementation std frac', 'value'), :] = imp_frac
    df.loc[('implementation std frac', 'uncertainty'), :] = imp_frac_unc
    # Get implementation RMSEs (calculated using the values RMSE instead of
    # values std)
    if 'values rmse' in set(df.index.get_level_values('calculation type')):
        imp_rmse, imp_rmse_unc, imp_frac, imp_frac_unc = \
            nestcheck.error_analysis.implementation_std(
                df.loc[('values rmse', 'value')],
                df.loc[('values rmse', 'uncertainty')],
                df.loc[('bootstrap std mean', 'value')],
                df.loc[('bootstrap std mean', 'uncertainty')])
        df.loc[('implementation rmse', 'value'), df.columns] = imp_rmse
        df.loc[('implementation rmse', 'uncertainty'), df.columns] = \
            imp_rmse_unc
        df.loc[('implementation rmse frac', 'value'), :] = imp_frac
        df.loc[('implementation rmse frac', 'uncertainty'), :] = imp_frac_unc
    # Return only the calculation types we are interested in, in order
    calcs_to_keep = ['true values', 'values mean', 'values std',
                     'values rmse', 'bootstrap std mean',
                     'implementation std', 'implementation std frac',
                     'implementation rmse', 'implementation rmse frac',
                     'thread ks pvalue mean', 'bootstrap ks distance mean',
                     'bootstrap energy distance mean',
                     'bootstrap earth mover distance mean']
    df = pd.concat([df.xs(calc, level='calculation type', drop_level=False) for
                    calc in calcs_to_keep if calc in
                    df.index.get_level_values('calculation type')])
    return df


def run_list_error_summary(run_list, estimator_list, estimator_names,
                           n_simulate, **kwargs):
    """
    Wrapper which runs run_list_error_values then applies error_values summary
    to the resulting dataframe. See the docstrings for those two funcions for
    more details and for descriptions of parameters and output.
    """
    true_values = kwargs.pop('true_values', None)
    include_true_values = kwargs.pop('include_true_values', False)
    include_rmse = kwargs.pop('include_rmse', False)
    error_values = run_list_error_values(run_list, estimator_list,
                                         estimator_names, n_simulate, **kwargs)
    return error_values_summary(error_values, true_values=true_values,
                                include_true_values=include_true_values,
                                include_rmse=include_rmse)


def bs_values_df(run_list, estimator_list, estimator_names, n_simulate,
                 **kwargs):
    """
    Computes a data frame of bootstrap resampled values.

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs
        Name of each func in estimator_list.
    n_simulate: int
        Number of bootstrap replications to use on each run.
    kwargs:
        Kwargs to pass to parallel_apply.

    Returns
    -------
    bs_values_df: pandas data frame
        Columns represent estimators and rows represent runs.
        Each cell contains a 1d array of bootstrap resampled values for the run
        and estimator.
    """
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'desc': 'bs values'})
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    bs_values_list = pu.parallel_apply(
        nestcheck.error_analysis.run_bootstrap_values, run_list,
        func_args=(estimator_list,), func_kwargs={'n_simulate': n_simulate},
        tqdm_kwargs=tqdm_kwargs, **kwargs)
    df = pd.DataFrame()
    for i, name in enumerate(estimator_names):
        df[name] = [arr[i, :] for arr in bs_values_list]
    # Check there are the correct number of bootstrap replications in each cell
    for vals_shape in df.loc[0].apply(lambda x: x.shape).values:
        assert vals_shape == (n_simulate,), (
            'Should be n_simulate=' + str(n_simulate) + ' values in ' +
            'each cell. The cell contains array with shape ' +
            str(vals_shape))
    return df


def thread_values_df(run_list, estimator_list, estimator_names, **kwargs):
    """
    Calculates estimator values for the constituent threads of the input runs.

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs
        Name of each func in estimator_list.
    kwargs:
        Kwargs to pass to parallel_apply.

    Returns
    -------
    df: pandas data frame
        Columns represent estimators and rows represent runs.
        Each cell contains a 1d numpy array with length equal to the number
        of threads in the run, containing the results from evaluating the
        estimator on each thread.
    """
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'desc': 'thread values'})
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    # get thread results
    thread_vals_arrays = pu.parallel_apply(
        nestcheck.error_analysis.run_thread_values, run_list,
        func_args=(estimator_list,), tqdm_kwargs=tqdm_kwargs, **kwargs)
    df = pd.DataFrame()
    for i, name in enumerate(estimator_names):
        df[name] = [arr[i, :] for arr in thread_vals_arrays]
    # Check there are the correct number of thread values in each cell
    for vals_shape in df.loc[0].apply(lambda x: x.shape).values:
        assert vals_shape == (run_list[0]['thread_min_max'].shape[0],), \
            ('Should be nlive=' + str(run_list[0]['thread_min_max'].shape[0]) +
             ' values in each cell. The cell contains array with shape ' +
             str(vals_shape))
    return df


def pairwise_dists_on_cols(df_in, earth_mover_dist=True, energy_dist=True):
    """
    Computes pairwise statistical distance measures.

    parameters
    ----------
    df_in: pandas data frame
        Columns represent estimators and rows represent runs.
        Each data frane element is an array of values which are used as samples
        in the distance measures.

    returns
    -------
    df: pandas data frame with kl values for each pair.
    """
    df = pd.DataFrame()
    for col in df_in.columns:
        df[col] = nestcheck.error_analysis.pairwise_distances(
            df_in[col].values, earth_mover_dist=earth_mover_dist,
            energy_dist=energy_dist)
    return df
