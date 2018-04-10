#!/usr/bin/env python
"""
Diagnostic tests for nested sampling runs.
"""

import numpy as np
import pandas as pd
import scipy.stats
import nestcheck.analyse_run as ar
import nestcheck.parallel_utils as pu
import nestcheck.pandas_functions as pf
import nestcheck.io_utils


@nestcheck.io_utils.save_load_result
def run_list_error_values(run_list, estimator_list, estimator_names,
                          n_simulate=100, **kwargs):
    """
    Gets a data frame with calculation values and error estimates for each run
    in the input run list.

    NB when parallelised the results will not be produced in order (so results
    from some run number will not nessesarily correspond to that number run in
    run_list).

    Parameters
    ----------
    run_list: list of dicts
        list of nested sampling runs
    estimator_list: list of functions
        Estimators to apply to runs
    estimator_names: list of strs
        must be same length as estimator_list
    n_simulate: int, optional
        number of bootstrap replications to use on each run
    thread_pvalue: bool, optional
        Whether or not to compute KS test diaganostic for correlations between
        threads within a run.
    bs_stat_dist: bool, optional
        Whether or not to compute statistical distance between bootstrap error
        distributions diaganostic.
    parallel: bool, optional
        whether or not to parallelise - see parallel_utils.parallel_apply
    save_name: str or None, optional
        See nestcheck.io_utils.save_load_result
    save: bool, optional
        See nestcheck.io_utils.save_load_result
    load: bool, optional
        See nestcheck.io_utils.save_load_result
    overwrite_existing: bool, optional
        See nestcheck.io_utils.save_load_result

    Returns
    -------
    df: pandas DataFrame
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
    values_list = pu.parallel_apply(
        ar.run_estimators, run_list, func_args=(estimator_list,),
        parallel=parallel)
    df = pd.DataFrame(np.stack(values_list, axis=0))
    df.index = df.index.map(str)
    df.columns = estimator_names
    df.index.name = 'run'
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
        t_d_df = pairwise_distances_on_cols(
            t_vals_df, earth_mover_dist=False, energy_dist=False)
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
        b_d_df = pairwise_distances_on_cols(bs_vals_df)
        # Select only statistical distances - not KS pvalue as this is not
        # useful for the bootstrap resample distributions (see Higson 2018 for more
        # details).
        dists = ['ks distance', 'earth mover distance', 'energy distance']
        b_d_df = b_d_df.loc[pd.IndexSlice[dists, :], :]
        # Append 'bootstrap ' to caclulcation type
        new_ind = ['bootstrap ' + b_d_df.index.get_level_values('calculation type'),
                   b_d_df.index.get_level_values('run')]
        b_d_df.set_index(new_ind, inplace=True)
        df = pd.concat([df, b_d_df])
    return df


def error_values_summary(error_values, **summary_df_kwargs):
    """
    Get summary statistics about calculation errors, including estimated
    implementation errors.

    Parameters
    ----------
    error_values: pandas DataFrame
        Of format output by run_list_error_values (look at it for more details)
    summary_df_kwargs: dict, optional
        See pandas_functions.summary_df docstring for more details.
    """
    df = pf.summary_df_from_multi(error_values, **summary_df_kwargs)
    # get implementation stds
    imp_std, imp_std_unc, imp_frac, imp_frac_unc = implementation_std(
        df.loc[('values std', 'value')],
        df.loc[('values std', 'uncertainty')],
        df.loc[('bootstrap std mean', 'value')],
        df.loc[('bootstrap std mean', 'uncertainty')])
    df.loc[('implementation std', 'value'), df.columns] = imp_std
    df.loc[('implementation std', 'uncertainty'), df.columns] = imp_std_unc
    df.loc[('implementation std frac', 'value'), :] = imp_frac
    df.loc[('implementation std frac', 'uncertainty'), :] = imp_frac_unc
    # Return only the calculation types we are interested in, in order
    calcs_to_keep = ['true values', 'values mean', 'values std',
                     'values rmse', 'bootstrap std mean',
                     'implementation std', 'implementation std frac',
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
    more details.
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

    Returns
    -------
    bs_values_df: pandas data frame
        columns represent estimators and rows represent runs.
        each list element is an array of bootstrap resampled values of that
        estimator applied to that run.
    """
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'desc': 'bs values'})
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    bs_values_list = pu.parallel_apply(ar.run_bootstrap_values, run_list,
                                       func_args=(estimator_list,),
                                       func_kwargs={'n_simulate': n_simulate},
                                       tqdm_kwargs=tqdm_kwargs, **kwargs)
    df = pd.DataFrame()
    for i, name in enumerate(estimator_names):
        df[name] = [arr[i, :] for arr in bs_values_list]
    # Check there are the correct number of bootstrap replications in each cell
    for vals_shape in df.loc[0].apply(lambda x: x.shape).values:
        assert vals_shape == (n_simulate,), \
            ('Should be n_simulate=' + str(n_simulate) + ' values in ' +
             'each cell. The cell contains array with shape ' +
             str(vals_shape))
    return df


def thread_values_df(run_list, estimator_list, estimator_names, **kwargs):
    """Returns df containing estimator values for individual threads."""
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'desc': 'thread values'})
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    # get thread results
    thread_vals_arrays = pu.parallel_apply(run_thread_values, run_list,
                                           func_args=(estimator_list,),
                                           tqdm_kwargs=tqdm_kwargs, **kwargs)
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


# Helper functions
# ----------------


def implementation_std(vals_std, vals_std_u, bs_std, bs_std_u):
    """
    Estimates implementation errors from the standard deviations of results
    and of bootstrap values. See "Diagnostic tests for nested sampling
    calculations" (Higson et al. 2018) for more details.

    Simulate errors dirstributions using the fact that (from central limit
    theorem) our uncertainties on vals_std and bs_std are normal
    distributions
    """
    # if the implementation errors are uncorrelated with the
    # sampling errrors: var results = var imp + var sampling
    # so std imp = sqrt(var results - var sampling)
    imp_var = (vals_std ** 2) - (bs_std ** 2)
    imp_std = np.sqrt(np.abs(imp_var)) * np.sign(imp_var)
    ind = np.where(imp_std <= 0)[0]
    imp_std[ind] = 0
    imp_std_u = np.zeros(imp_std.shape)
    imp_frac = imp_std / vals_std
    imp_frac_u = np.zeros(imp_frac.shape)
    # Simulate errors distributions
    size = 10 ** 6
    for i, _ in enumerate(imp_std_u):
        sim_vals_std = np.random.normal(vals_std[i], vals_std_u[i], size=size)
        sim_bs_std = np.random.normal(bs_std[i], bs_std_u[i], size=size)
        sim_imp_var = (sim_vals_std ** 2) - (sim_bs_std ** 2)
        sim_imp_std = np.sqrt(np.abs(sim_imp_var)) * np.sign(sim_imp_var)
        imp_std_u[i] = np.std(sim_imp_std, ddof=1)
        imp_frac_u[i] = np.std((sim_imp_std / sim_vals_std), ddof=1)
    return imp_std, imp_std_u, imp_frac, imp_frac_u


def run_thread_values(run, estimator_list):
    """Helper function for parallelising thread_values_df."""
    threads = ar.get_run_threads(run)
    vals_list = [ar.run_estimators(th, estimator_list) for th in threads]
    vals_array = np.stack(vals_list, axis=1)
    return vals_array


def pairwise_distances_on_cols(df_in, earth_mover_dist=True, energy_dist=True):
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
        df[col] = pairwise_distances(df_in[col].values,
                                     earth_mover_dist=earth_mover_dist,
                                     energy_dist=energy_dist)
    return df


def pairwise_distances(dist_list, earth_mover_dist=True, energy_dist=True):
    """
    Applies statistical_distances to each unique pair of distributions in
    dist_list.
    """
    out = []
    index = []
    for i, samp_i in enumerate(dist_list):
        for j, samp_j in enumerate(dist_list):
            if j < i:
                index.append(str((i, j)))
                out.append(statistical_distances(
                    samp_i, samp_j, earth_mover_dist=earth_mover_dist,
                    energy_dist=energy_dist))
    columns = ['ks pvalue', 'ks distance']
    if earth_mover_dist:
        columns.append('earth mover distance')
    if energy_dist:
        columns.append('energy distance')
    ser = pd.DataFrame(out, index=index, columns=columns).unstack()
    ser.index.names = ['calculation type', 'run']
    return ser


def statistical_distances(samples1, samples2, earth_mover_dist=True,
                          energy_dist=True):
    """
    Gets 4 measures of the statistical distance between samples.
    """
    out = []
    temp = scipy.stats.ks_2samp(samples1, samples2)
    out.append(temp.pvalue)
    out.append(temp.statistic)
    if earth_mover_dist:
        out.append(scipy.stats.wasserstein_distance(samples1, samples2))
    if energy_dist:
        out.append(scipy.stats.energy_distance(samples1, samples2))
    return np.asarray(out)
