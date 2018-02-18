#!/usr/bin/env python
"""
Diagnostic tests for nested sampling runs.
"""

import numpy as np
import pandas as pd
import scipy
import tqdm
import nestcheck.analyse_run as ar
import nestcheck.parallel_utils as pu
import nestcheck.pandas_functions as pf


def analyse_run_errors(run_list, estimator_list, n_simulate, **kwargs):
    """
    Gets a data frame with calculation values and error estimates for each run
    in the input run list.
    """
    thread_dist = kwargs.pop('thread_test', True)
    bs_dist = kwargs.pop('bs_test', True)
    cache_root = kwargs.pop('cache_root', None)
    # Do caching
    # ----------
    if cache_root is not None:
        save = kwargs.pop('save', True)
        load = kwargs.pop('load', True)
        cache_name = ('cache/' + cache_root + '_' + str(len(run_list)) +
                      'runs_' + str(n_simulate) + 'sim')
        if thread_dist:
            cache_name += '_td'
        if bs_dist:
            cache_name += '_bd'
        cache_name += '.pkl'
    else:
        save = kwargs.pop('save', False)
        if save:
            print('WARNING: analyse_run_errors cannot save: no cache given')
        load = kwargs.pop('load', False)
        if load:
            print('WARNING: analyse_run_errors cannot load: no cache given')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    if load:
        try:
            df = pd.read_pickle(cache_name)
            return df
        except OSError:
            load = False
    if not load:
        # Calculate results
        # -----------------
        # Values
        values_list = pu.parallel_apply(ar.run_estimators, run_list,
                                        func_args=(estimator_list,),
                                        parallelise=True)
        estimator_names = [est.name for est in estimator_list]
        df = pd.DataFrame(np.stack(values_list, axis=0))
        df.index = df.index.map(str)
        df.columns = estimator_names
        df.index.name = 'run'
        df['calculation type'] = 'values'
        df.set_index('calculation type', drop=True, append=True, inplace=True)
        df = df.reorder_levels(['calculation type', 'run'])
        # Bootstrap
        bs_vals_df = bs_values_df(run_list, estimator_list, n_simulate,
                                  use_tqdm=True)
        # ####################################
        # For checking values are as expected
        bs_mean_df = bs_vals_df.applymap(np.mean)
        bs_mean_df.index.name = 'run'
        bs_mean_df['calculation type'] = 'bootstrap mean'
        bs_mean_df.set_index('calculation type', drop=True, append=True,
                             inplace=True)
        bs_mean_df = bs_mean_df.reorder_levels(['calculation type', 'run'])
        df = pd.concat([df, bs_mean_df])
        # ####################################
        bs_std_df = bs_vals_df.applymap(lambda x: np.std(x, ddof=1))
        bs_std_df.index.name = 'run'
        bs_std_df['calculation type'] = 'bootstrap std'
        bs_std_df.set_index('calculation type', drop=True, append=True,
                            inplace=True)
        bs_std_df = bs_std_df.reorder_levels(['calculation type', 'run'])
        df = pd.concat([df, bs_std_df])
        # Pairwise distances on thread distributions
        if thread_dist:
            t_vals_df = thread_values_df(run_list, estimator_list)
            # ####################################
            # For checking values are as expected
            t_mean_df = t_vals_df.applymap(np.mean)
            t_mean_df.index.name = 'run'
            t_mean_df['calculation type'] = 'thread mean'
            t_mean_df.set_index('calculation type', drop=True, append=True,
                                inplace=True)
            t_mean_df = t_mean_df.reorder_levels(['calculation type', 'run'])
            df = pd.concat([df, t_mean_df])
            # ####################################
            t_d_df = pairwise_distances_on_cols(t_vals_df,
                                                tqdm_desc='thread dists')
            new_ind = ['thread ' + t_d_df.index
                       .get_level_values('calculation type'),
                       t_d_df.index.get_level_values('run')]
            t_d_df.set_index(new_ind, inplace=True)
            df = pd.concat([df, t_d_df])
        # Pairwise distances on BS distributions
        if bs_dist:
            b_d_df = pairwise_distances_on_cols(bs_vals_df,
                                                tqdm_desc='bs dists')
            new_ind = ['bootstrap ' + b_d_df.
                       index.get_level_values('calculation type'),
                       b_d_df.index.get_level_values('run')]
            b_d_df.set_index(new_ind, inplace=True)
            df = pd.concat([df, b_d_df])
        if save:
            df.to_pickle(cache_name)
    return df


def run_error_summary(df):
    """
    Gets mean error estimates and uncertainties for the run list.
    """
    df = pf.summary_df_from_multi(df)
    # get implementation stds
    imp_std, imp_std_unc, imp_frac, imp_frac_unc = implementation_std(
        df.loc[('values std', 'value')],
        df.loc[('values std', 'uncertainty')],
        df.loc[('bootstrap std mean', 'value')],
        df.loc[('bootstrap std mean', 'uncertainty')], return_frac=True)
    df.loc[('implementation std', 'value'), df.columns] = imp_std
    df.loc[('implementation std', 'uncertainty'), df.columns] = imp_std_unc
    df.loc[('implementation std frac', 'value'), :] = imp_frac
    df.loc[('implementation std frac', 'uncertainty'), :] = imp_frac_unc
    return df


def implementation_std(vals_std, vals_std_u, bs_std, bs_std_u,
                       return_frac=True):
    """
    Estimates implementation errors from the standard
    deviations of results and of bootstrap values.

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
    if return_frac:
        imp_frac = imp_std / vals_std
        imp_frac_u = np.zeros(imp_frac.shape)
    # Simulate errors dirstributions using the fact that (from central limit
    # theorem) our uncertainties on vals_std and bs_std are normal
    # distributions
    size = 10 ** 6
    for i, _ in enumerate(imp_std_u):
        sim_vals_std = np.random.normal(vals_std[i], vals_std_u[i], size=size)
        sim_bs_std = np.random.normal(bs_std[i], bs_std_u[i], size=size)
        sim_imp_var = (sim_vals_std ** 2) - (sim_bs_std ** 2)
        sim_imp_std = np.sqrt(np.abs(sim_imp_var)) * np.sign(sim_imp_var)
        imp_std_u[i] = np.std(sim_imp_std, ddof=1)
        if return_frac:
            imp_frac_u[i] = np.std((sim_imp_std / sim_vals_std), ddof=1)
    if return_frac:
        return imp_std, imp_std_u, imp_frac, imp_frac_u
    else:
        return imp_std, imp_std_u


def bs_values_df(run_list, estimator_list, n_simulate, use_tqdm=False):
    """
    Computes a data frame of bootstrap resampled values.

    returns
    -------
    bs_values_df: pandas data frame
        columns represent estimators and rows represent runs.
        each list element is an array of bootstrap resampled values of that
        estimator applied to that run.
    """
    bs_values_list = pu.parallel_apply(ar.run_bootstrap_values, run_list,
                                       func_args=(estimator_list,),
                                       func_kwargs={'n_simulate': n_simulate},
                                       tqdm_desc='bs_values_df')
    df = pd.DataFrame()
    for i, est in enumerate(estimator_list):
        df[est.name] = [arr[i, :] for arr in bs_values_list]
    # Check there are the correct number of bootstrap replications in each cell
    for vals_shape in df.loc[0].apply(lambda x: x.shape).values:
        assert vals_shape == (n_simulate,), \
            ('Should be n_simulate=' + str(n_simulate) + ' values in ' +
             'each cell. The cell contains array with shape ' +
             str(vals_shape))
    return df


def thread_values_df(run_list, estimator_list):
    """Returns df containing estimator values for individual threads."""
    # get thread results
    thread_arr_l = []
    threads_list = pu.parallel_apply(ar.get_run_threads, run_list,
                                     tqdm_desc='separating threads')
    for threads in tqdm.tqdm_notebook(threads_list, desc='thread values',
                                      leave=False):
        threads_as_runs = []
        for th in threads:
            min_max = np.reshape(np.asarray((np.nan, th[-1, 0])), (1, 2))
            threads_as_runs.append(ar.dict_given_samples_array(th, min_max))
        thread_vals = pu.parallel_apply(ar.run_estimators, threads_as_runs,
                                        func_args=(estimator_list,))
        # convert list of estimator results on each thread to a numpy array
        # with a column for each thread and a row for each estimator
        thread_arr = np.stack(thread_vals, axis=1)
        thread_arr_l.append(thread_arr)
    df = pd.DataFrame()
    # print(len(thread_arr_l), [a.shape for a in thread_arr_l])
    for i, est in enumerate(estimator_list):
        df[est.name] = [arr[i, :] for arr in thread_arr_l]
    # Check there are the correct number of thread values in each cell
    for vals_shape in df.loc[0].apply(lambda x: x.shape).values:
        assert vals_shape == (len(threads_list[0]),), \
            ('Should be nlive=' + str(len(threads_list[0])) + ' values in ' +
             'each cell. The cell contains array with shape ' +
             str(vals_shape))
    return df


def pairwise_distances_on_cols(df_in, tqdm_desc=None):
    """
    Computes pairwise Kullback–Leibler divergences from a pandas data frame
    of bootstrap value arrays.

    parameters
    ----------
    bs_values_df: pandas data frame
        columns represent estimators and rows represent runs.
        each list element is an array of bootstrap resampled values of that
        estimator applied to that run.

    returns
    -------
    df: pandas data frame with kl values for each pair.
    """
    df = pd.DataFrame()
    for col in tqdm.tqdm_notebook(df_in.columns, desc=tqdm_desc,
                                  leave=False):
        df[col] = pairwise_distances(df_in[col].values)
    return df


def statistical_distances(samples1, samples2):
    """
    Gets 4 measures of the statistical distance between samplese
    ser = pd.DataFrame(out, index=index, columns=['ks pvalue', 'ks distance',
                       'earth mover distance', 'energy distance']).unstack()
    """
    out = np.zeros(4)
    temp = scipy.stats.ks_2samp(samples1, samples2)
    out[0] = temp.pvalue
    out[1] = temp.statistic
    out[2] = scipy.stats.wasserstein_distance(samples1, samples2)
    out[3] = scipy.stats.energy_distance(samples1, samples2)
    return out


def pairwise_distances(dist_list):
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
                out.append(statistical_distances(samp_i, samp_j))
    ser = pd.DataFrame(out, index=index,
                       columns=['ks pvalue', 'ks distance',
                                'earth mover distance',
                                'energy distance']).unstack()
    ser.index.names = ['calculation type', 'run']
    return ser
