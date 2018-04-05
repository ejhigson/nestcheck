#!/usr/bin/env python
"""
Transformations for pandas data frames.
"""


import warnings
import copy
import numpy as np
import pandas as pd


def summary_df_from_array(results_array, names, axis=0, **kwargs):
    """
    Make a panda data frame of the mean and std devs of an array of results,
    including the uncertainties on the values.

    This function converts the array to a DataFrame and calls summary_df on it.

    Parameters
    ----------
    results_array: 2d numpy array
    names: list of str
        names for the output df's columns
    axis: int, optional
        axis on which to calculate summary statistics

    Returns
    -------
    df: MultiIndex DataFrame
        see summary_df docstring for more details
    """
    assert axis == 0 or axis == 1
    df = pd.DataFrame(results_array)
    if axis == 1:
        df = df.T
    df.columns = names
    return summary_df(df, **kwargs)


def summary_df_from_list(results_list, names, **kwargs):
    """
    Make a panda data frame of the mean and std devs of each element of a list
    of 1d arrays, including the uncertainties on the values.

    This just converts the array to a DataFrame and calls summary_df on it.

    Parameters
    ----------
    results_list: list of 1d numpy arrays of length names
    names: list of str
        names for the output df's columns

    Returns
    -------
    df: MultiIndex DataFrame
        see summary_df docstring for more details
    """
    for arr in results_list:
        assert arr.shape == (len(names),)
    df = pd.DataFrame(np.stack(results_list, axis=0))
    df.columns = names
    return summary_df(df, **kwargs)


def summary_df_from_multi(multi_in, inds_to_keep=None, **kwargs):
    """
    summary_df with option to preserve some indexes of a multiindex.
    """
    if inds_to_keep is None:
        inds_to_keep = list(multi_in.index.names)[:-1]
    # Need to pop include true values and add seperately at the end as
    # otherwise we get multiple true values added
    include_true_values = kwargs.pop('include_true_values', False)
    true_values = kwargs.get('true_values', None)
    df = multi_in.groupby(inds_to_keep).apply(
        summary_df, include_true_values=False, **kwargs)
    if 'calculation type' in inds_to_keep:
        # If there is already an index called 'calculation type' in multi,
        # prepend the 'calculation type' values ('mean' and 'std') produced by
        # summary_df to it instead of making a second 'calculation type' index.
        ct_lev = [i for i in range(len(df.index.names)) if df.index.names[i] ==
                  'calculation type']
        ind = (df.index.get_level_values(ct_lev[0]) + ' ' +
               df.index.get_level_values(ct_lev[1]))
        order = list(df.index.names)
        del order[ct_lev[1]]
        df.index = df.index.droplevel(ct_lev)
        df['calculation type'] = list(ind)
        df.set_index('calculation type', append=True, inplace=True)
        df = df.reorder_levels(order)
    if include_true_values:
        assert true_values is not None
        tv_ind = ['true values' if name == 'calculation type' else '' for name in
                  df.index.names[:-1]] + ['value']
        df.loc[tuple(tv_ind), :] = true_values
    return df


def summary_df(df_in, **kwargs):
    """
    Make a panda data frame of the mean and std devs of an array of results,
    including the uncertainties on the values.

    This is similar to pandas.DataFrame.describe but also includes estimates of
    the numerical uncertainties.

    Parameters
    ----------
    df_in: data frame
    true_values: array
        analytical values if known for comparison with mean. Used to
        calculate root mean squared errors (RMSE).
    include_true_values: bool, optional
    include_rmse: bool, optional


    Returns
    -------
    df: MultiIndex DataFrame
        data frame with multiindex ['calculation type', 'result type'] holding
        mean and standard deviations of the data and statistical uncertainties
        on each.
                                            [names]
    calculation type    result type
    mean                value
    mean                uncertainty
    std                 value
    std                 uncertainty
    """
    true_values = kwargs.pop('true_values', None)
    include_true_values = kwargs.pop('include_true_values', False)
    include_rmse = kwargs.pop('include_rmse', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if true_values is not None:
        assert true_values.shape[0] == df_in.shape[1], (
            'There should be one true value for every column! '
            'true_values.shape=' + str(true_values.shape) + ', '
            'df_in.shape=' + str(df_in.shape))
    # make the data frame
    df = pd.DataFrame([df_in.mean(axis=0), df_in.std(axis=0, ddof=1)],
                      index=['mean', 'std'])
    if include_true_values:
        assert true_values is not None
        df.loc['true values'] = true_values
    # Make index categorical to allow sorting
    df.index = pd.CategoricalIndex(df.index.values, ordered=True,
                                   categories=['true values', 'mean', 'std',
                                               'rmse'],
                                   name='calculation type')
    # add uncertainties
    num_cals = df_in.shape[0]
    mean_unc = df.loc['std'] / np.sqrt(num_cals)
    std_unc = df.loc['std'] * np.sqrt(1 / (2 * (num_cals - 1)))
    df['result type'] = pd.Categorical(['value'] * df.shape[0], ordered=True,
                                       categories=['value', 'uncertainty'])
    df.set_index(['result type'], drop=True, append=True, inplace=True)
    df.loc[('mean', 'uncertainty'), :] = mean_unc.values
    df.loc[('std', 'uncertainty'), :] = std_unc.values
    if include_rmse:
        assert true_values is not None, \
            'Need to input true values for RMSE!'
        rmse, rmse_unc = rmse_and_unc(df_in.values, true_values)
        df.loc[('rmse', 'value'), :] = rmse
        df.loc[('rmse', 'uncertainty'), :] = rmse_unc
    # Ensure correct row order by sorting
    df.sort_index(inplace=True)
    # Cast calclulation type index back from categorical to string to allow
    # adding new calculation types
    df.set_index(
        [df.index.get_level_values('calculation type').astype(str),
         df.index.get_level_values('result type')],
        inplace=True)
    return df


def efficiency_gain_df(method_names, method_values, est_names, **kwargs):
    """
    Calculated data frame showing

    efficiency gain ~ [(st dev standard) / (st dev new method)] ** 2

    The standard method on which to base the gain is assumed to be the first
    method input.

    Parameters
    ----------
    method names: list of strs
    method values: list
        Each element is a list of 1d arrays of results for the method. Each
        array must have shape (len(est_names),).
    est_names: list of strs
        Provide column titles for output df.
    true_values: iterable of same length as estimators list
        True values of the estimators for the given likelihood and prior.

    Returns
    -------
    results: pandas data frame
        results data frame.
        Contains rows:
            mean [dynamic goal]: mean calculation result for standard nested
                sampling and dynamic nested sampling with each input dynamic
                goal.
            std [dynamic goal]: standard deviation of results for standard
                nested sampling and dynamic nested sampling with each input
                dynamic goal.
            gain [dynamic goal]: the efficiency gain (computational speedup)
                from dynamic nested sampling compared to standard nested
                sampling. This equals (variance of standard results) /
                (variance of dynamic results); see the dynamic nested
                sampling paper for more details.
    """
    true_values = kwargs.pop('true_values', None)
    include_true_values = kwargs.pop('include_true_values', False)
    include_rmse = kwargs.pop('include_rmse', False)
    adjust_nsamp = kwargs.pop('adjust_nsamp', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if adjust_nsamp is not None:
        assert adjust_nsamp.shape == (len(method_names),)
    assert len(method_names) == len(method_values)
    df_dict = {}
    for i, method_name in enumerate(method_names):
        # Set include_true_values=False as we don't want them repeated for
        # every method
        df = summary_df_from_list(method_values[i], est_names,
                                  true_values=true_values,
                                  include_true_values=False,
                                  include_rmse=include_rmse)
        if i != 0:
            stats = ['std']
            if include_rmse:
                stats.append('rmse')
            for stat in stats:
                # Calculate efficiency gain vs standard ns
                ratio = (df_dict[method_names[0]].loc[(stat, 'value')]
                         / df.loc[(stat, 'value')])
                ratio_unc = array_ratio_std(
                    df_dict[method_names[0]].loc[(stat, 'value')],
                    df_dict[method_names[0]].loc[(stat, 'uncertainty')],
                    df.loc[(stat, 'value')],
                    df.loc[(stat, 'uncertainty')])
                key = stat + ' efficiency gain'
                df.loc[(key, 'value'), :] = ratio ** 2
                df.loc[(key, 'uncertainty'), :] = 2 * ratio * ratio_unc
                if adjust_nsamp is not None:
                    # Efficiency gain meansures performance per number of
                    # samples (proportional to computational work). If the
                    # number of samples is not the same we can adjust this.
                    adjust = (adjust_nsamp[0] / adjust_nsamp[i])
                    df.loc[(key, 'value'), :] *= adjust
                    df.loc[(key, 'uncertainty'), :] *= adjust
        df_dict[method_name] = df
    results = pd.concat(df_dict)
    results.index.rename('dynamic settings', level=0, inplace=True)
    new_ind = []
    new_ind.append(pd.CategoricalIndex(
        results.index.get_level_values('calculation type'), ordered=True,
        categories=['true values', 'mean', 'std', 'rmse',
                    'std efficiency gain', 'rmse efficiency gain']))
    new_ind.append(pd.CategoricalIndex(
        results.index.get_level_values('dynamic settings'),
        ordered=True, categories=[''] + method_names))
    new_ind.append(results.index.get_level_values('result type'))
    results.set_index(new_ind, inplace=True)
    if include_true_values:
        with warnings.catch_warnings():
            # Performance not an issue here so suppress annoying warning
            warnings.filterwarnings('ignore', message=(
                'indexing past lexsort depth may impact performance.'))
            results.loc[('true values', '', 'value'), :] = true_values
    results.sort_index(inplace=True)
    return results


def paper_format_efficiency_gain_df(eff_gain_df):
    """
    Transform efficiency gain data frames output by nestcheck into the format
    used in the dns paper.
    """
    paper_df = copy.deepcopy(
        eff_gain_df.loc[pd.IndexSlice[['std', 'std efficiency gain'], :, :], :])
    # Show mean number of samples and likelihood calls instead of st dev
    means = (eff_gain_df.xs('mean', level='calculation type')
             .xs('value', level='result type'))
    for col in ['samples', 'likelihood calls']:
        try:
            col_vals = []
            for val in means[col].values:
                col_vals += [int(np.rint(val)), np.nan]
            col_vals += [np.nan] * (paper_df.shape[0] - len(col_vals))
            paper_df[col] = col_vals
        except KeyError:
            pass
    row_name_map = {'std efficiency gain': 'Efficiency gain',
                    'St.Dev. efficiency gain': 'Efficiency gain',
                    'dynamic ': '',
                    'std': 'St.Dev.'}
    row_names = (paper_df.index.get_level_values(0).astype(str) + ' ' +
                 paper_df.index.get_level_values(1).astype(str))
    for key, value in row_name_map.items():
        row_names = row_names.str.replace(key, value)
    paper_df.index = [row_names, paper_df.index.get_level_values(2)]
    return paper_df


def rmse_and_unc(values_array, true_values):
    """
    Calculate the root meet squared error and its numerical uncertainty.
    """
    assert true_values.shape == (values_array.shape[1],)
    errors = values_array - true_values[np.newaxis, :]
    sq_errors = errors ** 2
    sq_errors_mean = np.mean(sq_errors, axis=0)
    sq_errors_mean_unc = (np.std(sq_errors, axis=0, ddof=1) /
                          np.sqrt(sq_errors.shape[0]))
    # With a reasonably large number of values in values_list the uncertainty
    # on sq_errors should be approximately normal (from the central limit
    # theorem).
    # Use error propogation: if sigma is the error on X then the error on X^0.5
    # is (X^0.5 / X) * 0.5 * sigma = 0.5 * (X^-0.5) * sigma
    rmse = np.sqrt(sq_errors_mean)
    rmse_unc = 0.5 * (1 / rmse) * sq_errors_mean_unc
    return rmse, rmse_unc


def array_ratio_std(values_n, sigmas_n, values_d, sigmas_d):
    """
    Gives error on the ratio of 2 floats or 2 1dimensional arrays given their
    values and errors assuming the errors are uncorrelated.
    This assumes covariance = 0. _n and _d denote the numerator and
    denominator.
    """
    return (values_n / values_d) * (((sigmas_n / values_n) ** 2 +
                                     (sigmas_d / values_d) ** 2)) ** 0.5
