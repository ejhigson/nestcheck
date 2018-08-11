#!/usr/bin/env python
"""
Useful transformations and operations on pandas DataFrames.
"""

import copy
import warnings
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
        Names for the output df's columns.
    axis: int, optional
        Axis on which to calculate summary statistics.

    Returns
    -------
    df: MultiIndex DataFrame
        See summary_df docstring for more details.
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
    results_list: list of 1d numpy arrays
        Must have same length as names.
    names: list of strs
        Names for the output df's columns.
    kwargs: dict, optional
        Keyword arguments to pass to summary_df.

    Returns
    -------
    df: MultiIndex DataFrame
        See summary_df docstring for more details.
    """
    for arr in results_list:
        assert arr.shape == (len(names),)
    df = pd.DataFrame(np.stack(results_list, axis=0))
    df.columns = names
    return summary_df(df, **kwargs)


def summary_df_from_multi(multi_in, inds_to_keep=None, **kwargs):
    """
    Apply summary_df to a multiindex while preserving some levels.

    Parameters
    ----------
    multi_in: multiindex pandas DataFrame
    inds_to_keep: None or list of strs, optional
        Index levels to preserve.
    kwargs: dict, optional
        Keyword arguments to pass to summary_df.

    Returns
    -------
    df: MultiIndex DataFrame
        See summary_df docstring for more details.
    """
    # Need to pop include true values and add separately at the end as
    # otherwise we get multiple true values added
    include_true_values = kwargs.pop('include_true_values', False)
    true_values = kwargs.get('true_values', None)
    if inds_to_keep is None:
        inds_to_keep = list(multi_in.index.names)[:-1]
    if 'calculation type' not in inds_to_keep:
        df = multi_in.groupby(inds_to_keep).apply(
            summary_df, include_true_values=False, **kwargs)
    else:
        # If there is already a level called 'calculation type' in multi,
        # summary_df will try making a second 'calculation type' index and (as
        # of pandas v0.23.0) throw an error. Avoid this by renaming.
        inds_to_keep = [lev if lev != 'calculation type' else
                        'calculation type temp' for lev in inds_to_keep]
        multi_temp = copy.deepcopy(multi_in)
        multi_temp.index.set_names(
            [lev if lev != 'calculation type' else 'calculation type temp' for
             lev in list(multi_temp.index.names)], inplace=True)
        df = multi_temp.groupby(inds_to_keep).apply(
            summary_df, include_true_values=False, **kwargs)
        # add the 'calculation type' values ('mean' and 'std') produced by
        # summary_df to the input calculation type names (now in level
        # 'calculation type temp')
        ind = (df.index.get_level_values('calculation type temp') + ' ' +
               df.index.get_level_values('calculation type'))
        order = list(df.index.names)
        order.remove('calculation type temp')
        df.index = df.index.droplevel(
            ['calculation type', 'calculation type temp'])
        df['calculation type'] = list(ind)
        df.set_index('calculation type', append=True, inplace=True)
        df = df.reorder_levels(order)
    if include_true_values:
        assert true_values is not None
        tv_ind = ['true values' if name == 'calculation type' else '' for
                  name in df.index.names[:-1]] + ['value']
        df.loc[tuple(tv_ind), :] = true_values
    return df


def summary_df(df_in, **kwargs):
    """
    Make a panda data frame of the mean and std devs of an array of results,
    including the uncertainties on the values.

    This is similar to pandas.DataFrame.describe but also includes estimates of
    the numerical uncertainties.

    The output DataFrame has multiindex levels:

    'calculation type':  mean and standard deviations of the data.
    'result type': value and uncertainty for each quantity.

    calculation type    result type        column_1     column_2      ...
    mean                value
    mean                uncertainty
    std                 value
    std                 uncertainty

    Parameters
    ----------
    df_in: pandas DataFrame
    true_values: array
        Analytical values if known for comparison with mean. Used to
        calculate root mean squared errors (RMSE).
    include_true_values: bool, optional
        Whether or not to include true values in the output DataFrame.
    include_rmse: bool, optional
        Whether or not to include root-mean-squared-errors in the output
        DataFrame.


    Returns
    -------
    df: MultiIndex DataFrame
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
    # Cast calculation type index back from categorical to string to allow
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

    See the dynamic nested sampling paper (Higson et al. 2017) for more
    details.

    The standard method on which to base the gain is assumed to be the first
    method input.

    The output DataFrame will contain rows:
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
        Results data frame.
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
        df = summary_df_from_list(
            method_values[i], est_names, true_values=true_values,
            include_true_values=False, include_rmse=include_rmse)
        if i != 0:
            stats = ['std']
            if include_rmse:
                stats.append('rmse')
            if adjust_nsamp is not None:
                # Efficiency gain measures performance per number of
                # samples (proportional to computational work). If the
                # number of samples is not the same we can adjust this.
                adjust = (adjust_nsamp[0] / adjust_nsamp[i])
            else:
                adjust = 1
            for stat in stats:
                # Calculate efficiency gain vs standard nested sampling
                gain, gain_unc = get_eff_gain(
                    df_dict[method_names[0]].loc[(stat, 'value')],
                    df_dict[method_names[0]].loc[(stat, 'uncertainty')],
                    df.loc[(stat, 'value')],
                    df.loc[(stat, 'uncertainty')], adjust=adjust)
                key = stat + ' efficiency gain'
                df.loc[(key, 'value'), :] = gain
                df.loc[(key, 'uncertainty'), :] = gain_unc
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
    shown in the dynamic nested sampling paper (Higson et al. 2017).

    Parameters
    ----------
    eff_gain_df: pandas DataFrame
        DataFrame of the from produced by efficiency_gain_df.

    Returns
    -------
    paper_df: pandas DataFrame
    """
    idxs = pd.IndexSlice[['std', 'std efficiency gain'], :, :]
    paper_df = copy.deepcopy(eff_gain_df.loc[idxs, :])
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


# Helper functions
# ----------------


def get_eff_gain(base_std, base_std_unc, meth_std, meth_std_unc, adjust=1):
    """Calculates efficiency gain:

    efficiency gain ~ [(st dev standard) / (st dev new method)] ** 2

    as well as an estimate of its uncertainty.

    Parameters
    ----------
    base_std: 1d numpy array
    base_std_unc: 1d numpy array
        Uncertainties on base_std.
    meth_std: 1d numpy array
    meth_std_unc: 1d numpy array
        Uncertainties on base_std.

    Returns
    -------
    gain: 1d numpy array
    gain_unc: 1d numpy array
        Uncertainties on gain.
    """
    ratio = base_std / meth_std
    ratio_unc = array_ratio_std(
        base_std, base_std_unc, meth_std, meth_std_unc)
    gain = ratio ** 2
    gain_unc = 2 * ratio * ratio_unc
    gain *= adjust
    gain_unc *= adjust
    return gain, gain_unc


def rmse_and_unc(values_array, true_values):
    """
    Calculate the root meet squared error and its numerical uncertainty.

    With a reasonably large number of values in values_list the uncertainty
    on sq_errors should be approximately normal (from the central limit
    theorem).
    Uncertainties are calculated via error propagation: if sigma is the error
    on X then the error on X^0.5
    is (X^0.5 / X) * 0.5 * sigma = 0.5 * (X^-0.5) * sigma

    Parameters
    ----------
    values_array: 2d numpy array
        Array of results: each row corresponds to a different estimate of the
        quantities considered.
    true_values: 1d numpy array
        Correct values for the quantities considered.

    Returns
    -------
    rmse: 1d numpy array
        Root-mean-squared-error for each quantity.
    rmse_unc: 1d numpy array
        Numerical uncertainties on each element of rmse.
    """
    assert true_values.shape == (values_array.shape[1],)
    errors = values_array - true_values[np.newaxis, :]
    sq_errors = errors ** 2
    sq_errors_mean = np.mean(sq_errors, axis=0)
    sq_errors_mean_unc = (np.std(sq_errors, axis=0, ddof=1) /
                          np.sqrt(sq_errors.shape[0]))
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
    std = np.sqrt((sigmas_n / values_n) ** 2 + (sigmas_d / values_d) ** 2)
    std *= (values_n / values_d)
    return std
