#!/usr/bin/env python
"""
Transformations for pandas data frames.
"""


import numpy as np
import pandas as pd


def summary_df_from_array(results_array, names, axis=0):
    """
    Make a panda data frame of the mean and std devs of an array of results,
    including the uncertainties on the values.

    This just converts the array to a DataFrame and calls summary_df on it.

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
    if axis == 0:
        df.columns = names
    elif axis == 1:
        df.index = names
    return summary_df(df, axis=axis)


def summary_df_from_list(results_list, names):
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
    return summary_df(df, axis=0)


def summary_df_from_multi(multi_in, inds_to_keep=None):
    """
    summary_df with option to preserve some indexes of a multiindex.
    """
    if inds_to_keep is None:
        inds_to_keep = list(multi_in.index.names)[:-1]
    df = multi_in.groupby(inds_to_keep).apply(summary_df)
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
    return df


def summary_df(df_in, axis=0):
    """
    Make a panda data frame of the mean and std devs of an array of results,
    including the uncertainties on the values.

    This is similar to pandas.DataFrame.describe but also includes estimates of
    the numerical uncertainties.

    Parameters
    ----------
    df_in: data frame
    axis: int, optional
        axis on which to calculate summary statistics

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
    # make the data frame
    df = pd.DataFrame([df_in.mean(axis=axis), df_in.std(axis=axis, ddof=1)],
                      index=['mean', 'std'])
    df.index.name = 'calculation type'
    # add uncertainties
    num_cals = df_in.shape[axis]
    mean_unc = df.loc['std'] / np.sqrt(num_cals)
    std_unc = df.loc['std'] * np.sqrt(1 / (2 * (num_cals - 1)))
    df['result type'] = pd.Categorical(['value', 'value'], ordered=True,
                                       categories=['value', 'uncertainty'])
    df.set_index(['result type'], drop=True, append=True, inplace=True)
    df.loc[('mean', 'uncertainty'), :] = mean_unc.values
    df.loc[('std', 'uncertainty'), :] = std_unc.values
    return df.sort_index()


# def add_ratio_row(df, ind_n, ind_d, row_name=None, shared_labels=None):
#     """
#     Given two row indexes, adds another row with their ratio.
#
#     Needs input data fram in format:
#
#         col1 col1_unc col2 col2_unc ...
#
#     row1
#     row2
#
#     with _unc denoting uncertainty on columns and all columns having an
#     uncertainty.
#     """
#     # divide cols into values and uncertainties
#     cols_in = list(df.columns)
#     if shared_labels is not None:
#         cols_in = [c for c in cols_in if c not in shared_labels]
#     cols_val = []
#     cols_unc = []
#     for c in cols_in:
#         if c not in shared_labels:
#             if c[-4:] == '_unc':
#                 cols_unc.append(c)
#             else:
#                 cols_val.append(c)
#     # check the format of the data frame is correct
#     assert len(cols_val) == len(cols_unc)
#     for i, _ in enumerate(cols_val):
#         assert cols_val[i] + '_unc' == cols_unc[i]
#     # get the ratio of the two rows as a series
#     ratio = df[cols_val].loc[ind_n] / df[cols_val].loc[ind_d]
#     unc_array = mf.array_ratio_std(df[cols_val].loc[ind_n].values,
#                                    df[cols_unc].loc[ind_n].values,
#                                    df[cols_val].loc[ind_d].values,
#                                    df[cols_unc].loc[ind_d].values)
#     for i, cu in enumerate(cols_unc):
#         ratio[cu] = unc_array[i]
#     if shared_labels is not None:
#         for i, sl in enumerate(shared_labels):
#             assert df[sl].loc[ind_n] == df[sl].loc[ind_d]
#             ratio[sl] = df[sl].loc[ind_d]
#     # add to data frame
#     if row_name is None:
#         row_name = ind_n + ' / ' + ind_d
#     df.loc[row_name] = ratio
#     return df
