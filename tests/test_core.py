#!/usr/bin/env python
"""
Test suite for the core modules in the nestcheck package.
"""
import functools
import os
import sys
import copy
import unittest
import warnings
import numpy as np
import numpy.testing
import pandas as pd
import pandas.testing
import scipy.special
import nestcheck.data_processing
import nestcheck.diagnostics_tables
import nestcheck.dummy_data
import nestcheck.error_analysis
import nestcheck.estimators as e
import nestcheck.io_utils
import nestcheck.ns_run_utils
import nestcheck.parallel_utils
import nestcheck.plots
import nestcheck.write_polychord_output


# Define a directory to output files produced by tests (this will be deleted
# when the tests finish).
TEST_CACHE_DIR = 'temp_test_data_to_delete'


def setUpModule():
    """Before running the test suite, check that TEST_CACHE_DIR does not
    already exist - as the tests will delete it."""
    assert not os.path.exists(TEST_CACHE_DIR), (
        'Directory ' + TEST_CACHE_DIR + ' exists! Tests use this directory to '
        'check caching then delete it afterwards, so its path should be left '
        'empty. You should manually delete or move ' + TEST_CACHE_DIR
        + ' before running the tests.')


class TestPandasFunctions(unittest.TestCase):

    """Tests for pandas_functions.py"""

    def setUp(self):
        """Set up dummy data in a pandas DataFrame and make a summary data
        frame for testing."""
        self.nrows = 100
        self.ncols = 3
        self.data = np.random.random((self.nrows, self.ncols))
        self.col_names = ['samples']
        self.col_names += ['est ' + str(i) for i in range(self.ncols - 1)]
        self.df = pd.DataFrame(self.data, columns=self.col_names)
        self.sum_df = nestcheck.pandas_functions.summary_df(
            self.df, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True)

    def test_summary_df(self):
        """Check summary DataFrame has expected values."""
        self.assertEqual(self.sum_df.shape, (7, self.ncols))
        numpy.testing.assert_array_equal(
            self.sum_df.loc[('mean', 'value'), :].values,
            np.mean(self.data, axis=0))
        numpy.testing.assert_array_equal(
            self.sum_df.loc[('mean', 'uncertainty'), :].values,
            np.std(self.data, axis=0, ddof=1) / np.sqrt(self.nrows))
        numpy.testing.assert_array_equal(
            self.sum_df.loc[('std', 'value'), :].values,
            np.std(self.data, axis=0, ddof=1))
        numpy.testing.assert_array_equal(
            self.sum_df.loc[('rmse', 'value'), :].values,
            np.sqrt(np.mean(self.data ** 2, axis=0)))
        self.assertRaises(
            TypeError, nestcheck.pandas_functions.summary_df,
            self.df, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True, unexpected=1)

    def test_summary_df_from_array(self):
        """Check function making summary data frame from a numpy array."""
        df = nestcheck.pandas_functions.summary_df_from_array(
            self.data, self.col_names, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True)
        pandas.testing.assert_frame_equal(df, self.sum_df)
        # check axis argument
        df = nestcheck.pandas_functions.summary_df_from_array(
            self.data.T, self.col_names, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True, axis=1)
        pandas.testing.assert_frame_equal(df, self.sum_df)

    def test_summary_df_from_list(self):
        """Check function making summary data frame from a list."""
        data_list = [self.data[i, :] for i in range(self.nrows)]
        df = nestcheck.pandas_functions.summary_df_from_list(
            data_list, self.col_names, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True)
        pandas.testing.assert_frame_equal(df, self.sum_df)

    def test_summary_df_from_multi(self):
        """Check function making summary data frame from a MultiIndex
        DataFrame."""
        multi = self.df
        multi['method'] = 'method 1'
        multi.set_index('method', drop=True, append=True, inplace=True)
        multi = multi.reorder_levels([1, 0])
        df = nestcheck.pandas_functions.summary_df_from_multi(
            multi, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True)
        # NB true values are not assigned to method 1 in summary_df_from_multi
        # so need to select from self.sum_df without them (i.e. remove first
        # row)
        pandas.testing.assert_frame_equal(
            df.xs('method 1', level='method'), self.sum_df.iloc[1:, :])

    def test_efficiency_gain_df(self):
        """Check function for calculating efficiency gains from different
        methods."""
        data_list = [self.data[i, :] for i in range(self.nrows)]
        method_names = ['old', 'new']
        method_values = [data_list] * len(method_names)
        # Check gain without adjusting nsamp
        adjust_nsamp = None
        df = nestcheck.pandas_functions.efficiency_gain_df(
            method_names, method_values, est_names=self.col_names,
            true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True,
            adjust_nsamp=adjust_nsamp)
        for i, method in enumerate(method_names[1:]):
            gains = np.asarray([1] * self.ncols)
            for gain_type in ['rmse efficiency gain', 'std efficiency gain']:
                numpy.testing.assert_array_equal(
                    df.loc[(gain_type, method, 'value'), :].values, gains)
        # Test with nsamp adjustment
        adjust_nsamp = np.asarray([1, 2])
        df = nestcheck.pandas_functions.efficiency_gain_df(
            method_names, method_values, est_names=self.col_names,
            true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True,
            adjust_nsamp=adjust_nsamp)
        for i, method in enumerate(method_names[1:]):
            gains = np.asarray([adjust_nsamp[0] / adjust_nsamp[i + 1]] *
                               self.ncols)
            for gain_type in ['rmse efficiency gain', 'std efficiency gain']:
                numpy.testing.assert_array_equal(
                    df.loc[(gain_type, method, 'value'), :].values, gains)
        self.assertRaises(
            TypeError, nestcheck.pandas_functions.efficiency_gain_df,
            method_names, method_values, est_names=self.col_names,
            unexpected=1)
        # Use the efficiency gain df we just made to check
        # paper_format_efficiency_gain_df
        paper_df = (nestcheck.pandas_functions
                    .paper_format_efficiency_gain_df(df))
        cols = [col for col in self.col_names if col != 'samples']
        numpy.testing.assert_array_equal(
            paper_df[cols].values,
            df.loc[pd.IndexSlice[['std', 'std efficiency gain'], :, :],
                   cols].values)


class TestNSRunUtils(unittest.TestCase):

    """Tests for ns_run_utils.py."""

    def test_combine_threads(self):
        """Check combining threads when birth contours are not present or are
        duplicated."""
        nsamples = 5
        # Get two threads
        threads = [nestcheck.dummy_data.get_dummy_thread(nsamples, seed=0),
                   nestcheck.dummy_data.get_dummy_thread(nsamples, seed=False)]
        # Sort in order of final logl
        threads = sorted(threads, key=lambda run: run['logl'][-1])
        t1 = threads[0]
        t2 = threads[1]
        # Get another thread starting on the last point of t2 (meaning it will
        # not overlap with t1)
        t_no_overlap = nestcheck.dummy_data.get_dummy_thread(
            nsamples, seed=False, logl_start=t2['logl'][-1] + 1000)
        # combining with t1 should throw an assertion error as nlive drops to
        # zero in between the threads
        self.assertRaises(
            AssertionError, nestcheck.ns_run_utils.combine_threads,
            [t1, t_no_overlap], assert_birth_point=False)
        # Get another thread starting on the last point of t1 so it overlaps
        # with t2
        t3 = nestcheck.dummy_data.get_dummy_thread(
            nsamples, seed=False, logl_start=t1['logl'][-1])
        # When birth point not in run:
        # Should raise assertion error only if assert_birth_point = True
        nestcheck.ns_run_utils.combine_threads([t2, t3])
        self.assertRaises(
            AssertionError, nestcheck.ns_run_utils.combine_threads,
            [t2, t3], assert_birth_point=True)
        # When birth point in run once:
        # should work with assert_birth_point = True
        nestcheck.ns_run_utils.combine_threads(
            [t1, t2, t3], assert_birth_point=True)
        # When birth point in run twice:
        # Should raise assertion error only if assert_birth_point = True
        nestcheck.ns_run_utils.combine_threads([t1, t1, t2, t3])
        self.assertRaises(
            AssertionError, nestcheck.ns_run_utils.combine_threads,
            [t1, t1, t2, t3], assert_birth_point=True)

    def test_combine_runs(self):
        """Check combining runs is consistent with combining threads."""
        runs = [nestcheck.dummy_data.get_dummy_run(2, 10),
                nestcheck.dummy_data.get_dummy_run(2, 10)]
        # Add ndead key to output to check combining them
        for i, _ in enumerate(runs):
            runs[i]['output'] = {'ndead': runs[i]['logl'].shape[0]}
        comb = nestcheck.ns_run_utils.combine_ns_runs(runs)
        # Check vs threads
        all_threads = []
        counter = 0
        for run in runs:
            threads = nestcheck.ns_run_utils.get_run_threads(run)
            threads = sorted(threads, key=lambda th: th['logl'][0])
            for i, _ in enumerate(threads):
                threads[i]['thread_labels'] = np.full(
                    threads[i]['thread_labels'].shape, counter)
                counter += 1
            all_threads += threads
        comb_th = nestcheck.ns_run_utils.combine_threads(all_threads)
        for key, value in comb.items():
            if key not in ['output']:
                self.assertTrue(key in comb_th)
                numpy.testing.assert_array_equal(
                    value, comb_th[key], err_msg=key + ' not the same')
        # Check lists of runs of length 1 are returned unchanged (including
        # non-additive properties in run['output'])
        run_in = nestcheck.dummy_data.get_dummy_run(2, 10)
        run_in['output'] = {'some key': 'its value'}
        run_out = nestcheck.ns_run_utils.combine_ns_runs([run_in])
        self.assertEqual(run_in['output'], run_out['output'])

    def test_get_logw(self):
        """Check IndexError raising"""
        self.assertRaises(IndexError, nestcheck.ns_run_utils.get_logw,
                          {'nlive_array': np.asarray(1.),
                           'logl': np.asarray([])})

    def test_dict_given_run_array(self):
        """Check dict_given_run_array when some threads start on logl.min()."""
        logl_start = -100.
        runs = [nestcheck.dummy_data.get_dummy_run(1, 10),
                nestcheck.dummy_data.get_dummy_run(1, 10)]
        for i, _ in enumerate(runs):
            runs[i]['logl'][0] = logl_start
        # Check for just one thread starting on logl_min
        run_in = copy.deepcopy(runs[0])
        run_in['thread_min_max'][:, 0] = logl_start
        run_out = nestcheck.ns_run_utils.dict_given_run_array(
            nestcheck.ns_run_utils.array_given_run(run_in),
            run_in['thread_min_max'])
        numpy.testing.assert_array_equal(
            run_in['nlive_array'], run_out['nlive_array'])
        # Check multiple threads
        run_in = nestcheck.ns_run_utils.combine_ns_runs(runs)
        run_in['thread_min_max'][:, 0] = logl_start
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            run_out = nestcheck.ns_run_utils.dict_given_run_array(
                nestcheck.ns_run_utils.array_given_run(run_in),
                run_in['thread_min_max'])
            self.assertEqual(len(war), 1)
        numpy.testing.assert_array_equal(
            run_in['nlive_array'], run_out['nlive_array'])

    def test_check_ns_run_logls(self):
        """Ensure check_ns_run_logls raises error if and only if
        warn_only=False"""
        repeat_logl_run = {'logl': np.asarray([0, 0, 1])}
        self.assertRaises(
            AssertionError, nestcheck.ns_run_utils.check_ns_run_logls,
            repeat_logl_run, dup_assert=True)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            nestcheck.ns_run_utils.check_ns_run_logls(
                repeat_logl_run, dup_warn=True)
            self.assertEqual(len(war), 1)


class TestErrorAnalysis(unittest.TestCase):

    """Tests for error_analysis.py"""

    def test_bootstrap_resample_run(self):
        """Check bootstrap resampling of nested sampling runs."""
        run = nestcheck.dummy_data.get_dummy_run(2, 1)
        run['settings'] = {'ninit': 1}
        # With only 2 threads and ninit=1, separating initial threads means
        # that the resampled run can only contain each thread once
        resamp = nestcheck.error_analysis.bootstrap_resample_run(
            run, ninit_sep=True)
        self.assertTrue(np.array_equal(run['theta'], resamp['theta']))
        # With random_seed=1 and 2 threads each with a single points,
        # bootstrap_resample_run selects the second thread twice.
        resamp = nestcheck.error_analysis.bootstrap_resample_run(
            run, random_seed=0)
        numpy.testing.assert_allclose(
            run['theta'][0, :], resamp['theta'][0, :])
        numpy.testing.assert_allclose(
            run['theta'][1, :], resamp['theta'][1, :])
        # Check error handeled if no ninit
        del run['settings']
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            resamp = nestcheck.error_analysis.bootstrap_resample_run(
                run, ninit_sep=True)
            self.assertEqual(len(war), 1)

    def test_run_std_bootstrap(self):
        """Check bootstrap std is zero when the run only contains one
        thread."""
        run = nestcheck.dummy_data.get_dummy_run(1, 10)
        stds = nestcheck.error_analysis.run_std_bootstrap(
            run, [e.param_mean], n_simulate=10)
        self.assertAlmostEqual(stds[0], 0, places=12)
        self.assertRaises(
            TypeError, nestcheck.error_analysis.run_std_bootstrap, run,
            [e.param_mean], n_simulate=10, unexpected=1)

    def test_run_ci_bootstrap(self):
        """Check bootstrap ci equals estimator expected value when the
        run only contains one thread."""
        run = nestcheck.dummy_data.get_dummy_run(1, 10)
        ci = nestcheck.error_analysis.run_ci_bootstrap(
            run, [e.param_mean], n_simulate=10, cred_int=0.5)
        self.assertAlmostEqual(ci[0], e.param_mean(run), places=12)

    def test_run_std_simulate(self):
        """Check simulate std is zero when the run only contains one
        point."""
        run = nestcheck.dummy_data.get_dummy_run(1, 1)
        stds = nestcheck.error_analysis.run_std_simulate(
            run, [e.param_mean], n_simulate=10)
        self.assertAlmostEqual(stds[0], 0, places=12)

    def test_pairwise_distances(self):
        """Check that when two sets of samples are the same, the KS pvalue is 1
        and the statistical distances are zero."""
        a = np.random.random(10)
        df = nestcheck.error_analysis.pairwise_distances([a, a])
        self.assertTrue(np.array_equal(df.values, np.asarray([1, 0, 0, 0])))

    def test_implementation_std(self):
        """Check implementation_std raises error with unexpected kwargs."""
        self.assertRaises(
            TypeError, nestcheck.error_analysis.implementation_std,
            np.random.random(4), np.random.random(4), np.random.random(4),
            np.random.random(4), unexpected=1)


class TestEstimators(unittest.TestCase):

    """Tests for estimators.py."""

    def setUp(self):
        """Set up a dummy run to test estimators on."""
        self.nsamples = 10
        self.ns_run = nestcheck.dummy_data.get_dummy_run(1, self.nsamples)
        self.logw = nestcheck.ns_run_utils.get_logw(self.ns_run)
        self.w_rel = np.exp(self.logw - self.logw.max())
        self.w_rel /= np.sum(self.w_rel)

    def test_count_samples(self):
        """Check count_samples estimator."""
        self.assertEqual(e.count_samples(self.ns_run), self.nsamples)

    def test_run_estimators(self):
        """Check nestcheck.ns_run_utils.run_estimators wrapper is working."""
        out = nestcheck.ns_run_utils.run_estimators(
            self.ns_run, [e.count_samples])
        self.assertEqual(out.shape, (1,))  # out should be np array
        self.assertEqual(out[0], self.nsamples)

    def test_logx(self):
        """Check logx estimator."""
        self.assertAlmostEqual(e.logz(self.ns_run),
                               scipy.special.logsumexp(self.logw), places=12)

    def test_evidence(self):
        """Check evidence estimator."""
        self.assertAlmostEqual(e.evidence(self.ns_run),
                               np.exp(scipy.special.logsumexp(self.logw)),
                               places=12)

    def test_param_mean(self):
        """Check param_mean estimator."""
        self.assertAlmostEqual(
            e.param_mean(self.ns_run),
            np.sum(self.w_rel * self.ns_run['theta'][:, 0]), places=12)
        ndim = self.ns_run['theta'].shape[1]
        self.assertRaises(
            IndexError, e.param_mean, self.ns_run, param_ind=ndim)
        self.assertTrue(np.isnan(
            e.param_mean(self.ns_run, param_ind=ndim, handle_indexerror=True)))

    def test_param_squared_mean(self):
        """ Check param_squared_mean estimator."""
        self.assertAlmostEqual(
            e.param_squared_mean(self.ns_run),
            np.sum(self.w_rel * (self.ns_run['theta'][:, 0] ** 2)),
            places=12)

    def test_r_mean(self):
        """Check r_mean estimator."""
        r = np.sqrt(self.ns_run['theta'][:, 0] ** 2 +
                    self.ns_run['theta'][:, 1] ** 2)
        self.assertAlmostEqual(e.r_mean(self.ns_run),
                               np.sum(self.w_rel * r), places=12)

    def test_param_cred(self):
        """Check param_cred estimator."""
        # Check results agree with np.median when samples are equally weighted
        self.assertAlmostEqual(
            e.param_cred(self.ns_run, logw=np.zeros(self.nsamples)),
            np.median(self.ns_run['theta'][:, 0]), places=12)
        # Check another probability while using weighted samples
        prob = 0.84
        self.assertAlmostEqual(
            e.param_cred(self.ns_run, probability=prob),
            e.weighted_quantile(prob, self.ns_run['theta'][:, 0], self.w_rel),
            places=12)

    def test_r_cred(self):
        """Check r_cred estimator."""
        r = np.sqrt(self.ns_run['theta'][:, 0] ** 2 +
                    self.ns_run['theta'][:, 1] ** 2)
        # Check results agree with np.median when samples are equally weighted
        self.assertAlmostEqual(
            e.r_cred(self.ns_run, logw=np.zeros(self.nsamples)), np.median(r),
            places=12)
        # Check another probability while using weighted samples
        prob = 0.84
        self.assertAlmostEqual(
            e.r_cred(self.ns_run, probability=prob),
            e.weighted_quantile(prob, r, self.w_rel),
            places=12)

    def test_outputs_unique_strings(self):
        """
        Check get_latex_names produces a unique string for each of a list of
        commonly used estimators.
        """
        estimator_list = [e.count_samples,
                          e.logz,
                          e.evidence,
                          e.param_mean,
                          functools.partial(e.param_mean, param_ind=1),
                          e.param_squared_mean,
                          functools.partial(e.param_cred, probability=0.5),
                          functools.partial(e.param_cred, probability=0.84),
                          e.r_mean,
                          functools.partial(e.r_cred, probability=0.5),
                          functools.partial(e.r_cred, probability=0.84)]
        estimator_names = [e.get_latex_name(est) for est in estimator_list]
        for name in estimator_names:
            self.assertIsInstance(name, str)
        # Check names are unique
        self.assertEqual(len(estimator_names), len(set(estimator_names)))

    def test_latex_name_unexpected_kwargs(self):
        """Check unexpected kwarg for get_latex_name raises expected error."""
        self.assertRaises(TypeError, e.get_latex_name, e.logz, unexpected=1)

    def test_latex_name_unknown_func(self):
        """Check unexpected func for get_latex_name raises expected error."""
        self.assertRaises(KeyError, e.get_latex_name, np.mean)


class TestParallelUtils(unittest.TestCase):

    """Tests for parallel_utils.py."""

    def setUp(self):
        """Define some variables."""
        self.x = list(range(5))
        self.func = parallel_apply_func
        self.func_args = (1,)
        self.func_kwargs = {'kwarg': 2}

    def test_parallel_apply_parallelised(self):
        """Check parallel_apply with parallel=True."""
        results_list = nestcheck.parallel_utils.parallel_apply(
            self.func, self.x, func_args=self.func_args,
            func_kwargs=self.func_kwargs, parallel=True)
        res_arr = np.vstack(results_list)
        self.assertTrue(np.all(res_arr[:, 1] == self.func_args[0]))
        self.assertTrue(np.all(res_arr[:, 2] == self.func_kwargs['kwarg']))
        # Need to sort results as may come back in any order
        self.assertTrue(np.array_equal(np.sort(res_arr[:, 0]),
                                       np.asarray(self.x)))

    def test_parallel_apply_not_parallelised(self):
        """Check parallel_apply with parallel=False."""
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            results_list = nestcheck.parallel_utils.parallel_apply(
                self.func, self.x, func_args=self.func_args,
                func_kwargs=self.func_kwargs, parallel=False)
            self.assertEqual(len(war), 1)
        res_arr = np.vstack(results_list)
        self.assertTrue(np.all(res_arr[:, 1] == self.func_args[0]))
        self.assertTrue(np.all(res_arr[:, 2] == self.func_kwargs['kwarg']))
        # Don't need to sort res_arr[:, 0] as will be in order when
        # parallel=False
        self.assertTrue(np.array_equal(res_arr[:, 0], np.asarray(self.x)))

    def test_parallel_apply_unexpected_kwargs(self):
        """Unexpected kwarg should throw exception."""
        self.assertRaises(TypeError, nestcheck.parallel_utils.parallel_apply,
                          self.func, self.x, func_args=self.func_args,
                          unexpected=1)

    def test_parallel_map_not_parallelised(self):
        """Check parallel_map with parallel=False."""
        func_pre_args = self.func_args
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            results_list = nestcheck.parallel_utils.parallel_map(
                self.func, self.x, func_pre_args=func_pre_args,
                func_kwargs=self.func_kwargs, parallel=False)
            self.assertEqual(len(war), 1)
        res_arr = np.vstack(results_list)
        self.assertTrue(np.all(res_arr[:, 0] == func_pre_args[0]))
        self.assertTrue(np.all(res_arr[:, 2] == self.func_kwargs['kwarg']))
        # Don't need to sort as will be in order for map
        self.assertTrue(np.array_equal(res_arr[:, 1], np.asarray(self.x)))

    def test_parallel_map_parallelised(self):
        """Check parallel_map with parallel=True."""
        func_pre_args = self.func_args
        try:
            results_list = nestcheck.parallel_utils.parallel_map(
                self.func, self.x, func_pre_args=func_pre_args,
                func_kwargs=self.func_kwargs, parallel=True)
            res_arr = np.vstack(results_list)
            self.assertTrue(np.all(res_arr[:, 0] == func_pre_args[0]))
            self.assertTrue(np.all(res_arr[:, 2] == self.func_kwargs['kwarg']))
            # Don't need to sort as will be in order for map
            self.assertTrue(np.array_equal(res_arr[:, 1], np.asarray(self.x)))
        except TypeError:
            # Chunksize argument only added to concurrent.futures.Executor map
            # function in python 3.5 so only raise if sys.version is >= 3.5
            if tuple(sys.version_info) >= (3, 5):
                raise

    def test_parallel_map_unexpected_kwargs(self):
        """Unexpected kwarg should throw exception."""
        self.assertRaises(TypeError, nestcheck.parallel_utils.parallel_map,
                          self.func, self.x, unexpected=1)


class TestDiagnosticsTables(unittest.TestCase):

    """Tests for diagnostics_tables.py."""

    def test_run_list_error_summary(self):
        """Test error df summary using numpy seeds."""
        run_list = []
        for i in range(5):
            run_list.append(nestcheck.dummy_data.get_dummy_run(
                5, 10, seed=i))
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            df = nestcheck.diagnostics_tables.run_list_error_summary(
                run_list, [e.param_mean], ['param_mean'], 10,
                thread_pvalue=True, bs_stat_dist=True, parallel=False,
                true_values=np.asarray([0.5]), include_rmse=True)
            self.assertEqual(len(war), 3)
        self.assertTrue(np.all(~np.isnan(df.values)))
        expected_vals = np.asarray([[5.09427108e-01],
                                    [5.09720232e-02],
                                    [1.13976909e-01],
                                    [4.02969225e-02],
                                    [1.02378997e-01],
                                    [2.55413864e-02],
                                    [6.29938278e-02],
                                    [1.57592691e-02],
                                    [9.49869116e-02],
                                    [6.17742874e-02],
                                    [8.33387330e-01],
                                    [3.77401247e+01],
                                    [8.07046256e-02],
                                    [4.63611578e-02],
                                    [7.88292797e-01],
                                    [3.95552239e+00],
                                    [5.45196361e-01],
                                    [1.10751031e-01],
                                    [7.30000000e-01],
                                    [7.31057073e-02],
                                    [3.93198952e-01],
                                    [6.35544039e-02],
                                    [1.49018643e-01],
                                    [2.81601546e-02]])
        # Tricky getting seeding consistent in python2 so only test exact
        # numbers in python3
        print(np.column_stack(
            [df.values, expected_vals, df.values - expected_vals]))
        assert False
        if tuple(sys.version_info) >= (3, 0):
            numpy.testing.assert_allclose(df.values, expected_vals,
                                          rtol=1e-6, atol=1e-6)

    def test_run_list_error_values_unexpected_kwarg(self):
        """Check unexpected kwarg raises expected error."""
        self.assertRaises(
            TypeError, nestcheck.diagnostics_tables.run_list_error_values,
            [], [e.param_mean], ['param_mean'], 10, thread_pvalue=True,
            bs_stat_dist=True, unexpected=1)

    def test_estimator_values_df_unexpected_kwarg(self):
        """Check unexpected kwarg raises expected error."""
        self.assertRaises(
            TypeError, nestcheck.diagnostics_tables.estimator_values_df,
            [], [e.param_mean], unexpected=1)


# Helper functions for tests
# --------------------------

def parallel_apply_func(x, arg, kwarg=-1):
    """A test function for checking parallel_apply.

    Must be defined at module level to make it picklable."""
    return np.asarray([x, arg, kwarg])


if 'nose' in sys.modules.keys():
    if __name__ == '__main__':
        unittest.main()
else:
    # If not being run with nose, use cProfile to profile tests
    try:
        import cProfile
        import pstats
        cProfile.run('unittest.main()', 'restats')
        PSTAT = pstats.Stats('restats')
        os.remove('restats')
        PSTAT.strip_dirs().sort_stats('cumtime').print_stats('tests.py', 20)
    except ImportError:
        unittest.main()
