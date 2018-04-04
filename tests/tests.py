#!/usr/bin/env python
"""
Test the nestcheck module installation.
"""
import os
import sys
import shutil
import unittest
import functools
import importlib
import numpy as np
import numpy.testing
import pandas as pd
import pandas.testing
import matplotlib
import scipy.special
import nestcheck.parallel_utils
import nestcheck.io_utils
import nestcheck.estimators as e
import nestcheck.analyse_run as ar
import nestcheck.plots
import nestcheck.data_processing
import nestcheck.diagnostics
try:
    import PyPolyChord
    import PyPolyChord.priors
    from PyPolyChord.settings import PolyChordSettings

    def gaussian_likelihood(theta, sigma=1, n_derived=0):
        """ Simple Gaussian Likelihood centred on the origin."""
        phi = [0.0] * n_derived
        dim = len(theta)
        rad2 = sum([t ** 2 for t in theta])
        logl = -np.log(2 * np.pi * (sigma ** 2)) * dim / 2.0
        logl += -rad2 / (2 * sigma ** 2)
        return logl, phi

    def uniform_prior(hypercube, prior_scale=5):
        """Uniform prior."""
        ndims = len(hypercube)
        theta = [0.0] * ndims
        func = PyPolyChord.priors.UniformPrior(-prior_scale, prior_scale)
        for i, x in enumerate(hypercube):
            theta[i] = func(x)
        return theta

except ImportError:
    print('WARNING: could not import PyPolyChord. Install PyPolyChord to use '
          'full test suite.')


TEST_CACHE_DIR = 'cache_tests'
TEST_DIR_EXISTS_MSG = ('Directory ' + TEST_CACHE_DIR + ' exists! Tests use '
                       'this dir to check caching then delete it afterwards, '
                       'so the path should be left empty.')


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Make a directory for saving test results."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    # @unittest.skipIf('PyPolyChord' not in sys.modules,
    #                  'needs PyPolyChord to run')
    @unittest.skipIf(True, 'needs PyPolyChord to run')
    def test_polychord_processing(self):
        os.makedirs(TEST_CACHE_DIR)
        ndim = 2
        settings = PolyChordSettings(ndim, 0, base_dir=TEST_CACHE_DIR, seed=1,
                                     file_root='test', nlive=20, feedback=-1)
        PyPolyChord.run_polychord(gaussian_likelihood, 2, 0, settings,
                                  uniform_prior)
        run = nestcheck.data_processing.process_polychord_run(
            settings.file_root, base_dir=settings.base_dir)
        random_seed_msg = (
            'Your PolyChord install\'s random seed number generator is '
            'probably different to the one used to set the expected values.')
        self.assertEqual(run['logl'][0], -23.8600960993736,
                         msg=random_seed_msg)
        self.assertEqual(run['output']['nlike'], 12402)
        self.assertAlmostEqual(e.param_mean(run), -0.00019962261593814093, places=12)
        self.assertAlmostEqual(run['output']['logZ'], e.logz(run), places=1)

    def test_batch_process_data(self):
        # Try looking for chains which dont exist
        data = nestcheck.data_processing.batch_process_data(
            ['an_empty_path'], base_dir=TEST_CACHE_DIR)
        self.assertEqual(len(data), 0)
        # Test unexpected kwargs checks
        self.assertRaises(
            TypeError, nestcheck.data_processing.batch_process_data,
            ['path'], base_dir=TEST_CACHE_DIR, unexpected=1)


class TestIOUtils(unittest.TestCase):

    def setUp(self):
        """Make a directory and data for io testing."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        self.test_data = np.random.random(10)

        @nestcheck.io_utils.save_load_result
        def save_load_func(data):
            return data
        self.save_load_func = save_load_func

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_save_load_wrapper(self):
        """Try saving and loading some test data and check it dosnt change."""
        # Without save_name (will neither save nor load)
        data_out = self.save_load_func(self.test_data, save=True, load=True)
        self.assertTrue(np.array_equal(self.test_data, data_out))
        # Before any data saved (will save but not load)
        data_out = self.save_load_func(self.test_data, save=True, load=True,
                                       save_name=TEST_CACHE_DIR + '/io_test')
        self.assertTrue(np.array_equal(self.test_data, data_out))
        # After data saved (will load)
        data_out = self.save_load_func(self.test_data, save=True, load=True,
                                       save_name=TEST_CACHE_DIR + '/io_test')
        self.assertTrue(np.array_equal(self.test_data, data_out))

    def test_load_filenotfound(self):
        """Test loading files which dont exist causes FileNotFoundError."""
        self.assertRaises(FileNotFoundError, nestcheck.io_utils.pickle_load,
                          TEST_CACHE_DIR + 'not_here')

    def test_no_overwrite(self):
        """Check option to not overwrite existing files."""
        # Save our test data
        nestcheck.io_utils.pickle_save(self.test_data,
                                       TEST_CACHE_DIR + '/io_test',
                                       print_time=True)
        # Try saving some different data to same path
        nestcheck.io_utils.pickle_save(self.test_data - 100,
                                       TEST_CACHE_DIR + '/io_test',
                                       overwrite_existing=False)
        # Check the test data was not edited
        data_out = nestcheck.io_utils.pickle_load(TEST_CACHE_DIR + '/io_test')
        self.assertTrue(np.array_equal(self.test_data, data_out))

    def test_save_load_unexpected_kwargs(self):
        """Unexpected kwarg should throw exception."""
        self.assertRaises(TypeError, nestcheck.io_utils.pickle_load,
                          self.test_data, TEST_CACHE_DIR + '/io_test',
                          unexpected=1)
        self.assertRaises(TypeError, nestcheck.io_utils.pickle_save,
                          self.test_data, TEST_CACHE_DIR + '/io_test',
                          unexpected=1)


class TestPandasFunctions(unittest.TestCase):

    def setUp(self):
        self.nrows = 100
        self.ncols = 3
        self.data = np.random.random((self.nrows, self.ncols))
        self.col_names = ['est ' + str(i) for i in range(self.ncols)]
        self.df = pd.DataFrame(self.data, columns=self.col_names)
        self.sum_df = nestcheck.pandas_functions.summary_df(
            self.df, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True)

    def test_summary_df(self):
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
        data_list = [self.data[i, :] for i in range(self.nrows)]
        df = nestcheck.pandas_functions.summary_df_from_list(
            data_list, self.col_names, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True)
        pandas.testing.assert_frame_equal(df, self.sum_df)

    def test_summary_df_from_multi(self):
        multi = self.df
        multi['method'] = 'method 1'
        multi.set_index('method', drop=True, append=True, inplace=True)
        multi = multi.reorder_levels([1, 0])
        df = nestcheck.pandas_functions.summary_df_from_multi(
            multi, true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True)
        pandas.testing.assert_frame_equal(df.xs('method 1', level='method'),
                                          self.sum_df)

    def test_efficiency_gain_df(self):
        data_list = [self.data[i, :] for i in range(self.nrows)]
        method_names = ['old', 'new']
        adjust_nsamp = np.asarray([1, 2])
        method_values = [data_list] * len(method_names)
        df = nestcheck.pandas_functions.efficiency_gain_df(
            method_names, method_values, est_names=self.col_names,
            true_values=np.zeros(self.ncols),
            include_true_values=True, include_rmse=True,
            adjust_nsamp=adjust_nsamp)
        print(df)
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


class TestAnalyseRun(unittest.TestCase):

    def test_bootstrap_resample_run(self):
        run = get_dummy_ns_run(2, 1, 2)
        run['settings'] = {'ninit': 1}
        # With only 2 threads and ninit=1, separating initial threads means
        # that the resampled run can only contain each thread once
        resamp = ar.bootstrap_resample_run(run, ninit_sep=True)
        self.assertTrue(np.array_equal(run['theta'], resamp['theta']))
        # With random_seed=1 and 2 threads each with a single points,
        # bootstrap_resample_run selects the second thread twice.
        resamp = ar.bootstrap_resample_run(run, random_seed=1)
        self.assertTrue(np.array_equal(
            run['theta'][1, :], resamp['theta'][0, :]))
        self.assertTrue(np.array_equal(
            run['theta'][1, :], resamp['theta'][1, :]))
        # Check error handeled if no ninit
        del run['settings']
        resamp = ar.bootstrap_resample_run(run, ninit_sep=True)

    def test_rel_posterior_mass(self):
        self.assertTrue(np.array_equal(
            ar.rel_posterior_mass(np.asarray([0, 1]), np.asarray([1, 0])),
            np.asarray([1, 1])))

    def test_run_std_bootstrap(self):
        """Check bootstrap std is zero when the run only contains one
        thread."""
        run = get_dummy_ns_run(1, 10, 2)
        stds = ar.run_std_bootstrap(run, [e.param_mean], n_simulate=10)
        self.assertAlmostEqual(stds[0], 0, places=12)
        self.assertRaises(TypeError, ar.run_std_bootstrap, run,
                          [e.param_mean], n_simulate=10, unexpected=1)

    def test_run_ci_bootstrap(self):
        """Check bootstrap ci equals estimator expected value when the
        run only contains one thread."""
        run = get_dummy_ns_run(1, 10, 2)
        ci = ar.run_ci_bootstrap(run, [e.param_mean], n_simulate=10,
                                 cred_int=0.5)
        self.assertAlmostEqual(ci[0], e.param_mean(run), places=12)

    def test_run_std_simulate(self):
        """Check simulate std is zero when the run only contains one
        point."""
        run = get_dummy_ns_run(1, 1, 2)
        stds = ar.run_std_simulate(run, [e.param_mean], n_simulate=10)
        self.assertAlmostEqual(stds[0], 0, places=12)


class TestEstimators(unittest.TestCase):

    def setUp(self):
        self.nsamples = 10
        self.ns_run = get_dummy_ns_run(1, self.nsamples, 2)
        self.logw = ar.get_logw(self.ns_run)
        self.w_rel = np.exp(self.logw - self.logw.max())
        self.w_rel /= np.sum(self.w_rel)

    def test_count_samples(self):
        """Check count_samples estimator."""
        self.assertEqual(e.count_samples(self.ns_run), self.nsamples)

    def test_run_estimators(self):
        """Check ar.run_estimators wrapper is working."""
        out = ar.run_estimators(self.ns_run, [e.count_samples])
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
        self.assertAlmostEqual(e.param_mean(self.ns_run),
                               np.sum(self.w_rel * self.ns_run['theta'][:, 0]),
                               places=12)

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


class TestEstimatorLatexNames(unittest.TestCase):

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
        self.assertRaises(TypeError, e.get_latex_name, e.logz, unexpected=1)

    def test_latex_name_unknown_func(self):
        self.assertRaises(AssertionError, e.get_latex_name, np.mean)


class TestParallelUtils(unittest.TestCase):

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
        results_list = nestcheck.parallel_utils.parallel_apply(
            self.func, self.x, func_args=self.func_args,
            func_kwargs=self.func_kwargs, parallel=False)
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
        results_list = nestcheck.parallel_utils.parallel_map(
            self.func, self.x, func_pre_args=func_pre_args,
            func_kwargs=self.func_kwargs, parallel=False)
        res_arr = np.vstack(results_list)
        self.assertTrue(np.all(res_arr[:, 0] == func_pre_args[0]))
        self.assertTrue(np.all(res_arr[:, 2] == self.func_kwargs['kwarg']))
        # Don't need to sort as will be in order for map
        self.assertTrue(np.array_equal(res_arr[:, 1], np.asarray(self.x)))

    def test_parallel_map_parallelised(self):
        """Check parallel_map with parallel=True."""
        func_pre_args = self.func_args
        results_list = nestcheck.parallel_utils.parallel_map(
            self.func, self.x, func_pre_args=func_pre_args,
            func_kwargs=self.func_kwargs, parallel=True)
        res_arr = np.vstack(results_list)
        self.assertTrue(np.all(res_arr[:, 0] == func_pre_args[0]))
        self.assertTrue(np.all(res_arr[:, 2] == self.func_kwargs['kwarg']))
        # Don't need to sort as will be in order for map
        self.assertTrue(np.array_equal(res_arr[:, 1], np.asarray(self.x)))

    def test_parallel_map_unexpected_kwargs(self):
        """Unexpected kwarg should throw exception."""
        self.assertRaises(TypeError, nestcheck.parallel_utils.parallel_map,
                          self.func, self.x, unexpected=1)

class TestDiagnostics(unittest.TestCase):

    def test_run_list_error_summary(self):
        run_list = []
        for _ in range(10):
            run_list.append(get_dummy_ns_run(1, 10, 2))
        df = nestcheck.diagnostics.run_list_error_summary(
            run_list, [e.param_mean], ['param_mean'], 10, thread_pvalue=True,
            bs_stat_dist=True, cache_root='temp', save=False, load=True)
        self.assertTrue(np.all(~np.isnan(df.values)))
        # Uncomment below line to update values if they are deliberately
        # changed:
        df.to_pickle('tests/run_list_error_summary.pkl')
        # Check the values of every row for the theta1 estimator
        test_values = pd.read_pickle('tests/run_list_error_summary.pkl')
        numpy.testing.assert_allclose(df.values, test_values.values,
                                      rtol=1e-13, atol=1e-13)
        # print(df)

    def test_run_list_error_values_unexpected_kwarg(self):
        self.assertRaises(
            TypeError, nestcheck.diagnostics.run_list_error_values,
            [], [e.param_mean], ['param_mean'], 10, thread_pvalue=True,
            bs_stat_dist=True, save=True, load=True, unexpected=1)


class TestPlots(unittest.TestCase):

    def setUp(self):
        """Get some dummy data to plot."""
        self.ns_run = get_dummy_ns_run(10, 100, 2)
        nestcheck.data_processing.check_ns_run(self.ns_run,
                                               logl_warn_only=True)

    def test_plot_run_nlive(self):
        fig = nestcheck.plots.plot_run_nlive(
            ['standard'], {'standard': [self.ns_run] * 2})
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, nestcheck.plots.plot_run_nlive,
            ['standard'], {'standard': [self.ns_run] * 2}, unexpected=0)

    @unittest.skipIf(importlib.util.find_spec('fgivenx') is None,
                     'needs fgivenx to run')
    def test_param_logx_diagram(self):
        fig = nestcheck.plots.param_logx_diagram(
            self.ns_run, n_simulate=3, npoints=100)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, nestcheck.plots.param_logx_diagram,
            self.ns_run, unexpected=0)

    @unittest.skipIf(importlib.util.find_spec('fgivenx') is None,
                     'needs fgivenx to run')
    def test_bs_param_dists(self):
        fig = nestcheck.plots.bs_param_dists(
            self.ns_run, n_simulate=3, nx=10)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, nestcheck.plots.bs_param_dists,
            self.ns_run, unexpected=0)

    def test_kde_plot_df(self):
        bs_df = pd.DataFrame(index=['run_1', 'run_2'])
        bs_df['estimator_1'] = [np.random.random(10)] * 2
        bs_df['estimator_2'] = [np.random.random(10)] * 2
        fig = nestcheck.plots.kde_plot_df(bs_df)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, nestcheck.plots.kde_plot_df,
            unexpected=0)


# helper functions

def parallel_apply_func(x, arg, kwarg=-1):
    """A test function for checking parallel_apply."""
    return np.asarray([x, arg, kwarg])


def get_dummy_ns_run(nlive, nsamples, ndim, seed=False):
    """Generate template ns runs for quick testing without loading test
    data."""
    threads = []
    if seed is not False:
        np.random.seed(seed)
    for _ in range(nlive):
        thread = {'logl': np.zeros(nsamples),
                  'nlive_array': np.full(nsamples, 1.),
                  'theta': np.random.random((nsamples, ndim)),
                  'thread_labels': np.zeros(nsamples).astype(int)}
        thread['thread_min_max'] = np.asarray([[-np.inf, thread['logl'][-1]]])
        threads.append(thread)
    return ar.combine_ns_runs(threads)


if __name__ == '__main__':
    unittest.main()
