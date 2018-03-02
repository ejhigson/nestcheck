#!/usr/bin/env python
"""
Test the nestcheck module installation.
"""

import os
import shutil
import unittest
import functools
import numpy as np
import numpy.testing
import pandas as pd
import matplotlib
import scipy.special
import nestcheck.parallel_utils
import nestcheck.io_utils
import nestcheck.estimators as e
import nestcheck.analyse_run as ar
import nestcheck.plots
import nestcheck.data_processing
import nestcheck.diagnostics as d


class TestIOUtils(unittest.TestCase):

    def setUp(self):
        """Make a directory and data for io testing."""
        self.cache_dir = 'cache_tests'
        assert not os.path.exists(self.cache_dir), (
            'Directory ' + self.cache_dir + ' exists! Tests use this '
            'dir to check caching then delete it afterwards, so the path '
            'should be left empty.')
        self.test_data = np.random.random(10)

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(self.cache_dir)
        except FileNotFoundError:
            pass

    def test_save_load(self):
        """Try saving and loading some test data and check it dosnt change."""
        nestcheck.io_utils.pickle_save(self.test_data,
                                       self.cache_dir + '/io_test',
                                       print_time=True)
        data_out = nestcheck.io_utils.pickle_load(self.cache_dir + '/io_test')
        self.assertTrue(np.array_equal(self.test_data, data_out))

    def test_load_filenotfound(self):
        """Test loading files which dont exist causes FileNotFoundError."""
        self.assertRaises(FileNotFoundError, nestcheck.io_utils.pickle_load,
                          self.cache_dir + 'not_here')

    def test_no_overwrite(self):
        """Check option to not overwrite existing files."""
        # Save our test data
        nestcheck.io_utils.pickle_save(self.test_data,
                                       self.cache_dir + '/io_test',
                                       print_time=True)
        # Try saving some different data to same path
        nestcheck.io_utils.pickle_save(self.test_data - 100,
                                       self.cache_dir + '/io_test',
                                       overwrite_existing=False)
        # Check the test data was not edited
        data_out = nestcheck.io_utils.pickle_load(self.cache_dir + '/io_test')
        self.assertTrue(np.array_equal(self.test_data, data_out))

    def test_save_load_unexpected_kwargs(self):
        """Unexpected kwarg should throw exception."""
        self.assertRaises(TypeError, nestcheck.io_utils.pickle_load,
                          self.test_data, self.cache_dir + '/io_test',
                          unexpected=1)
        self.assertRaises(TypeError, nestcheck.io_utils.pickle_save,
                          self.test_data, self.cache_dir + '/io_test',
                          unexpected=1)


class TestEstimators(unittest.TestCase):

    def setUp(self):
        self.nsamples = 10
        self.ns_run = {'logl': np.zeros(self.nsamples),
                       'nlive_array': np.full(self.nsamples, 1.),
                       'theta': np.random.random((self.nsamples, 2)),
                       'thread_labels': np.zeros(self.nsamples),
                       'thread_min_max': np.asarray([[-np.inf, 0]])}
        self.logw = ar.get_logw(self.ns_run)
        self.w_rel = np.exp(self.logw - self.logw.max())
        self.w_rel /= np.sum(self.w_rel)
        self.r = np.sqrt(self.ns_run['theta'][:, 0] ** 2 +
                         self.ns_run['theta'][:, 1] ** 2)

    def test_count_samples(self):
        self.assertEqual(e.count_samples(self.ns_run), self.nsamples)

    def test_logx(self):
        self.assertAlmostEqual(e.logz(self.ns_run),
                               scipy.special.logsumexp(self.logw), places=12)

    def test_evidence(self):
        self.assertAlmostEqual(e.evidence(self.ns_run),
                               np.exp(scipy.special.logsumexp(self.logw)),
                               places=12)

    def test_param_mean(self):
        self.assertAlmostEqual(e.param_mean(self.ns_run),
                               np.sum(self.w_rel * self.ns_run['theta'][:, 0]),
                               places=12)

    def test_param_squared_mean(self):
        self.assertAlmostEqual(
            e.param_squared_mean(self.ns_run),
            np.sum(self.w_rel * (self.ns_run['theta'][:, 0] ** 2)),
            places=12)

    def test_r_mean(self):
        self.assertAlmostEqual(e.r_mean(self.ns_run),
                               np.sum(self.w_rel * self.r), places=12)


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
            func_kwargs=self.func_kwargs, parallelise=True)
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
            func_kwargs=self.func_kwargs, parallelise=False)
        res_arr = np.vstack(results_list)
        self.assertTrue(np.all(res_arr[:, 1] == self.func_args[0]))
        self.assertTrue(np.all(res_arr[:, 2] == self.func_kwargs['kwarg']))
        # Don't need to sort res_arr[:, 0] as will be in order when
        # parallelise=False
        self.assertTrue(np.array_equal(res_arr[:, 0], np.asarray(self.x)))

    def test_parallel_apply_unexpected_kwargs(self):
        """Unexpected kwarg should throw exception."""
        self.assertRaises(TypeError, nestcheck.parallel_utils.parallel_apply,
                          self.func, self.x, func_args=self.func_args,
                          unexpected=1)


class TestNestCheck(unittest.TestCase):

    def test_data_processing(self):
        # Try looking for chains which dont exist
        data = nestcheck.data_processing.get_polychord_data(
            'an_empty_path', 1,
            chains_dir='tests/data/', data_dir='another_empty_path', save=True,
            load=True)
        self.assertEqual(len(data), 0)
        # Test unexpected kwargs checks
        self.assertRaises(
            TypeError, nestcheck.data_processing.get_polychord_data,
            'test_gaussian_standard_2d_10nlive_20nrepeats', 1,
            chains_dir='tests/data/', unexpected=1)
#        self.ns_run = nestcheck.data_processing.get_polychord_data(
#            'test_gaussian_standard_2d_10nlive_20nrepeats', 1,
#            chains_dir='tests/data/', data_dir=self.cache_dir + '/',
#            save=True, load=True)


class TestDiagnostics(unittest.TestCase):

    def setUp(self):
        """
        Set up list of estimator objects and settings for each test.
        Use all the estimators in the module in each case, and choose settings
        so the tests run quickly.
        """
        self.cache_dir = 'cache_tests'
        assert not os.path.exists(self.cache_dir), (
            'Directory ' + self.cache_dir + ' exists! Tests use this '
            'dir to check caching then delete it afterwards, so the path '
            'should be left empty.')
        # Get some data
        self.standard_runs = nestcheck.data_processing.get_polychord_data(
            'v02_gaussian_standard_2d_10nlive_20nrepeats', 10,
            chains_dir='tests/data/', data_dir=self.cache_dir + '/', save=True,
            load=True)
        self.dynamic_runs = nestcheck.data_processing.get_polychord_data(
            'v02_gaussian_standard_2d_10nlive_20nrepeats', 10,
            chains_dir='tests/data/', data_dir=self.cache_dir + '/', save=True,
            load=True)
        self.estimator_list = [e.count_samples,
                               e.logz,
                               e.evidence,
                               e.param_mean,
                               functools.partial(e.param_mean, param_ind=1),
                               e.param_squared_mean,
                               functools.partial(e.param_cred,
                                                 probability=0.5),
                               functools.partial(e.param_cred,
                                                 probability=0.84),
                               e.r_mean,
                               functools.partial(e.r_cred, probability=0.5),
                               functools.partial(e.r_cred, probability=0.84)]
        self.estimator_names = [e.get_latex_name(est) for est in
                                self.estimator_list]

    def tearDown(self):
        shutil.rmtree(self.cache_dir)

    def test_error_summary(self):
        standard_df = d.run_error_summary(d.analyse_run_errors(
            self.standard_runs, self.estimator_list, self.estimator_names,
            100, thread_test=True,
            bs_test=True))
        standard_df.to_pickle('tests/data/standard_df_values.pkl')
        # Check the values of every row for the theta1 estimator
        test_values = pd.read_pickle('tests/data/standard_df_values.pkl')
        numpy.testing.assert_allclose(standard_df.values, test_values.values,
                                      rtol=1e-13)


class TestPlots(unittest.TestCase):

    def setUp(self):
        """
        Set up list of estimator objects and settings for each test.
        Use all the estimators in the module in each case, and choose settings
        so the tests run quickly.
        """
        # Get some data
        self.standard_runs = nestcheck.data_processing.get_polychord_data(
            'v02_gaussian_standard_2d_10nlive_20nrepeats', 10,
            chains_dir='tests/data/', save=False, load=False)

    def test_param_logx_diagram(self):
        fig = nestcheck.plots.param_logx_diagram(
            self.standard_runs[0], n_simulate=3, npoints=10)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, nestcheck.plots.param_logx_diagram,
            self.standard_runs[0], unexpected=0)

    def test_bs_param_dists(self):
        fig = nestcheck.plots.bs_param_dists(
            self.standard_runs[0], n_simulate=3, nx=10)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, nestcheck.plots.bs_param_dists,
            self.standard_runs[0], unexpected=0)


# helper functions

def parallel_apply_func(x, arg, kwarg=-1):
    """A test function for checking parallel_apply."""
    return np.asarray([x, arg, kwarg])


if __name__ == '__main__':
    unittest.main()
