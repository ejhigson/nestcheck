#!/usr/bin/env python
"""
Test the nestcheck module installation.
"""

import os
import shutil
import unittest
import numpy as np
import numpy.testing
import pandas as pd
import matplotlib
import nestcheck.estimators as e
import nestcheck.plots
import nestcheck.data_processing
import nestcheck.diagnostics as d
import nestcheck.io_utils


class TestNestCheck(unittest.TestCase):

    """Container for module tests."""

    def setUp(self):
        """
        Set up list of estimator objects and settings for each test.
        Use all the estimators in the module in each case, and choose settings
        so the tests run quickly.
        """
        self.cache_dir = 'cache_tests/'
        assert not os.path.exists(self.cache_dir[:-1]), \
            ('Directory ' + self.cache_dir[:-1] + ' exists! Tests use this ' +
             'dir to check caching then delete it afterwards, so the path ' +
             'should be left empty.')
        # Get some data
        self.standard_runs = nestcheck.data_processing.get_polychord_data(
            'v02_gaussian_standard_2d_10nlive_20nrepeats', 10,
            chains_dir='tests/data/', data_dir=self.cache_dir, save=True,
            load=True)
        self.dynamic_runs = nestcheck.data_processing.get_polychord_data(
            'v02_gaussian_standard_2d_10nlive_20nrepeats', 10,
            chains_dir='tests/data/', data_dir=self.cache_dir, save=True,
            load=True)
        self.estimator_list = [e.name_est(e.count_samples),
                               e.name_est(e.logz),
                               e.name_est(e.evidence),
                               e.name_est(e.param_mean, param_ind=0),
                               e.name_est(e.param_squared_mean),
                               e.name_est(e.param_cred, probability=0.5),
                               e.name_est(e.param_cred, probability=0.84),
                               e.name_est(e.r_mean),
                               e.name_est(e.r_cred, probability=0.5),
                               e.name_est(e.r_cred, probability=0.84)]
        self.ns_run = nestcheck.data_processing.get_polychord_data(
            'test_gaussian_standard_2d_10nlive_20nrepeats', 1,
            chains_dir='tests/data/', data_dir=self.cache_dir, save=True,
            load=True)

    def tearDown(self):
        # Remove any caches saved by the tests
        try:
            shutil.rmtree(self.cache_dir[:-1])
        except FileNotFoundError:
            pass

    def test_error_summary(self):
        standard_df = d.run_error_summary(d.analyse_run_errors(
            self.standard_runs, self.estimator_list, 100, thread_test=True,
            bs_test=True))
        standard_df.to_pickle('tests/data/standard_df_values.pkl')
        # Check the values of every row for the theta1 estimator
        test_values = pd.read_pickle('tests/data/standard_df_values.pkl')
        numpy.testing.assert_allclose(standard_df.values, test_values.values,
                                      rtol=1e-13)

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

    def test_io_utils(self):
        # Try saving and loading some test data and check it dosnt change
        data_in = np.random.random(10)
        nestcheck.io_utils.pickle_save(data_in, self.cache_dir + 'io_test',
                                       print_time=True)
        data_out = nestcheck.io_utils.pickle_load(self.cache_dir + 'io_test')
        self.assertTrue(np.array_equal(data_in, data_out))
        # Test loading files which dont exist
        self.assertRaises(FileNotFoundError, nestcheck.io_utils.pickle_load,
                          self.cache_dir + 'not_here')
        # Check saving files which already exist
        nestcheck.io_utils.pickle_save(data_in, self.cache_dir + 'io_test',
                                       overwrite_existing=False)
        # Test unexpected kwargs checks
        self.assertRaises(TypeError, nestcheck.io_utils.pickle_save,
                          data_in, self.cache_dir + 'io_test', unexpected=1)
        self.assertRaises(TypeError, nestcheck.io_utils.pickle_load,
                          data_in, self.cache_dir + 'io_test', unexpected=1)

    def test_data_processing(self):
        # Try looking for chains which dont exist
        data = nestcheck.data_processing.get_polychord_data(
            'an_empty_path', 1,
            chains_dir='tests/data/', data_dir=self.cache_dir, save=True,
            load=True)
        self.assertEqual(len(data), 0)
        # Test unexpected kwargs checks
        self.assertRaises(
            TypeError, nestcheck.data_processing.get_polychord_data,
            'test_gaussian_standard_2d_10nlive_20nrepeats', 1,
            chains_dir='tests/data/', unexpected=1)


if __name__ == '__main__':
    unittest.main()
