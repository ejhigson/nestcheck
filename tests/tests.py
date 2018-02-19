#!/usr/bin/env python
"""
Test the perfectns module installation.
"""

import os
import shutil
import unittest
import numpy as np
# import perfectns.settings
# import perfectns.estimators as e
# import perfectns.cached_gaussian_prior
# import perfectns.likelihoods as likelihoods
# import perfectns.nested_sampling as ns
# import perfectns.results_tables as rt
# import perfectns.maths_functions
# import perfectns.priors as priors
# import perfectns.plots
import nestcheck.data_processing
import nestcheck.io_utils


class TestPerfectNS(unittest.TestCase):

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
        # Process chains
        data = nestcheck.data_processing.get_polychord_data(
            'test_gaussian_standard_2d_10nlive_20nrepeats', 1,
            chains_dir='tests/data/', data_dir=self.cache_dir, save=True,
            load=True)
        self.assertEqual(len(data), 1)
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
