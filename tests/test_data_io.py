#!/usr/bin/env python
# pylint: disable=unexpected-keyword-arg,too-few-public-methods
"""
Test suite for data processing, dummy data and input/output functions.
"""
import os
import sys
import shutil
import unittest
import warnings
import numpy as np
import numpy.testing
import nestcheck.data_processing
import nestcheck.diagnostics_tables
import nestcheck.dummy_data
import nestcheck.error_analysis
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


class TestDataProcessing(unittest.TestCase):

    """Tests for data_processing.py"""

    def setUp(self):
        """Make a temporary directory for saving test results."""
        try:
            os.makedirs(TEST_CACHE_DIR)
        except FileExistsError:
            pass

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except OSError:
            pass

    def test_batch_process_data_unexpected_kwarg(self):
        """Test unexpected kwargs checks."""
        self.assertRaises(
            TypeError, nestcheck.data_processing.batch_process_data,
            ['path'], base_dir=TEST_CACHE_DIR, unexpected=1)

    def test_get_birth_inds_unexpected_kwarg(self):
        """Test unexpected kwargs checks."""
        self.assertRaises(
            TypeError, nestcheck.data_processing.get_birth_inds,
            'birth_logl', 'logl', unexpected=1)

    def test_get_birth_inds(self):
        """Check birth inds allocation function."""
        # Check get_birth_inds works when points born and dying on same contour
        logl = np.asarray([1, 1, 3, 5])
        birth_logl = np.asarray([-1, 1, 1, 3])
        inds = nestcheck.data_processing.get_birth_inds(birth_logl, logl)
        numpy.testing.assert_array_equal(inds, np.asarray([-1, 0, 1, 2]))
        # Check error handeling of PolyChord v1.13 bug with more births than
        # deaths on a contour
        logl = np.asarray([1, 1, 2, 3, 5])
        birth_logl = np.asarray([-1, 1, 1, 1, 3])
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            inds = nestcheck.data_processing.get_birth_inds(birth_logl, logl)
            print(inds)
            self.assertEqual(len(war), 1)
        numpy.testing.assert_array_equal(inds, np.asarray([-1, 0, 0, 1, 3]))

    def test_check_ns_run_logls(self):
        """Ensure check_ns_run_logls raises error if and only if
        warn_only=False"""
        repeat_logl_run = {'logl': np.asarray([0, 0, 1])}
        self.assertRaises(
            AssertionError, nestcheck.data_processing.check_ns_run_logls,
            repeat_logl_run, dup_assert=True)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            nestcheck.data_processing.check_ns_run_logls(
                repeat_logl_run, dup_warn=True)
            self.assertEqual(len(war), 1)

    def test_process_polychord_data(self):
        """Check processing some dummy PolyChord data."""
        file_root = 'dummy_run'
        run = nestcheck.dummy_data.get_dummy_dynamic_run(
            10, seed=0, nthread_init=2, nthread_dyn=3)
        dead = nestcheck.write_polychord_output.run_dead_birth_array(run)
        np.savetxt(os.path.join(
            TEST_CACHE_DIR, file_root + '_dead-birth.txt'), dead)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            processed_run = nestcheck.data_processing.process_polychord_run(
                file_root, TEST_CACHE_DIR)
            self.assertEqual(len(war), 1)
        nestcheck.data_processing.check_ns_run(processed_run)
        for key, value in processed_run.items():
            if key not in ['output']:
                numpy.testing.assert_array_equal(
                    value, run[key], err_msg=key + ' not the same')
        self.assertEqual(processed_run['output']['file_root'], file_root)
        self.assertEqual(processed_run['output']['base_dir'], TEST_CACHE_DIR)

    def test_process_polychord_stats_file(self):
        """Check reading in PolyChord's <root>.stats file by making and saving
        a dummy one."""
        file_root = 'temp'
        output = nestcheck.write_polychord_output.write_stats_file(
            {'file_root': file_root, 'base_dir': TEST_CACHE_DIR})
        self.assertEqual(nestcheck.data_processing.process_polychord_stats(
            file_root, TEST_CACHE_DIR), output)

    def test_process_multinest_data(self):
        """Check processing some dummy MultiNest data."""
        file_root = 'dummy_run'
        run = nestcheck.dummy_data.get_dummy_run(5, 10, seed=False)
        samples = nestcheck.write_polychord_output.run_dead_birth_array(run)
        # Replicate MultiNest's dead and live points files, including their
        # extra columns
        dead = samples[:-2, :]
        live = samples[-2:, :]
        dead = np.hstack((dead, np.zeros((dead.shape[0], 2))))
        live = np.hstack((live, np.zeros((live.shape[0], 1))))
        np.savetxt(os.path.join(
            TEST_CACHE_DIR, file_root + '-dead-birth.txt'), dead)
        np.savetxt(os.path.join(
            TEST_CACHE_DIR, file_root + '-phys_live-birth.txt'), live)
        processed_run = nestcheck.data_processing.process_multinest_run(
            file_root, TEST_CACHE_DIR)
        nestcheck.data_processing.check_ns_run(processed_run)
        for key, value in processed_run.items():
            if key not in ['output']:
                numpy.testing.assert_array_equal(
                    value, run[key], err_msg=key + ' not the same')
        self.assertEqual(processed_run['output']['file_root'], file_root)
        self.assertEqual(processed_run['output']['base_dir'], TEST_CACHE_DIR)

    def test_batch_process_data(self):
        """Test processing some dummy PolyChord data using
        batch_process_data."""
        file_root = 'dummy_run'
        run = nestcheck.dummy_data.get_dummy_dynamic_run(
            10, seed=0, nthread_init=2, nthread_dyn=3)
        dead = nestcheck.write_polychord_output.run_dead_birth_array(run)
        np.savetxt(os.path.join(
            TEST_CACHE_DIR, file_root + '_dead-birth.txt'), dead)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            run_list = nestcheck.data_processing.batch_process_data(
                [file_root, 'an_empty_path'], base_dir=TEST_CACHE_DIR,
                parallel=False, errors_to_handle=(OSError, IOError))
            self.assertEqual(len(war), 2)
        self.assertEqual(len(run_list), 1)

    def test_process_dynesty_run(self):
        """Test processing dynesty results into nestcheck format."""

        class DynestyResults(object):

            """A dummy dynesty results object for testing."""

            def __init__(self, run, dynamic=False):
                """Initialse dynesty-format attributes corresponding to the
                input run."""
                self.samples = run['theta']
                self.samples_id = run['thread_labels']
                self.logl = run['logl']
                if not dynamic:
                    assert np.all(run['thread_min_max'][:, 0] == -np.inf)
                    self.nlive = run['thread_min_max'].shape[0]
                else:
                    # Treat every thread as a seperate batch
                    self.batch_bounds = run['thread_min_max']
                    self.batch_nlive = np.full(
                        run['thread_min_max'].shape[0], 1)
                    self.samples_batch = run['thread_labels']

        run = nestcheck.dummy_data.get_dummy_run(1, 10)
        for dynamic in [True, False]:
            results = DynestyResults(run, dynamic=dynamic)
            processed = nestcheck.data_processing.process_dynesty_run(results)
            for key, value in run.items():
                if key not in ['output']:
                    numpy.testing.assert_array_equal(
                        value, processed[key],
                        err_msg=('{0} not the same. dynamic={1}'
                                 .format(key, dynamic)))

    def test_process_samples_array(self):
        """Check the handling of duplicate loglikelihood values."""
        # Make a samples array with some duplicate logl values
        samples = np.zeros((4, 3))
        samples[:, -1] = np.asarray([-1e30, -1e30, 1, 1])  # births
        samples[:, -2] = np.asarray([1, 1, 2, 3])  # logls
        # Should raise warning if dup_warn is True
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            nestcheck.data_processing.process_samples_array(
                samples, dup_warn=True)
            self.assertEqual(len(war), 1)
        # Should raise AssertionError if dup_assert is True
        self.assertRaises(
            AssertionError, nestcheck.data_processing.process_samples_array,
            samples, dup_assert=True)


class TestWritePolyChordOutput(unittest.TestCase):

    """Tests for write_polychord_output.py."""

    def setUp(self):
        """Make a temporary directory for saving test results."""
        try:
            os.makedirs(TEST_CACHE_DIR)
        except FileExistsError:
            pass

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except OSError:
            pass

    def test_write_run_output_unexpected_kwarg(self):
        """Check write_run_output raises TypeError with unexpected
        kwargs."""
        self.assertRaises(
            TypeError, nestcheck.write_polychord_output.write_run_output,
            {}, unexpected=1)

    def test_write_run_output(self):
        """Check writing PolyChord output files."""
        file_root = 'dummy_run'
        run = nestcheck.dummy_data.get_dummy_run(10, 10)
        run['output'] = {'file_root': file_root, 'base_dir': TEST_CACHE_DIR}
        # Run with and without equals=True and posterior=True to ensure full
        # coverage
        nestcheck.write_polychord_output.write_run_output(
            run, equals=True, posteriors=True)
        nestcheck.write_polychord_output.write_run_output(run)
        processed_run = nestcheck.data_processing.process_polychord_run(
            file_root, TEST_CACHE_DIR)
        self.assertEqual(set(run.keys()), set(processed_run.keys()))
        for key, value in processed_run.items():
            if key not in ['output']:
                numpy.testing.assert_allclose(
                    value, run[key], err_msg=key + ' not the same')
        self.assertEqual(processed_run['output']['file_root'], file_root)
        self.assertEqual(processed_run['output']['base_dir'], TEST_CACHE_DIR)


class TestDummyData(unittest.TestCase):

    """Tests for the dummy_data.py module."""

    def test_get_dummy_run_unexpected_kwarg(self):
        """Check get_dummy_run raises TypeError with unexpected
        kwargs."""
        self.assertRaises(
            TypeError, nestcheck.dummy_data.get_dummy_run,
            1, 2, unexpected=1)

    def test_get_dummy_thread_unexpected_kwarg(self):
        """Check get_dummy_thread raises TypeError with unexpected
        kwargs."""
        self.assertRaises(
            TypeError, nestcheck.dummy_data.get_dummy_thread,
            1, unexpected=1)

    def test_get_dummy_dynamic_run_unexpected_kwarg(self):
        """Check get_dummy_dynamic_run raises TypeError with unexpected
        kwargs."""
        self.assertRaises(
            TypeError, nestcheck.dummy_data.get_dummy_dynamic_run,
            1, unexpected=1)


class TestIOUtils(unittest.TestCase):

    """Tests for io_utils.py."""

    def setUp(self):
        """Get some data data for io testing.

        Note that the saving function in io_utils makes the specified
        directory if it does not already exist, so there is no need to
        make it in setUp.
        """
        self.test_data = np.random.random(10)

        @nestcheck.io_utils.save_load_result
        def io_func(data):
            """Helper for testing save and load functions via the
            io_utils.save_load_result decorator."""
            return data
        self.io_func = io_func

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except OSError:
            pass

    def test_save_load_wrapper(self):
        """Try saving and loading some test data and check it dosnt change."""
        # Without save_name (will neither save nor load)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            data_out = self.io_func(self.test_data, save=True, load=True)
            self.assertEqual(len(war), 2)
        self.assertTrue(np.array_equal(self.test_data, data_out))
        # Before any data saved (will save but not load)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            data_out = self.io_func(
                self.test_data, save=True, load=True, warn_if_error=True,
                save_name=TEST_CACHE_DIR + '/io_test')
            self.assertEqual(len(war), 1)
        self.assertTrue(np.array_equal(self.test_data, data_out))
        # After data saved (will load)
        data_out = self.io_func(self.test_data, save=True, load=True,
                                save_name=TEST_CACHE_DIR + '/io_test')
        self.assertTrue(np.array_equal(self.test_data, data_out))
        # Check handling of permission and memory errors when saving
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            nestcheck.io_utils.pickle_save(data_out, '//')
            self.assertEqual(len(war), 1)

    def test_load_filenotfound(self):
        """Test loading files which dont exist causes FileNotFoundError."""
        if sys.version_info[0] >= 3:
            self.assertRaises(
                FileNotFoundError, nestcheck.io_utils.pickle_load,
                TEST_CACHE_DIR + 'not_here')
        else:
            # FileNotFoundError not defined in python2 - use IOError instead
            self.assertRaises(
                IOError, nestcheck.io_utils.pickle_load,
                TEST_CACHE_DIR + 'not_here')

    def test_no_overwrite(self):
        """Check option to not overwrite existing files."""
        # Save our test data
        nestcheck.io_utils.pickle_save(
            self.test_data, TEST_CACHE_DIR + '/io_test', print_time=True)
        # Try saving some different data to same path
        nestcheck.io_utils.pickle_save(
            self.test_data - 100, TEST_CACHE_DIR + '/io_test',
            overwrite_existing=False)
        # Check the test data was not edited
        data_out = nestcheck.io_utils.pickle_load(TEST_CACHE_DIR + '/io_test')
        self.assertTrue(np.array_equal(self.test_data, data_out))

    def test_save_load_unexpected_kwargs(self):
        """Unexpected kwarg should throw exception."""
        self.assertRaises(
            TypeError, nestcheck.io_utils.pickle_load,
            self.test_data, TEST_CACHE_DIR + '/io_test', unexpected=1)
        self.assertRaises(
            TypeError, nestcheck.io_utils.pickle_save,
            self.test_data, TEST_CACHE_DIR + '/io_test', unexpected=1)
