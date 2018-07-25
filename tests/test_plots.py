#!/usr/bin/env python
"""
Test suite for nestcheck/plots.py.
"""
import unittest
import matplotlib
import numpy as np
import numpy.testing
import pandas as pd
import scipy.special
import nestcheck.data_processing
import nestcheck.diagnostics_tables
import nestcheck.dummy_data
import nestcheck.error_analysis
import nestcheck.io_utils
import nestcheck.ns_run_utils
import nestcheck.parallel_utils
import nestcheck.plots
import nestcheck.write_polychord_output


class TestPlots(unittest.TestCase):

    """Tests for plots.py."""

    def setUp(self):
        """Get some dummy data to plot."""
        self.ns_run = nestcheck.dummy_data.get_dummy_run(3, 10)
        nestcheck.data_processing.check_ns_run(self.ns_run)

    def test_alternate_helper(self):
        """Check alternate_helper."""
        alt_samps = np.random.random(3)
        alt_samps[-1] = np.nan
        x = np.random.random()

        def temp_func(xarg, arg1, arg2):
            """Func for testing alternate_helper."""
            return sum((xarg, arg1, arg2))

        ans = nestcheck.plots.alternate_helper(
            x, alt_samps, func=temp_func)
        self.assertEqual(ans, sum((x, alt_samps[0], alt_samps[1])))

    def test_rel_posterior_mass(self):
        """Check calculation of run's posterior mass with a simple test case
        where the answer is known analytically."""
        self.assertTrue(np.array_equal(
            nestcheck.plots.rel_posterior_mass(
                np.asarray([0, 1]), np.asarray([1, 0])),
            np.asarray([1, 1])))

    def test_plot_run_nlive(self):
        """Check plots of run's number of live points as a function of
        log X."""
        fig = nestcheck.plots.plot_run_nlive(
            ['type 1'], {'type 1': [self.ns_run] * 2},
            logl_given_logx=lambda x: x, ymax=100)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Check unexpected kwarg raises type error
        self.assertRaises(
            TypeError, nestcheck.plots.plot_run_nlive,
            ['type 1'], {'type 1': [self.ns_run] * 2}, unexpected=0)
        # Check with logx_given_logl specified
        fig = nestcheck.plots.plot_run_nlive(
            ['type 1'], {'type 1': [self.ns_run] * 2},
            logx_given_logl=lambda x: x,
            logl_given_logx=lambda x: x,
            cum_post_mass_norm='str', post_mass_norm='str')
        # Using logx from run should raise type error (expects no logx key or
        # else a perfectns-style logx numpy array)
        self.ns_run['logx'] = None
        self.assertRaises(
            TypeError, nestcheck.plots.plot_run_nlive,
            ['type 1'], {'type 1': [self.ns_run] * 2})

    def test_weighted_1d_gaussian_kde(self):
        """Check 1d gaussian kde function gives same results as scipy function
        when the weights of the samples are equal."""
        x = np.linspace(-0.5, 1.5, 10)
        samps = np.random.random(40)
        weights = np.ones(samps.shape)
        # Should be same as non-weighted scipy version when weights equal
        numpy.testing.assert_allclose(
            scipy.stats.gaussian_kde(samps)(x),
            nestcheck.plots.weighted_1d_gaussian_kde(x, samps, weights))
        # If x not 1d, should be AssertionError
        self.assertRaises(
            AssertionError, nestcheck.plots.weighted_1d_gaussian_kde,
            np.atleast_2d(x), samps, weights)
        # If samps not 1d, should be AssertionError
        self.assertRaises(
            AssertionError, nestcheck.plots.weighted_1d_gaussian_kde,
            x, np.atleast_2d(samps), weights)
        # If weights not same shape as samps, should be AssertionError
        self.assertRaises(
            AssertionError, nestcheck.plots.weighted_1d_gaussian_kde,
            x, samps, weights[1:])

    def test_param_logx_diagram(self):
        """Check function for making plots of sample distributions in logX."""
        fig = nestcheck.plots.param_logx_diagram(
            self.ns_run, n_simulate=2, npoints=10, parallel=True, thin=0.5)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Unexpected kwarg
        self.assertRaises(
            TypeError, nestcheck.plots.param_logx_diagram,
            self.ns_run, unexpected=0)
        # ftheta not same length as labels
        self.assertRaises(
            AssertionError, nestcheck.plots.param_logx_diagram,
            self.ns_run, fthetas=[lambda x: x], labels=['a', 'b'],
            plot_means=False)

    def test_plot_bs_dists_unexpected_kwarg(self):
        """Check unexpected kwargs raise the expected error."""
        self.assertRaises(
            TypeError, nestcheck.plots.plot_bs_dists,
            self.ns_run, [], [], unexpected=0)

    def test_bs_param_dists(self):
        """Check plots of the variation of posterior distributions calculated
        from bootstrap resamples."""
        fig = nestcheck.plots.bs_param_dists(
            self.ns_run, n_simulate=2, nx=10,
            parallel=True)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Check unexpected kwargs
        self.assertRaises(
            TypeError, nestcheck.plots.bs_param_dists,
            self.ns_run, unexpected=0)

    def test_kde_plot_df(self):
        """Check 1dimensional kde plots."""
        df = pd.DataFrame(index=['run_1', 'run_2'])
        df['estimator_1'] = [np.random.random(10)] * 2
        df['estimator_2'] = [np.random.random(10)] * 2
        df['estimator_3'] = [np.random.random(10)] * 2
        df['estimator_4'] = [np.random.random(10)] * 2
        fig = nestcheck.plots.kde_plot_df(
            df, xlims={}, num_xticks=3, legend=True, normalize=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        fig = nestcheck.plots.kde_plot_df(df, nrows=2)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, nestcheck.plots.kde_plot_df, df,
            unexpected=0)
