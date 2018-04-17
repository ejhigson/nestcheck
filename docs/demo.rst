.. _demo:

.. note:: This page can be downloaded as an IPython notebook `here <https://github.com/ejhigson/nestcheck/blob/master/examples/nestcheck_demo.ipynb>`_ and run interactively.

Quickstart demo
===============

This is a brief demonstratation of loading nested sampling run data,
performing evidence and parameter estimations, and running error
analysis diagnostic tests. The diagnostic tests and plots described in:

-  "Diagnostic Tests for Nested Sampling Calculations" (`Higson et al.
   2018 <https://arxiv.org/abs/TBC>`__);
-  "Sampling Errors in Nested Sampling Parameter Estimation" (`Higson et
   al. 2017 <https://doi.org/10.1214/17-BA1075>`__).

For detailed explanations of the diagnostic tests and plots see the
papers. For more details about the code and the other options, see the
`documentation <http://nestcheck.readthedocs.io>`__. The code to make
all the results and diagrams in the diagnostic tests paper (`Higson et
al. 2018 <https://arxiv.org/abs/TBC>`__) is available in the
```diagnostics_paper_code.ipynb`` <https://github.com/ejhigson/nestcheck/blob/master/examples/diagnostics_paper_code.ipynb>`__
notebook and provides more examples.

Loading nested sampling runs
----------------------------

``nestcheck`` currently has functions for loading nested sampling data
from `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`__
and `PolyChord <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`__,
and you can easily add your own function to load data from other
sources.

The examples use nested sampling runs from
`PolyChord <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`__ -
you can download the data used from https://github.com/ejhigson/URL_TBC.
This contains 40 runs, 10 each for the 2-dimensional Gaussian, Gaussian
shell, Rosenbrock likelihoods used in the diagnostic tests paper (see
`Higson et al. 2018 <https://arxiv.org/abs/TBC>`__ Section 3 for more
detials). Each uses a uniform prior on each parameter in
:math:`[-10, 10]`.

For example, a PolyChord run can be loaded as follows:

.. code:: ipython3

    import nestcheck.data_processing
    
    base_dir = 'polychord_chains'  # directory containing run (PolyChord's 'base_dir' setting)
    file_root = 'gaussian_2d_100nlive_5nrepeats_1'  # directory containing run (PolyChord's 'base_dir' setting)
    run = nestcheck.data_processing.process_polychord_run(file_root, base_dir)
    run.keys()

For more information about the dictionary format and keys ``nestcheck``
uses to store nested sampling runs, see the `API
documentation <url_tbc>`__.

Data from multiple runs can be loaded and processed together with
optional parallelisation:

.. code:: ipython3

    file_roots = ['gaussian_2d_100nlive_5nrepeats_' + str(i) for i in range(1, 11)]
    run_list = nestcheck.data_processing.batch_process_data(
        file_roots, base_dir=base_dir, parallel=True,
        process_func=nestcheck.data_processing.process_polychord_run)

Evidence and parameter estimation calculations from runs
--------------------------------------------------------

Posterior inferences can be easily made from nested sampling runs (see
``estimators.py`` for more example functions)

.. code:: ipython3

    import nestcheck.estimators as e
    
    print('The log evidence estimate from the first run is',
          e.logz(run_list[0]))
    print('The estimate of the mean of the first parameter from the first run is',
          e.param_mean(run_list[0], param_ind=0))

You can get a pandas data frame of estimates of a list of quantites for
a list of runs as follows:

.. code:: ipython3

    import nestcheck.diagnostics_tables
    
    estimator_list = [e.logz, e.param_mean, e.param_squared_mean, e.r_mean]
    estimator_names = [e.get_latex_name(est) for est in estimator_list]  # Use nestcheck's stored LaTeX format estimator names
    vals_df = nestcheck.diagnostics_tables.estimator_values_df(run_list, estimator_list, estimator_names=estimator_names)
    vals_df

Bootstrap sampling error estimates
----------------------------------

We can estimate the sampling error on nested sampling calculations using
the bootstrap approach (see `Higson et al.
2017 <https://doi.org/10.1214/17-BA1075>`__ Section 4 for more details).

.. code:: ipython3

    import pandas as pd
    import nestcheck.error_analysis
    
    bs_error_df = pd.DataFrame(columns=estimator_names)
    for i, run in enumerate(run_list):
        bs_error_df.loc[i] = nestcheck.error_analysis.run_std_bootstrap(run, estimator_list, n_simulate=100)
    print('Run boostrap error estimates:')
    bs_error_df

Diagrams of uncertainties on posterior distributions using bootstrap resamples
------------------------------------------------------------------------------

Bootstrap resamples of nested sampling runs can be used to plot
numerical uncertainties on whole posterior distributions (rather than
just scalare quantities) using ``nestcheck``'s ``bs_param_dists``
function. For a discussion of this type of diagram and its uses, see
(`Higson et al. 2017 <https://doi.org/10.1214/17-BA1075>`__ Section 4.1
and Figure 3).

.. code:: ipython3

    import nestcheck.plots
    %matplotlib inline
    
    fig = nestcheck.plots.bs_param_dists(run_list[:2])

Diagrams of samples in :math:`\log X`
-------------------------------------

The ``param_logx_diagram`` function plots diagrams of samples of the
type proposed in (`Higson et al.
2017 <https://doi.org/10.1214/17-BA1075>`__ Section 4.2).

.. code:: ipython3

    fig = nestcheck.plots.param_logx_diagram(run_list[0], logx_min=-15)

Calculating errors due to implementation-specific effects
---------------------------------------------------------

The part of the variation in results which cannot be explained by the
intrinsic stochasticity of the nested sampling algorithm and is due to
the nesteded sampling software failing to produce uncorrelated sampling
within iso-likelihood contours can be estimated using the method
described in Section 5 of (`Higson et al. 2018 <URL_TBC>`__).

.. code:: ipython3

    df = nestcheck.diagnostics_tables.run_list_error_summary(run_list, estimator_list, estimator_names, 100)
    df

The 2-dimensional Gaussian likelihood is unimodal and easy for PolyChord
to sample, so as expected we see that the standard deviation of the
result values is close to the mean bootstrap standard deviation.
Consequently the estimated errors due to impelementation-specific
effects are low.

Tests for implementation specific effects using only 2 nested sampling runs
---------------------------------------------------------------------------

The diagnostic tests for only two runs presented in Section 6 of
(`Higson et al. 2018 <URL_TBC>`__) can also be easily calculated:

.. code:: ipython3

    # perform error analysis on two runs
    error_vals_df = nestcheck.diagnostics_tables.run_list_error_values(
        run_list[:2], estimator_list, estimator_names, thread_pvalue=True, bs_stat_dist=True, n_simulate=100)
    # select only rows containing pairwise tests to output
    error_vals_df.loc[pd.IndexSlice[['thread ks pvalue', 'bootstrap ks distance'], :], :]

If more than two runs are provided, the diagnostics are calculated for
each pairwise combination.
