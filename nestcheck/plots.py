#!/usr/bin/env python
"""
Functions for diagnostic plots of nested sampling runs.

Includes functions for plots described "Diagnostic tests for nested sampling
calculations" (Higson et al. 2018).
"""

import functools
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import scipy.stats
import fgivenx
import fgivenx.plot
import nestcheck.error_analysis
import nestcheck.ns_run_utils


def plot_run_nlive(method_names, run_dict, **kwargs):
    """
    Plot the allocations of live points as a function of logX for the input
    sets of nested sampling runs of the type used in the dynamic nested
    sampling paper (Higson et al. 2017).
    Plots also include analytically calculated distributions of relative
    posterior mass and relative posterior mass remaining.

    Parameters
    ----------
    method_names: list of strs
    run_dict: dict of lists of nested sampling runs.
        Keys of run_dict must be method_names.
    logx_given_logl: function, optional
        For mapping points' logl values to logx values.
        If not specified the logx coordinates for each run are estimated using
        its numbers of live points.
    logl_given_logx: function, optional
        For calculating the relative posterior mass and posterior mass
        remaining at each logx coordinate.
    logx_min: float, optional
        Lower limit of logx axis. If not specified this is set to the lowest
        logx reached by any of the runs.
    ymax: bool, optional
        Maximum value for plot's nlive axis (yaxis).
    npoints: int, optional
        Number of points to have in the fgivenx plot grids.
    figsize: tuple, optional
        Size of figure in inches.
    post_mass_norm: str or None, optional
        Specify method_name for runs use form normalising the analytic
        posterior mass curve. If None, all runs are used.
    cum_post_mass_norm: str or None, optional
        Specify method_name for runs use form normalising the analytic
        cumulative posterior mass remaining curve. If None, all runs are used.

    Returns
    -------
    fig: matplotlib figure
    """
    logx_given_logl = kwargs.pop('logx_given_logl', None)
    logl_given_logx = kwargs.pop('logl_given_logx', None)
    logx_min = kwargs.pop('logx_min', None)
    ymax = kwargs.pop('ymax', None)
    npoints = kwargs.pop('npoints', 100)
    figsize = kwargs.pop('figsize', (6.4, 2))
    post_mass_norm = kwargs.pop('post_mass_norm', None)
    cum_post_mass_norm = kwargs.pop('cum_post_mass_norm', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert set(method_names) == set(run_dict.keys()), (
        'input method names=' + str(method_names) + ' do not match run_dict '
        'keys=' + str(run_dict.keys()))
    # Plotting
    # --------
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Reserve colors for certain common method_names so they are always the
    # same regardless of method_name order for consistency in the paper.
    linecolor_dict = {'standard': colors[2],
                      'dynamic $G=0$': colors[8],
                      'dynamic $G=1$': colors[9]}
    ax.set_prop_cycle('color', [colors[i] for i in [4, 1, 6, 0, 3, 5, 7]])
    integrals_dict = {}
    logx_min_list = []
    for method_name in method_names:
        integrals = np.zeros(len(run_dict[method_name]))
        for nr, run in enumerate(run_dict[method_name]):
            if 'logx' in run:
                logx = run['logx']
            elif logx_given_logl is not None:
                logx = logx_given_logl(run['logl'])
            else:
                logx = nestcheck.ns_run_utils.get_logx(
                    run['nlive_array'], simulate=False)
            logx_min_list.append(logx[-1])
            logx[0] = 0  # to make lines extend all the way to the end
            if nr == 0:
                # Label the first line and store it so we can access its color
                try:
                    line, = ax.plot(logx, run['nlive_array'], linewidth=1,
                                    label=method_name,
                                    color=linecolor_dict[method_name])
                except KeyError:
                    line, = ax.plot(logx, run['nlive_array'], linewidth=1,
                                    label=method_name)
            else:
                # Set other lines to same color and don't add labels
                ax.plot(logx, run['nlive_array'], linewidth=1,
                        color=line.get_color())
            # for normalising analytic weight lines
            integrals[nr] = -np.trapz(run['nlive_array'], x=logx)
        integrals_dict[method_name] = integrals[np.isfinite(integrals)]
    # if not specified, set logx min to the lowest logx reached by a run
    if logx_min is None:
        logx_min = np.asarray(logx_min_list).min()
    if logl_given_logx is not None:
        # Plot analytic posterior mass and cumulative posterior mass
        logx_plot = np.linspace(logx_min, 0, npoints)
        logl = logl_given_logx(logx_plot)
        # Remove any NaNs
        logx_plot = logx_plot[np.where(~np.isnan(logl))[0]]
        logl = logl[np.where(~np.isnan(logl))[0]]
        w_an = rel_posterior_mass(logx_plot, logl)
        # Try normalising the analytic distribution of posterior mass to have
        # the same area under the curve as the runs with dynamic_goal=1 (the
        # ones which we want to compare to it). If they are not available just
        # normalise it to the average area under all the runs (which should be
        # about the same if they have the same number of samples).
        w_an *= average_by_key(integrals_dict, post_mass_norm)
        ax.plot(logx_plot, w_an,
                linewidth=2, label='relative posterior mass',
                linestyle=':', color='k')
        # plot cumulative posterior mass
        w_an_c = np.cumsum(w_an)
        w_an_c /= np.trapz(w_an_c, x=logx_plot)
        # Try normalising the cumulative distribution of posterior mass to have
        # the same area under the curve as the runs with dynamic_goal=0 (the
        # ones which we want to compare to it). If they are not available just
        # normalise it to the average area under all the runs (which should be
        # about the same if they have the same number of samples).
        w_an_c *= average_by_key(integrals_dict, cum_post_mass_norm)
        ax.plot(logx_plot, w_an_c, linewidth=2, linestyle='--', dashes=(2, 3),
                label='posterior mass remaining', color='darkblue')
    ax.set_ylabel('number of live points')
    ax.set_xlabel(r'$\log X $')
    # set limits
    if ymax is not None:
        ax.set_ylim([0, ymax])
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim([logx_min, 0])
    ax.legend()
    return fig


def kde_plot_df(df, xlims=None, **kwargs):
    """
    Plots kde estimates of distributions of samples in each cell of the input
    pandas DataFrame.

    There is one subplot for each dataframe column, and on each subplot there
    is one kde line.

    Parameters
    ----------
    df: pandas data frame
        Each cell must contain a 1d numpy array of samples.
    xlims: dict, optional
        Dictionary of xlimits - keys are column names and values are lists of
        length 2.
    num_xticks: int, optional
        Number of xticks on each subplot.
    figsize: tuple, optional
        Size of figure in inches.
    nrows: int, optional
        Number of rows of subplots.
    ncols: int, optional
        Number of columns of subplots.
    normalize: bool, optional
        If true, kde plots are normalized to have the same area under their
        curves. If False, their max value is set to 1.
    legend: bool, optional
        Should a legend be added?
    legend_kwargs: dict, optional
        Additional kwargs for legend.

    Returns
    -------
    fig: matplotlib figure
    """
    assert xlims is None or isinstance(xlims, dict)
    figsize = kwargs.pop('figsize', (6.4, 1.5))
    num_xticks = kwargs.pop('num_xticks', None)
    nrows = kwargs.pop('nrows', 1)
    ncols = kwargs.pop('ncols', int(np.ceil(len(df.columns) / nrows)))
    normalize = kwargs.pop('normalize', True)
    legend = kwargs.pop('legend', False)
    legend_kwargs = kwargs.pop('legend_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for nax, col in enumerate(df):
        if nrows == 1:
            ax = axes[nax]
        else:
            ax = axes[nax // ncols, nax % ncols]
        supmin = df[col].apply(np.min).min()
        supmax = df[col].apply(np.max).max()
        support = np.linspace(supmin - 0.1 * (supmax - supmin),
                              supmax + 0.1 * (supmax - supmin), 200)
        handles = []
        labels = []
        for name, samps in df[col].iteritems():
            pdf = scipy.stats.gaussian_kde(samps)(support)
            if not normalize:
                pdf /= pdf.max()
            handles.append(ax.plot(support, pdf, label=name)[0])
            labels.append(name)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        if xlims is not None:
            try:
                ax.set_xlim(xlims[col])
            except KeyError:
                pass
        ax.set_xlabel(col)
        if num_xticks is not None:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(
                nbins=num_xticks))
    if legend:
        fig.legend(handles, labels, **legend_kwargs)
    return fig


def bs_param_dists(run_list, **kwargs):
    """
    Creates posterior distributions and their bootstrap error functions for
    input runs and estimators.

    For a more detailed description and some example use cases, see "Diagnostic
    tests for nested sampling calculations" (Higson et al. 2018).

    Parameters
    ----------
    run_list: dict or list of dicts
        Nested sampling run(s) to plot.
    fthetas: list of functions, optional
        Quantities to plot. Each must map a 2d theta array to 1d ftheta array -
        i.e. map every sample's theta vector (every row) to a scalar quantity.
        E.g. use lambda x: x[:, 0] to plot the first parameter.
    labels: list of strs, optional
        Labels for each ftheta.
    ftheta_lims: list, optional
        Plot limits for each ftheta.
    n_simulate: int, optional
        Number of bootstrap replications to be used for the fgivenx
        distributions.
    random_seed: int, optional
        Seed to make sure results are consistent and fgivenx caching can be
        used.
    figsize: tuple, optional
        Matplotlib figsize in (inches).
    nx: int, optional
        Size of x-axis grid for fgivenx plots.
    ny: int, optional
        Size of y-axis grid for fgivenx plots.
    cache: str or None
        Root for fgivenx caching (no caching if None).
    parallel: bool, optional
        fgivenx parallel option.
    rasterize_contours: bool, optional
        fgivenx rasterize_contours option.
    tqdm_kwargs: dict, optional
        Keyword arguments to pass to the tqdm progress bar when it is used in
        fgivenx while plotting contours.

    Returns
    -------
    fig: matplotlib figure
    """
    fthetas = kwargs.pop('fthetas', [lambda theta: theta[:, 0],
                                     lambda theta: theta[:, 1]])
    labels = kwargs.pop('labels', [r'$\theta_' + str(i + 1) + '$' for i in
                                   range(len(fthetas))])
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    n_simulate = kwargs.pop('n_simulate', 100)
    random_seed = kwargs.pop('random_seed', 0)
    figsize = kwargs.pop('figsize', (6.4, 2))
    nx = kwargs.pop('nx', 100)
    ny = kwargs.pop('ny', nx)
    cache_in = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    rasterize_contours = kwargs.pop('rasterize_contours', True)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'disable': True})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Use random seed to make samples consistent and allow caching.
    # To avoid fixing seed use random_seed=None
    state = np.random.get_state()  # save initial random state
    np.random.seed(random_seed)
    if not isinstance(run_list, list):
        run_list = [run_list]
    assert len(labels) == len(fthetas), (
        'There should be the same number of axes and labels')
    width_ratios = [40] * len(fthetas) + [1] * len(run_list)
    fig, axes = plt.subplots(nrows=1, ncols=len(run_list) + len(fthetas),
                             gridspec_kw={'wspace': 0.1,
                                          'width_ratios': width_ratios},
                             figsize=figsize)
    colormaps = ['Reds_r', 'Blues_r', 'Greys_r', 'Greens_r', 'Oranges_r']
    mean_colors = ['darkred', 'darkblue', 'darkgrey', 'darkgreen',
                   'darkorange']
    # plot in reverse order so reds are final plot and always on top
    for nrun, run in reversed(list(enumerate(run_list))):
        try:
            cache = cache_in + '_' + str(nrun)
        except TypeError:
            cache = None
        # add bs distribution plots
        cbar = plot_bs_dists(run, fthetas, axes[:len(fthetas)],
                             parallel=parallel,
                             ftheta_lims=ftheta_lims, cache=cache,
                             n_simulate=n_simulate, nx=nx, ny=ny,
                             rasterize_contours=rasterize_contours,
                             mean_color=mean_colors[nrun],
                             colormap=colormaps[nrun],
                             tqdm_kwargs=tqdm_kwargs)
        # add colorbar
        colorbar_plot = plt.colorbar(cbar, cax=axes[len(fthetas) + nrun],
                                     ticks=[1, 2, 3])
        colorbar_plot.solids.set_edgecolor('face')
        colorbar_plot.ax.set_yticklabels([])
        if nrun == len(run_list) - 1:
            colorbar_plot.ax.set_yticklabels(
                [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
    # Format axis ticks and labels
    for nax, ax in enumerate(axes[:len(fthetas)]):
        ax.set_yticks([])
        ax.set_xlabel(labels[nax])
        if ax.is_first_col():
            ax.set_ylabel('probability')
        # Prune final xtick label so it doesn't overlap with next plot
        prune = 'upper' if nax != len(fthetas) - 1 else None
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(
            nbins=5, prune=prune))
    np.random.set_state(state)  # return to original random state
    return fig


def param_logx_diagram(run_list, **kwargs):
    """
    Creates diagrams of a nested sampling run's evolution as it iterates
    towards higher likelihoods, expressed as a function of log X, where X(L) is
    the fraction of the prior volume with likelihood greater than some value L.

    For a more detailed description and some example use cases, see "Diagnostic
    tests for nested sampling calculations" (Higson et al. 2018).

    Parameters
    ----------
    run_list: dict or list of dicts
        Nested sampling run(s) to plot.
    fthetas: list of functions, optional
        Quantities to plot. Each must map a 2d theta array to 1d ftheta array -
        i.e. map every sample's theta vector (every row) to a scalar quantity.
        E.g. use lambda x: x[:, 0] to plot the first parameter.
    labels: list of strs, optional
        Labels for each ftheta.
    ftheta_lims: dict, optional
        Plot limits for each ftheta.
    plot_means: bool, optional
        Should the mean value of each ftheta be plotted?
    n_simulate: int, optional
        Number of bootstrap replications to use for the fgivenx distributions.
    random_seed: int, optional
        Seed to make sure results are consistent and fgivenx caching can be
        used.
    logx_min: float, optional
        Lower limit of logx axis.
    figsize: tuple, optional
        Matplotlib figure size (in inches).
    colors: list of strs, optional
        Colors to plot run scatter plots with.
    colormaps: list of strs, optional
        Colormaps to plot run fgivenx plots with.
    npoints: int, optional
        How many points to have in the logx array used to calculate and plot
        analytical weights.
    cache: str or None
        Root for fgivenx caching (no caching if None).
    parallel: bool, optional
        fgivenx parallel optional
    point_size: float, optional
        size of markers on scatter plot (in pts)
    thin: float, optional
        factor by which to reduce the number of samples before plotting the
        scatter plot. Must be in half-closed interval (0, 1].
    rasterize_contours: bool, optional
        fgivenx rasterize_contours option.
    tqdm_kwargs: dict, optional
        Keyword arguments to pass to the tqdm progress bar when it is used in
        fgivenx while plotting contours.

    Returns
    -------
    fig: matplotlib figure
    """
    fthetas = kwargs.pop('fthetas', [lambda theta: theta[:, 0],
                                     lambda theta: theta[:, 1]])
    labels = kwargs.pop('labels', [r'$\theta_' + str(i + 1) + '$' for i in
                                   range(len(fthetas))])
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    threads_to_plot = kwargs.pop('threads_to_plot', [0])
    plot_means = kwargs.pop('plot_means', True)
    n_simulate = kwargs.pop('n_simulate', 100)
    random_seed = kwargs.pop('random_seed', 0)
    logx_min = kwargs.pop('logx_min', None)
    figsize = kwargs.pop('figsize', (6.4, 2 * (1 + len(fthetas))))
    colors = kwargs.pop('colors', ['red', 'blue', 'grey', 'green', 'orange'])
    colormaps = kwargs.pop('colormaps', ['Reds_r', 'Blues_r', 'Greys_r',
                                         'Greens_r', 'Oranges_r'])
    # Options for fgivenx
    cache_in = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    rasterize_contours = kwargs.pop('rasterize_contours', True)
    point_size = kwargs.pop('point_size', 0.2)
    thin = kwargs.pop('thin', 1)
    npoints = kwargs.pop('npoints', 100)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'disable': True})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if not isinstance(run_list, list):
        run_list = [run_list]
    # Use random seed to make samples consistent and allow caching.
    # To avoid fixing seed use random_seed=None
    state = np.random.get_state()  # save initial random state
    np.random.seed(random_seed)
    if not plot_means:
        mean_colors = [None] * len(colors)
    else:
        mean_colors = ['dark' + col for col in colors]
    nlogx = npoints
    ny_posterior = npoints
    assert len(fthetas) == len(labels)
    assert len(fthetas) == len(ftheta_lims)
    thread_linestyles = ['-', '-.', ':']
    # make figure
    # -----------
    fig, axes = plt.subplots(nrows=1 + len(fthetas), ncols=2, figsize=figsize,
                             gridspec_kw={'wspace': 0,
                                          'hspace': 0,
                                          'width_ratios': [15, 40]})
    # make colorbar axes in top left corner
    axes[0, 0].set_visible(False)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(axes[0, 0])
    colorbar_ax_list = []
    for i in range(len(run_list)):
        colorbar_ax_list.append(divider.append_axes("left", size=0.05,
                                                    pad=0.05))
    # Reverse color bar axis order so when an extra run is added the other
    # colorbars stay in the same place
    colorbar_ax_list = list(reversed(colorbar_ax_list))
    # plot runs in reverse order to put the first run on top
    for nrun, run in reversed(list(enumerate(run_list))):
        # Weight Plot
        # -----------
        ax_weight = axes[0, 1]
        ax_weight.set_ylabel('posterior\nmass')
        samples = np.zeros((n_simulate, run['nlive_array'].shape[0] * 2))
        for i in range(n_simulate):
            logx_temp = nestcheck.ns_run_utils.get_logx(
                run['nlive_array'], simulate=True)[::-1]
            logw_rel = logx_temp + run['logl'][::-1]
            w_rel = np.exp(logw_rel - logw_rel.max())
            w_rel /= np.trapz(w_rel, x=logx_temp)
            samples[i, ::2] = logx_temp
            samples[i, 1::2] = w_rel
        if logx_min is None:
            logx_min = samples[:, 0].min()
        logx_sup = np.linspace(logx_min, 0, nlogx)
        try:
            cache = cache_in + '_' + str(nrun) + '_weights'
        except TypeError:
            cache = None
        interp_alt = functools.partial(alternate_helper, func=np.interp)
        y, pmf = fgivenx.drivers.compute_pmf(
            interp_alt, logx_sup, samples, cache=cache, ny=npoints,
            parallel=parallel, tqdm_kwargs=tqdm_kwargs)
        cbar = fgivenx.plot.plot(
            logx_sup, y, pmf, ax_weight, rasterize_contours=rasterize_contours,
            colors=plt.get_cmap(colormaps[nrun]))
        ax_weight.set_xlim([logx_min, 0])
        ax_weight.set_ylim(bottom=0)
        ax_weight.set_yticks([])
        ax_weight.set_xticklabels([])
        # color bar plot
        # --------------
        colorbar_plot = plt.colorbar(cbar, cax=colorbar_ax_list[nrun],
                                     ticks=[1, 2, 3])
        colorbar_ax_list[nrun].yaxis.set_ticks_position('left')
        colorbar_plot.solids.set_edgecolor('face')
        colorbar_plot.ax.set_yticklabels([])
        if nrun == 0:
            colorbar_plot.ax.set_yticklabels(
                [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
        # samples plot
        # ------------
        logx = nestcheck.ns_run_utils.get_logx(run['nlive_array'],
                                               simulate=False)
        scatter_x = logx
        scatter_theta = run['theta']
        if thin != 1:
            assert 0 < thin <= 1, (
                'thin={} should be in the half-closed interval(0, 1]'
                .format(thin))
            state = np.random.get_state()  # save initial random state
            np.random.seed(random_seed)
            inds = np.where(np.random.random(logx.shape) <= thin)[0]
            np.random.set_state(state)  # return to original random state
            scatter_x = logx[inds]
            scatter_theta = run['theta'][inds, :]
        for nf, ftheta in enumerate(fthetas):
            ax_samples = axes[1 + nf, 1]
            ax_samples.scatter(scatter_x, ftheta(scatter_theta),
                               s=point_size, color=colors[nrun])
            if threads_to_plot is not None:
                for i in threads_to_plot:
                    thread_inds = np.where(run['thread_labels'] == i)[0]
                    ax_samples.plot(logx[thread_inds],
                                    ftheta(run['theta'][thread_inds]),
                                    linestyle=thread_linestyles[nrun],
                                    color='black', lw=1)
            ax_samples.set_xlim([logx_min, 0])
            ax_samples.set_ylim(ftheta_lims[nf])
        # Plot posteriors
        # ---------------
        posterior_axes = [axes[i + 1, 0] for i in range(len(fthetas))]
        _ = plot_bs_dists(run, fthetas, posterior_axes,
                          ftheta_lims=ftheta_lims,
                          flip_axes=True, n_simulate=n_simulate,
                          rasterize_contours=rasterize_contours,
                          cache=cache_in, nx=npoints, ny=ny_posterior,
                          colormap=colormaps[nrun],
                          mean_color=mean_colors[nrun],
                          parallel=parallel, tqdm_kwargs=tqdm_kwargs)
        # Plot means onto scatter plot
        # ----------------------------
        if plot_means:
            w_rel = nestcheck.ns_run_utils.get_w_rel(run, simulate=False)
            w_rel /= np.sum(w_rel)
            means = [np.sum(w_rel * f(run['theta'])) for f in fthetas]
            for nf, mean in enumerate(means):
                axes[nf + 1, 1].axhline(y=mean, lw=1, linestyle='--',
                                        color=mean_colors[nrun])
    # Format axes
    for nf, ax in enumerate(posterior_axes):
        ax.set_ylim(ftheta_lims[nf])
        ax.invert_xaxis()  # only invert each axis once, not for every run!
    axes[-1, 1].set_xlabel(r'$\log X$')
    # Add labels
    for i, label in enumerate(labels):
        axes[i + 1, 0].set_ylabel(label)
        # Prune final ytick label so it doesn't overlap with next plot
        prune = 'upper' if i != 0 else None
        axes[i + 1, 0].yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=3, prune=prune))
    for _, ax in np.ndenumerate(axes):
        if not ax.is_first_col():
            ax.set_yticklabels([])
        if not (ax.is_last_row() and ax.is_last_col()):
            ax.set_xticks([])
    np.random.set_state(state)  # return to original random state
    return fig


# Helper functions
# ----------------


def plot_bs_dists(run, fthetas, axes, **kwargs):
    """
    Helper function for plotting uncertainties on posterior distributions using
    bootstrap resamples and the fgivenx module. Used by bs_param_dists and
    param_logx_diagram.

    Parameters
    ----------
    run: dict
        Nested sampling run to plot.
    fthetas: list of functions
        Quantities to plot. Each must map a 2d theta array to 1d ftheta array -
        i.e. map every sample's theta vector (every row) to a scalar quantity.
        E.g. use lambda x: x[:, 0] to plot the first parameter.
    axes: list of matplotlib axis objects
    ftheta_lims: list, optional
        Plot limits for each ftheta.
    n_simulate: int, optional
        Number of bootstrap replications to use for the fgivenx
        distributions.
    colormap: matplotlib colormap
        Colors to plot fgivenx distribution.
    mean_color: matplotlib color as str
        Color to plot mean of each parameter. If None (default) means are not
        plotted.
    nx: int, optional
        Size of x-axis grid for fgivenx plots.
    ny: int, optional
        Size of y-axis grid for fgivenx plots.
    cache: str or None
        Root for fgivenx caching (no caching if None).
    parallel: bool, optional
        fgivenx parallel option.
    rasterize_contours: bool, optional
        fgivenx rasterize_contours option.
    smooth: bool, optional
        fgivenx smooth option.
    flip_axes: bool, optional
        Whether or not plot should be rotated 90 degrees anticlockwise onto its
        side.
    tqdm_kwargs: dict, optional
        Keyword arguments to pass to the tqdm progress bar when it is used in
        fgivenx while plotting contours.

    Returns
    -------
    cbar: matplotlib colorbar
        For use in higher order functions.
    """
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    n_simulate = kwargs.pop('n_simulate', 100)
    colormap = kwargs.pop('colormap', plt.get_cmap('Reds_r'))
    mean_color = kwargs.pop('mean_color', None)
    nx = kwargs.pop('nx', 100)
    ny = kwargs.pop('ny', nx)
    cache_in = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    rasterize_contours = kwargs.pop('rasterize_contours', True)
    smooth = kwargs.pop('smooth', False)
    flip_axes = kwargs.pop('flip_axes', False)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'leave': False})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert len(fthetas) == len(axes), \
        'There should be the same number of axes and functions to plot'
    assert len(fthetas) == len(ftheta_lims), \
        'There should be the same number of axes and functions to plot'
    threads = nestcheck.ns_run_utils.get_run_threads(run)
    # get a list of evenly weighted theta samples from bootstrap resampling
    bs_samps = []
    for i in range(n_simulate):
        run_temp = nestcheck.error_analysis.bootstrap_resample_run(
            run, threads=threads)
        w_temp = nestcheck.ns_run_utils.get_w_rel(run_temp, simulate=False)
        bs_samps.append((run_temp['theta'], w_temp))
    for nf, ftheta in enumerate(fthetas):
        # Make an array where each row contains one bootstrap replication's
        # samples
        max_samps = 2 * max([bs_samp[0].shape[0] for bs_samp in bs_samps])
        samples_array = np.full((n_simulate, max_samps), np.nan)
        for i, (theta, weights) in enumerate(bs_samps):
            nsamp = 2 * theta.shape[0]
            samples_array[i, :nsamp:2] = ftheta(theta)
            samples_array[i, 1:nsamp:2] = weights
        ftheta_vals = np.linspace(ftheta_lims[nf][0], ftheta_lims[nf][1], nx)
        try:
            cache = cache_in + '_' + str(nf)
        except TypeError:
            cache = None
        samp_kde = functools.partial(alternate_helper,
                                     func=weighted_1d_gaussian_kde)
        y, pmf = fgivenx.drivers.compute_pmf(
            samp_kde, ftheta_vals, samples_array, ny=ny, cache=cache,
            parallel=parallel, tqdm_kwargs=tqdm_kwargs)
        if flip_axes:
            cbar = fgivenx.plot.plot(
                y, ftheta_vals, np.swapaxes(pmf, 0, 1), axes[nf],
                colors=colormap, rasterize_contours=rasterize_contours,
                smooth=smooth)
        else:
            cbar = fgivenx.plot.plot(
                ftheta_vals, y, pmf, axes[nf], colors=colormap,
                rasterize_contours=rasterize_contours, smooth=smooth)
    # Plot means
    # ----------
    if mean_color is not None:
        w_rel = nestcheck.ns_run_utils.get_w_rel(run, simulate=False)
        w_rel /= np.sum(w_rel)
        means = [np.sum(w_rel * f(run['theta'])) for f in fthetas]
        for nf, mean in enumerate(means):
            if flip_axes:
                axes[nf].axhline(y=mean, lw=1, linestyle='--',
                                 color=mean_color)
            else:
                axes[nf].axvline(x=mean, lw=1, linestyle='--',
                                 color=mean_color)
    return cbar


def alternate_helper(x, alt_samps, func=None):
    """Helper function for making fgivenx plots of functions with 2 array
    arguments of variable lengths."""
    alt_samps = alt_samps[~np.isnan(alt_samps)]
    arg1 = alt_samps[::2]
    arg2 = alt_samps[1::2]
    return func(x, arg1, arg2)


def weighted_1d_gaussian_kde(x, samples, weights):
    """
    Gaussian kde with weighted samples (1d only). Uses Scott bandwidth factor.

    When all the sample weights are equal, this is equivalent to

    kde = scipy.stats.gaussian_kde(theta)
    return kde(x)

    When the weights are not all equal, we compute the effective number
    of samples as the information content (Shannon entropy)

    nsamp_eff = exp(- sum_i (w_i log(w_i)))

    Alternative ways to estimate nsamp_eff include Kish's formula

    nsamp_eff = (sum_i w_i) ** 2 / (sum_i w_i ** 2)

    See https://en.wikipedia.org/wiki/Effective_sample_size and "Effective
    sample size for importance sampling based on discrepancy measures"
    (Martino et al. 2017) for more information.

    Parameters
    ----------
    x: 1d numpy array
        Coordinates at which to evaluate the kde.
    samples: 1d numpy array
        Samples from which to calculate kde.
    weights: 1d numpy array of same shape as samples
        Weights of each point. Need not be normalised as this is done inside
        the function.

    Returns
    -------
    result: 1d numpy array of same shape as x
        Kde evaluated at x values.
    """
    assert x.ndim == 1
    assert samples.ndim == 1
    assert samples.shape == weights.shape
    # normalise weights and find effective number of samples
    weights /= np.sum(weights)
    nz_weights = weights[np.nonzero(weights)]
    nsamp_eff = np.exp(-1. * np.sum(nz_weights * np.log(nz_weights)))
    # Calculate the weighted sample variance
    mu = np.sum(weights * samples)
    var = np.sum(weights * ((samples - mu) ** 2))
    var *= nsamp_eff / (nsamp_eff - 1)  # correct for bias using nsamp_eff
    # Calculate bandwidth
    scott_factor = np.power(nsamp_eff, -1. / (5))  # 1d Scott factor
    sig = np.sqrt(var) * scott_factor
    # Calculate and weight residuals
    xx, ss = np.meshgrid(x, samples)
    chisquared = ((xx - ss) / sig) ** 2
    energy = np.exp(-0.5 * chisquared) / np.sqrt(2 * np.pi * (sig ** 2))
    result = np.sum(energy * weights[:, np.newaxis], axis=0)
    return result


def rel_posterior_mass(logx, logl):
    """
    Calculate the relative posterior mass for some array of logx values
    given the likelihood, prior and number of dimensions.
    The posterior mass at each logX value is proportional to L(X)X, where L(X)
    is the likelihood.
    The weight is returned normalized so that the integral of the weight with
    respect to logX is 1.

    Parameters
    ----------
    logx: 1d numpy array
        Logx values at which to calculate posterior mass.
    logl: 1d numpy array
        Logl values corresponding to each logx (same shape as logx).

    Returns
    -------
    w_rel: 1d numpy array
        Relative posterior mass at each input logx value.
    """
    logw = logx + logl
    w_rel = np.exp(logw - logw.max())
    w_rel /= np.abs(np.trapz(w_rel, x=logx))
    return w_rel


def average_by_key(dict_in, key):
    """
    Helper function for plot_run_nlive.

    Try returning the average of dict_in[key] and, if this does not work or if
    key is None, return average of whole dict.

    Parameters
    ----------
    dict_in: dict
        Values should be arrays.
    key: str

    Returns
    -------
    average: float
    """
    if key is None:
        return np.mean(np.concatenate(list(dict_in.values())))
    else:
        try:
            return np.mean(dict_in[key])
        except KeyError:
            print('method name "' + key + '" not found, so ' +
                  'normalise area under the analytic relative posterior ' +
                  'mass curve using the mean of all methods.')
            return np.mean(np.concatenate(list(dict_in.values())))
