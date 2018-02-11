#!/usr/bin/env python
"""
Functions for diagnostic plots of nested sampling runs.

Includes functions for plotting empirical parameter estimation diagrams of the
type described
in Section 3.1 and Figure 3 of "Sampling errors in nested sampling parameter
estimation" (Higson et al, 2018) for nest sampling runs.
"""

import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import PerfectNS.analyse_run as ar
import fgivenx.plot
import fgivenx
# Make matplotlib use tex and a nice font where the tex matches the non-tex
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def bootstrap_kde_plot(bs_df, labels=None, **kwargs):
    """
    Plots distributions of bootstrap values for each estimator. There is one
    subplot for each estimator with all runs plotted on the same axis.
    """
    figsize = kwargs.pop('figsize', (6.4, 1.5))
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    fig, axes = plt.subplots(nrows=1, ncols=len(bs_df.columns),
                             figsize=figsize)
    # get labels, using [1:-1] to strip start and end $ from est.latex_name as
    # getdist adds $s so otherwise there is an error
    if labels is not None:
        bs_df.columns = labels
    for nax, col in enumerate(bs_df):
        ax = axes[nax]
        supmin = bs_df[col].apply(np.min).min()
        supmax = bs_df[col].apply(np.max).max()
        support = np.linspace(supmin - 0.1 * (supmax - supmin),
                              supmax + 0.1 * (supmax - supmin), 100)
        for nrun, samps in enumerate(bs_df[col].values):
            kernel = scipy.stats.gaussian_kde(samps)
            pdf = kernel(support)
            pdf /= pdf.max()
            ax.plot(support, pdf)
        ax.set_ylim([0, 1.1])
        ax.set_yticks([])
        ax.set_xlabel(col)
        if ax.is_first_col():
            ax.set_ylabel('probability density')
        # if ax.is_last_col():
        #     prune = None
        # else:
        #     prune = 'upper'
        # ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,
        #                                                          prune=prune))
    # fig.subplots_adjust(wspace=0)
    return fig


# Model definitions
# =================

def interp_alternate(x, theta):
    theta = theta[~np.isnan(theta)]
    x_int = theta[::2]
    y_int = theta[1::2]
    return np.interp(x, x_int, y_int)


def samp_kde(x, theta):
    theta = theta[~np.isnan(theta)]
    kde = scipy.stats.gaussian_kde(theta)
    return kde(x)


def plot_bs_dists(run, fthetas, axes, **kwargs):
    n_simulate = kwargs.pop('n_simulate', 100)
    parallel = kwargs.pop('parallel', True)
    smooth = kwargs.pop('smooth', False)
    cache_in = kwargs.pop('cache', None)
    nx = kwargs.pop('nx', 100)
    ny = kwargs.pop('ny', nx)
    flip_x = kwargs.pop('flip_x', False)
    colormap = kwargs.pop('colormap', plt.get_cmap('Reds_r'))
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    assert len(fthetas) == len(axes), \
        'There should be the same number of axes and functions to plot'
    assert len(fthetas) == len(ftheta_lims), \
        'There should be the same number of axes and functions to plot'
    threads = ar.get_run_threads(run)
    # get a list of evenly weighted theta samples from bootstrap resampling
    bs_even_samps = []
    for i in range(n_simulate):
        run_temp = ar.bootstrap_resample_run(run, threads=threads)
        logw = ar.get_logw(run_temp, simulate=False)
        w_rel = np.exp(logw - logw.max())
        even_w_inds = np.where(w_rel > np.random.random(w_rel.shape))[0]
        bs_even_samps.append(run_temp['theta'][even_w_inds, :])
    for nf, ftheta in enumerate(fthetas):
        # Make an array where each row contains one bootstrap replication's
        # samples
        max_samps = max([a.shape[0] for a in bs_even_samps])
        samples_array = np.full((n_simulate, max_samps), np.nan)
        for i, samps in enumerate(bs_even_samps):
            samples_array[i, :samps.shape[0]] = ftheta(samps)
        theta = np.linspace(ftheta_lims[nf][0], ftheta_lims[nf][1], nx)
        if cache_in is not None:
            cache = cache_in + '_' + str(nf)
        else:
            cache = cache_in
        y, pmf = fgivenx.compute_pmf(samp_kde, theta, samples_array, ny=ny,
                                     cache=cache, parallel=parallel)
        if flip_x:
            cbar = fgivenx.plot.plot(y[::-1], theta, pmf.T, axes[nf],
                                     colors=colormap, smooth=smooth)
        else:
            cbar = fgivenx.plot.plot(theta, y, pmf, axes[nf],
                                     colors=colormap, smooth=smooth)
    return cbar


def bs_param_dists(run_list, **kwargs):
    """
    Creates parameter estimation diagrams using the
    settings module for likelihoods and the specified estimator.
    """
    n_simulate = kwargs.pop('n_simulate', 100)
    cache_in = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    smooth = kwargs.pop('smooth', False)
    nx = kwargs.pop('nx', 100)
    ny = kwargs.pop('ny', nx)
    # Use random seed to make samples consistent and allow caching.
    # To avoid fixing seed use random_seed=None
    random_seed = kwargs.pop('random_seed', 0)
    state = np.random.get_state()  # save initial random state
    np.random.seed(random_seed)
    figsize = kwargs.pop('figsize', (6.4, 3))
    fthetas = kwargs.pop('fthetas', [lambda theta: theta[:, 0],
                                     lambda theta: theta[:, 1]])
    labels = kwargs.pop('labels', [r'$\theta_' + str(i + 1) + '$' for i in
                                   range(len(fthetas))])
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    assert len(labels) == len(fthetas), \
        'There should be the same number of axes and labels'
    width_ratios = [40] * len(fthetas) + [1] * len(run_list)
    fig, axes = plt.subplots(nrows=1, ncols=len(run_list) + len(fthetas),
                             gridspec_kw={'wspace': 0.05,
                                          'width_ratios': width_ratios},
                             figsize=figsize)
    colormaps = ['Blues_r', 'Reds_r', 'Greys_r', 'Greens_r', 'Oranges_r']
    for nrun, run in enumerate(run_list):
        if cache_in is not None:
            cache = cache_in + '_' + str(nrun)
        else:
            cache = cache_in
        # add bs distribution plots
        cbar = plot_bs_dists(run, fthetas, axes[:len(fthetas)],
                             parallel=parallel, smooth=smooth,
                             ftheta_lims=ftheta_lims, cache=cache,
                             n_simulate=n_simulate, nx=nx, ny=ny,
                             colormap=colormaps[nrun])
        # add colorbar
        colorbar_plot = plt.colorbar(cbar, cax=axes[len(fthetas) + nrun],
                                     ticks=[1, 2, 3])
        colorbar_plot.solids.set_edgecolor('face')
        if nrun == len(run_list) - 1:
            colorbar_plot.ax.set_yticklabels(
                [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
        else:
            colorbar_plot.ax.set_yticklabels([])
    # Format axis ticks and labels
    for nax, ax in enumerate(axes[:len(fthetas)]):
        ax.set_yticks([])
        ax.set_xlabel(labels[nax])
        if ax.is_first_col():
            ax.set_ylabel('probability density')
        # Prune final xtick label so it dosn't overlap with next plot
        if nax != len(fthetas) - 1:
            prune = 'upper'
        else:
            prune = None
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5,
                                                                 prune=prune))
    np.random.set_state(state)  # return to original random state
    return fig


def param_est_diagram(run, **kwargs):
    """
    Creates parameter estimation diagrams using the
    settings module for likelihoods and the specified estimator.
    """
    cache_in = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    # Use random seed to make samples consistent and allow caching.
    # To avoid fixing seed use random_seed=None
    random_seed = kwargs.pop('random_seed', 0)
    state = np.random.get_state()  # save initial random state
    np.random.seed(random_seed)
    smooth_logx = kwargs.pop('smooth_logx', True)
    scatter_plot = kwargs.pop('scatter_plot', True)
    n_simulate = kwargs.pop('n_simulate', 100)
    threads_to_plot = kwargs.pop('threads_to_plot', [0, 1, 2])
    fthetas = kwargs.pop('fthetas', [lambda theta: theta[:, 0],
                                     lambda theta: theta[:, 1]])
    labels = kwargs.pop('labels', [r'$\theta_' + str(i + 1) + '$' for i in
                                   range(len(fthetas))])
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    npoints = kwargs.pop('npoints', 100)
    logx_min = kwargs.pop('logx_min', None)
    ftheta_sups = [np.linspace(lim[0], lim[1], npoints) for lim in ftheta_lims]
    nlogx = kwargs.pop('nlogx', npoints)
    # make figure
    figsize = kwargs.pop('figsize', (6.4, 2 * (1 + len(fthetas))))
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    fig, axes = plt.subplots(nrows=1 + len(fthetas), ncols=2, figsize=figsize,
                             gridspec_kw={'wspace': 0,
                                          'hspace': 0,
                                          'width_ratios': [15, 40]})
    axes[0, 0].set_visible(False)
    # Weight Plot
    # -------------------------
    ax_weight = axes[0, 1]
    ax_weight.set_ylabel('posterior\nmass')
    samples = np.zeros((n_simulate, run['nlive_array'].shape[0] * 2))
    for i in range(n_simulate):
        logx_temp = ar.get_logx(run['nlive_array'], simulate=True)[::-1]
        logw_rel = logx_temp + run['logl'][::-1]
        w_rel = np.exp(logw_rel - logw_rel.max())
        w_rel /= np.trapz(w_rel, x=logx_temp)
        samples[i, ::2] = logx_temp
        samples[i, 1::2] = w_rel
    if logx_min is None:
        logx_min = samples[:, 0].min()
    logx_sup = np.linspace(logx_min, 0, nlogx)
    ax_weight.set_xlim([logx_min, 0])
    ax_weight.set_yticks([])
    ax_weight.set_xticklabels([])
    if cache_in is not None:
        cache = cache_in + '_weights'
    else:
        cache = cache_in
    y, pmf = fgivenx.compute_pmf(interp_alternate, logx_sup, samples,
                                 cache=cache, ny=npoints, parallel=parallel)
    cbar = fgivenx.plot.plot(logx_sup, y, pmf, ax_weight,
                             colors=plt.get_cmap('Reds_r'))
    # color bar plot
    # --------------
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(axes[0, 0])
    colorbar_ax = divider.append_axes("left", size="25%")
    # spacing = divider.append_axes("left", size="50%")
    # spacing.set_visible(False)
    colorbar_plot = plt.colorbar(cbar, cax=colorbar_ax, ticks=[1, 2, 3])
    colorbar_ax.yaxis.set_ticks_position('left')
    colorbar_plot.solids.set_edgecolor('face')
    colorbar_plot.ax.set_yticklabels(
        [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
    # samples plot
    # ------------
    run['logx'] = ar.get_logx(run['nlive_array'], simulate=False)
    logw_expected = ar.get_logw(run, simulate=False)
    w_rel = np.exp(logw_expected - logw_expected.max())
    w_rel /= np.sum(w_rel)
    means = [np.sum(w_rel * ftheta(run['theta'])) for ftheta in fthetas]
    # Plot samples
    # ------------
    threads = [ar.dict_given_samples_array(thread) for thread in
               ar.get_run_threads(run)]
    # thread_arrays = ar.get_run_threads(run)
    # threads = [ar.dict_given_samples_array(thread) for thread in
    #            thread_arrays]
    for nf, ftheta in enumerate(fthetas):
        ax_samples = axes[1 + nf, 1]
        for i in threads_to_plot:
            ax_samples.plot(threads[i]['logx'], ftheta(threads[i]['theta']),
                            color='black', lw=1)
        if scatter_plot:
            ax_samples.scatter(run['logx'], ftheta(run['theta']), s=0.4,
                               color='red')
        else:
            if cache_in is not None:
                cache = cache_in + '_param_' + str(nf)
            else:
                cache = cache_in
            # try sampling and combining batches
            # batches = []
            # for i in range(2000):
            #     inds = np.random.randint(0, len(thread_arrays), 4)
            #     temp = ar.combine_threads([thread_arrays[j] for j in inds],
            #                               run['thread_min_max'][inds])
            #     batches.append(temp)
            # threads = batches
            max_nsamp = max([thread['logx'].shape[0] for thread in threads])
            samples = np.full((len(threads), 2 * max_nsamp), np.nan)
            for i, thread in enumerate(threads):
                nsamp = thread['logx'].shape[0]
                samples[i, :2 * nsamp:2] = thread['logx'][::-1]
                samples[i, 1:2 * nsamp:2] = ftheta(thread['theta'])[::-1]
            y, pmf = fgivenx.compute_pmf(interp_alternate, logx_sup, samples,
                                         y=ftheta_sups[nf], cache=cache)
            _ = fgivenx.plot.plot(logx_sup, y, pmf, ax_samples,
                                  smooth=smooth_logx)
        ax_samples.set_xlim([logx_min, 0])
        ax_samples.set_ylim(ftheta_lims[nf])
        ax_samples.axhline(y=means[nf], lw=1, linestyle='--', color='black')
    # Plot posteriors
    # ---------------
    posterior_axes = [axes[i + 1, 0] for i in range(len(fthetas))]
    _ = plot_bs_dists(run, fthetas, posterior_axes,
                      ftheta_lims=ftheta_lims,
                      flip_x=True, n_simulate=n_simulate,
                      cache=cache_in, nx=npoints, ny=npoints,
                      parallel=parallel)
    for nf, ax in enumerate(posterior_axes):
        ax.set_ylim(ftheta_lims[nf])
        ax.axhline(y=means[nf], lw=1, linestyle='--', color='black')
    axes[-1, 1].set_xlabel(r'$\log X$')
    # Add labels
    for i, label in enumerate(labels):
        axes[i + 1, 0].set_ylabel(label)
        # Prune final ytick label so it dosn't overlap with next plot
        if i != 0:
            prune = 'upper'
        else:
            prune = None
        axes[i + 1, 0].yaxis.set_major_locator(matplotlib.ticker
                                               .MaxNLocator(nbins=3,
                                                            prune=prune))
    for _, ax in np.ndenumerate(axes):
        if not ax.is_first_col():
            ax.set_yticklabels([])
        if not (ax.is_last_row() and ax.is_last_col()):
            ax.set_xticks([])
    np.random.set_state(state)  # return to original random state
    return fig
