#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2023, INRIA
#
# This file is part of Openwind.
#
# Openwind is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Openwind is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Openwind.  If not, see <https://www.gnu.org/licenses/>.
#
# For more informations about authors, see the CONTRIBUTORS file

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from openwind.impedance_tools import plot_impedance
from openwind import InstrumentGeometry


def plot_evolution_geometry(inverse, params_evol, target_geom=None,
                             print_fig=False, save_name='', double_plot=True,
                             title='', xlim=None, ylim=None, **kwargs):
    """
    Plot the evolution of the geometry along the optimization process.

    One figure is created for each iteration

    Parameters
    ----------
    inverse : :py:class:`InverseFrequentialResponse<openwind.inversion.InverseFrequentialResponse>`
        The concerned inverse problem
    params_evol: list of array
        The evolution of the design parameters along the optimization process.
    target_geom : :py:class:`InstrumentGeometry\
        <openwind.technical.instrument_geometry.InstrumentGeometry>`, optional
        If known the target geometry. The default is None.
    print_fig : bool, optional
        If true, save the figures in png files. The default is False.
    save_name : str, optional
        The common name file part. The default is ''.
    double_plot : bool, optional
        If true, both side and top view are plotted. The default is False.
    title : str, optional
        The figure title. The default is ''.
    xlim : tuple, optional
        The xlim applied to all the figure. If None, the widest range is used.
        The default is None.
    ylim : tuple, optional
        The ylim applied to all the figure. If None, the widest range is used.
        The default is None.
    **kwargs : keywords
        The keywords given to plot.
    """
    instrument = inverse.instru_physics.instrument_geometry
    figs = list()
    xlim0_tot, ylim0_tot = (list(), list())
    for frame, params in enumerate(params_evol):

        cost = inverse.get_cost_grad_hessian(params)[0]

        it_title = title + 'Iteration: {:d}, Cost: {:.2e}'.format(frame, cost)

        fig_geom = plt.figure()
        instrument.plot_InstrumentGeometry(figure=fig_geom, label='Evolution',
                                           double_plot=double_plot, **kwargs)
        if isinstance(target_geom, InstrumentGeometry):
            target_geom.plot_InstrumentGeometry(figure=fig_geom,
                                                double_plot=double_plot,
                                                label='Target',
                                                color=[0, 0, 0],
                                                linewidth=.5)
        fig_geom.suptitle(it_title)
        ax = fig_geom.get_axes()
        xlim0_tot.append(ax[0].get_xlim())
        ylim0_tot.append(ax[0].get_ylim())
        figs.append(fig_geom)

    if xlim is None:
        xlim = (np.min(xlim0_tot), np.max(xlim0_tot))
    if ylim is None:
        ylim = (np.min(ylim0_tot), np.max(ylim0_tot))

    for fig_geom in figs:
        ax = fig_geom.get_axes()
        ax[0].axis('auto')
        ax[0].set_ylim(ylim)
        ax[0].set_xlim(xlim)
        ax[0].grid()
        if len(ax) > 1:
            ax[1].set_ylim(ylim)
            ax[1].set_xlim(xlim)
            ax[1].grid()

    if print_fig:
        for frame, fig_geom in enumerate(figs):
            fig_geom.savefig(save_name + 'Geometry_{:03d}.png'.format(frame))


def plot_evolution_impedance(inverse, params_evol,
                             note=None, Z_target=None, print_fig=False,
                             save_name='', title='', xlim=None, ylim=None,
                             **kwargs):
    """
    Plot the evolution of the impedance along the optimization process.

    One figure is created for each iteration

    Parameters
    ----------
    inverse : :py:class:`InverseFrequentialResponse<openwind.inversion.InverseFrequentialResponse>`
        The concerned inverse problem
    params_evol: list of array
        The evolution of the design parameters along the optimization process.
    note : str, optional
        The note name for which the impedance must be computed. If none, the
        fingering is unchanged. The default is None.
    Z_target : array, optional
        If known the target impedance. The default is None.
    print_fig : bool, optional
        If true, save the figures in png files. The default is False.
    save_name : str, optional
        The common name file part. The default is ''.
    title : str, optional
        The figure title. The default is ''.
    xlim : tuple, optional
        The xlim applied to all the figure. If None, the widest range is used.
        The default is None.
    ylim : tuple, optional
        The ylim applied to all the figure. If None, the widest range is used.
        The default is None.
    **kwargs : keywords
        The keywords given to plot.
    """
    figs = list()
    xlim0_tot, ylim0_tot = (list(), list())
    for frame, params in enumerate(params_evol):

        cost = inverse.get_cost_grad_hessian(params)[0]

        it_title = title + 'Iteration: {:d}, Cost: {:.2e}'.format(frame, cost)

        fig = plt.figure()
        if note is not None:
            inverse.set_note(note)
        inverse.solve()
        inverse.plot_impedance(figure=fig, label='Evolution', **kwargs)
        if Z_target is not None and len(Z_target) > 0:
            plot_impedance(inverse.frequencies, Z_target, figure=fig,
                           label='Target', marker='', linewidth=.5,
                           color=[0, 0, 0])

        fig.suptitle(it_title)
        ax = fig.get_axes()
        xlim0_tot.append(ax[0].get_xlim())
        ylim0_tot.append(ax[0].get_ylim())
        figs.append(fig)

    if xlim is None:
        xlim = (np.min(xlim0_tot), np.max(xlim0_tot))
    if ylim is None:
        ylim = (np.min(ylim0_tot), np.max(ylim0_tot))

    for fig in figs:
        ax = fig.get_axes()
        ax[0].set_ylim(ylim)
        ax[0].set_xlim(xlim)

    if print_fig:
        for frame, fig in enumerate(figs):
            fig.savefig(save_name + 'Impedance_{:03d}.png'.format(frame))


def plot_evolution_observable(inverse, params_evol, note=None, Z_target=None,
                              print_fig=False, save_name='', title='',
                              xlim=None, ylim=None, **kwargs):
    """
    Plot the evolution of the observable along the optimization process.

    One figure is created for each iteration

    Parameters
    ----------
    inverse : :py:class:`InverseFrequentialResponse<openwind.inversion.InverseFrequentialResponse>`
        The concerned inverse problem
    params_evol: list of array
        The evolution of the design parameters along the optimization process.
    note : str, optional
        The note name for which the impedance must be computed. If none, the
        fingering is unchanged. The default is None.
    Z_target : array, optional
        If known the target impedance. The default is None.
    print_fig : bool, optional
        If true, save the figures in png files. The default is False.
    save_name : str, optional
        The common name file part. The default is ''.
    title : str, optional
        The figure title. The default is ''.
    xlim : tuple, optional
        The xlim applied to all the figure. If None, the widest range is used.
        The default is None.
    ylim : tuple, optional
        The ylim applied to all the figure. If None, the widest range is used.
        The default is None.
    **kwargs : keywords
        The keywords given to plot.
    """
    figs = list()
    xlim0_tot, ylim0_tot = (list(), list())

    for frame, params in enumerate(params_evol):

        cost = inverse.get_cost_grad_hessian(params)[0]

        it_title = title + 'Iteration: {:d}, Cost: {:.2e}'.format(frame, cost)

        fig = plt.figure()
        inverse.plot_observation(note=note, figure=fig, Z_target=Z_target,
                                 label='Evolution', **kwargs)
        fig.suptitle(it_title)
        ax = fig.get_axes()
        xlim0_tot.append(ax[0].get_xlim())
        ylim0_tot.append(ax[0].get_ylim())
        figs.append(fig)

    if xlim is None:
        xlim = (np.min(xlim0_tot), np.max(xlim0_tot))
    if ylim is None:
        ylim = (np.min(ylim0_tot), np.max(ylim0_tot))

    for fig in figs:
        ax = fig.get_axes()
        ax[0].set_ylim(ylim)
        ax[0].set_xlim(xlim)

    if print_fig:
        for frame, fig in enumerate(figs):
            fig.savefig(save_name + 'Observable_{:03d}.png'.format(frame))


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.1f}",
                     textcolors=("white", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
