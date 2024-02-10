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

"""
The file compares the reconstruction results obtained from three sets of
measured data with the direct geometric measurements of the design parameters.
The uncertainties are estimated from teh sensitivity of the observable with
respect to the design variable for each fingering.

.. warning::
    Please execute before `Cylinder4Holes_Reconstruction.py` to generate the
    results.

It generates Figures 8 and 9 of the article:
    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from openwind import InstrumentGeometry, Player, InstrumentPhysics
from openwind.inversion import InverseFrequentialResponse


font = {'family': 'serif', 'size': 12}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.close('all')

# Options for simulation
rad_type = {'bell': 'unflanged_non_causal', 'holes': 'flanged_non_causal'}
opts_phy = {'temperature': 20, 'player': Player(), 'losses': True,
            'nondim': True, 'radiation_category': rad_type,
            'matching_volume': True}
opts_freq = {'observable': 'reflection', 'l_ele': .05, 'order': 10}
frequencies = np.arange(100, 4001, 100)

geom_folder = 'Geometries/'
recons_folder = 'Results/'
fig_fold = 'Figures/'





# %% The reference geometries

# the measured geometry
common_name = 'Build_tube_Geom_'
measured_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Sensitivities.txt',
                                   geom_folder + common_name + 'Holes_Sensitivities.txt',
                                   geom_folder + 'fingering_chart_Tube_4_holes_all.txt')
index_loc = [2, 5, 8, 11, 0]
index_rad = [4, 7, 10, 13, 1]
index_chim = [3, 6, 9, 12]

# the measured value of the design parameters
geom_design = measured_geom.optim_params.get_geometric_values()
geom_loc = np.array([geom_design[k] for k in index_loc])*1e3
geom_rad = np.array([geom_design[k] for k in index_rad])*1e3
geom_chim = np.array([geom_design[k] for k in index_chim])*1e3

# For comparison: initial instrument
init_geom = InstrumentGeometry(geom_folder + 'Build_tube_Geom_Bore_Length_Rad_Var.txt',
                               geom_folder + 'Build_tube_Geom_Holes_Pos_Chimney_Radius_Var.txt',
                               geom_folder + 'fingering_chart_Tube_4_holes_all.txt')

# %% Uncertainties

# Direct geometric measurement uncertainties
error_geom_loc = 0.5
error_geom_rad = 0.05
error_geom_chim = 0.2

# Relative error on impedance measurements
err_impedance = .05

# computation of the uncertainties from the sensitivity of the observable
# this sentivity being few dependend to the absolute of the design parameters
# it is computed once and for all with the measured geometry

optim_params = measured_geom.optim_params
observable = 'reflection'
notes = measured_geom.fingering_chart.all_notes()
notes = [notes[k] for k in [0, 1, 2, 4, 8]]
Z_target0 = [np.ones_like(frequencies) for k in notes]
instru_phy = InstrumentPhysics(measured_geom, **opts_phy)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Z_target0,
                                     notes=notes, **opts_freq)
sensitivities = inverse.compute_sensitivity_observable()[0]
print('\nMinimal uncertainties from sensitivities per fingering (in mm): ')
for k, label in enumerate(optim_params.labels):
    print('{:20} {:=4.2f}'.format(label, min(err_impedance/sensitivities[:, k]*1e3)))

error_loc  = np.array([min(err_impedance/sensitivities[:, k]*1e3) for k in index_loc])
error_rad  = np.array([min(err_impedance/sensitivities[:, k]*1e3) for k in index_rad])
error_chim  = np.array([min(err_impedance/sensitivities[:, k]*1e3) for k in index_chim])


# %% the reconstructed geometries
recons_names = ['Mixed_noncaus_4k_100_Reconstruct_1_',
                'Mixed_noncaus_4k_100_Reconstruct_2_',
                'Mixed_noncaus_4k_100_Reconstruct_3_',
                'Unflanged_noncaus_4k_100_Reconstruct_1_',
                ]

recons_loc = list()
recons_rad = list()
recons_chim = list()

for k, recons_name in enumerate(recons_names):
    print(recons_name)
    recons = InstrumentGeometry(recons_folder + recons_name + 'MainBore.txt',
                                recons_folder + recons_name + 'Holes.txt',
                                recons_folder + recons_name + 'FingeringChart.txt')

    recons_design = recons.optim_params.get_geometric_values()
    recons_loc.append(np.array([recons_design[k] for k in index_loc])*1e3)
    recons_rad.append(np.array([recons_design[k] for k in index_rad])*1e3)
    recons_chim.append(np.array([recons_design[k] for k in index_chim])*1e3)

    # plot comparison of geometries for the first set
    if k == 0:
        fig_geom = plt.figure(figsize=(12, 4))
        measured_geom.plot_InstrumentGeometry(figure=fig_geom, color='k',
                                              linewidth=4, double_plot=False)
        init_geom.plot_InstrumentGeometry(figure=fig_geom, double_plot=False,
                                          linewidth=1,
                                          color=[0., 0., 0], linestyle='-')
        recons.plot_InstrumentGeometry(figure=fig_geom, double_plot=False,
                                       linewidth=2,
                                       color=[0.5, 0.5, .5], linestyle='-')

        ax = fig_geom.get_axes()
        ax[0].axis('auto')
        lines = ax[0].get_lines()
        lines[0].set_label('Direct Measurements')
        lines[5].set_label('Initial state')
        lines[10].set_label('Reconstructed 1')
        plt.ylim([1.9, 4.1])
        plt.xlim([0, 290])
        plt.ylabel('Distance to the main axis (mm)')
        plt.legend()
        plt.grid()
        plt.subplots_adjust(left=.100, bottom=.150, right=.950, top=.950,
                            wspace=.2, hspace=.2)
        plt.savefig(fig_fold + 'Recons_geom.pdf')


# %% Comparison parameter by parameter

def plot_params(x, recons, error_recons, geom, error_geom):
    xtick = [0, 1, 2, 3, 4]
    xticklabel = ['Hole1', 'Hole2', 'Hole3', 'Hole4', 'MainBore']
    plot_options = {'linewidth': 2, 'markersize': 10, 'markeredgewidth': 3}
    legends = ['Geom. Uncertainties', 'Reconstructed 1', 'Reconstructed 2',
               'Reconstructed 3', 'Recons.1, unflanged']
    color1 = [.3, .3, .3]
    color2 = [.6, .6, .6]
    offset = .15
    x_bound = np.array([-1, len(index_loc)+1, np.NaN, len(index_loc)+1, -1])
    bounds = np.array([1, 1, np.NaN, -1, -1])

    plt.figure()
    plt.plot(x_bound, bounds*error_geom, 'k:', linewidth=2)
    plt.errorbar(x-offset, recons[0]-geom, error_recons, marker='x', ls='',
                 color='k', **plot_options)
    plt.errorbar(x, recons[1]-geom, error_recons, marker='x', ls='',
                 color=color1, **plot_options)
    plt.errorbar(x + offset, recons[2]-geom, error_recons, marker='x', ls='',
                 color=color2, **plot_options)
    plt.errorbar(x - 2*offset, recons[3]-geom, error_recons, marker='v', ls='',
                 color='k',  markersize=10)
    plt.grid()
    plt.legend(legends)
    plt.xticks(xtick[:len(x)], xticklabel[:len(x)])
    plt.xlim([-.5, len(x)-.5])
    plt.ylabel('Deviation (mm)')
    plt.subplots_adjust(left=.180, bottom=.150, right=.950, top=.950, wspace=.2, hspace=.2)


x_loc = np.arange(len(index_loc))
x_rad = np.arange(len(index_rad))
x_chim = np.arange(len(index_chim))
plot_params(x_loc, recons_loc, error_loc, geom_loc, error_geom_loc)
plt.xlabel('Locations')
plt.savefig(fig_fold + 'Locations_deviation.pdf')

plot_params(x_rad, recons_rad, error_rad, geom_rad, error_geom_rad)
plt.xlabel('Radii')
plt.savefig(fig_fold + 'Radii_deviation.pdf')

plot_params(x_chim, recons_chim, error_chim, geom_chim, error_geom_chim)
plt.xlabel('Chimney height')
plt.savefig(fig_fold + 'Chimney_deviation.pdf')
