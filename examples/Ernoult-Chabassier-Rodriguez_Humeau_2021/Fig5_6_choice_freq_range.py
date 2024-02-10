#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
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
This file illustrates how the frequency range influences the evolution of the
cost function. Three frequency ranges are compared.

This example generates the figures 5 and 6 of the article:
    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

from openwind import InstrumentGeometry, Player, InstrumentPhysics
from openwind.impedance_tools import read_impedance
from openwind.inversion import InverseFrequentialResponse

# some option for nice figures
matplotlib.rc('font', family='serif', size=14)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.close('all')

# some directory considerations
root_data = 'Impedances/'
session = 'Impedance_Measure1_20degC_'
foldername = root_data + session
geom_folder = 'Geometries/'
figure_folder = 'Figures/'
os.makedirs(figure_folder, exist_ok=True)

# %% The direct problem

# Options for simulation
rad_type = {'bell': 'unflanged_non_causal', 'holes': 'flanged_non_causal'}
opts_phy = {'temperature': 20, 'player': Player(), 'losses': True,
            'nondim': True, 'radiation_category': rad_type,
            'matching_volume': True}
opts_freq = {'observable': 'reflection', 'l_ele': .05, 'order': 10}

# the three frequency ranges
frequencies_low = np.linspace(100, 501, 100)
frequencies_full = np.arange(100, 4001, 100)
frequencies_2Hz = np.arange(100, 4001, 2)

# the geometry
common_name = 'Build_tube_Geom_'
instru_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Sensitivities.txt',
                                 geom_folder + common_name + 'Holes_Sensitivities.txt',
                                 geom_folder + 'fingering_chart_Tube_4_holes.txt')
instru_physics = InstrumentPhysics(instru_geom, **opts_phy)
optim_params = instru_geom.optim_params
total_notes = instru_geom.fingering_chart.all_notes()


# %% Loop on the design parameters observed

index_params = [0, 2, 4]  # chose between [0, 1, 2, 3, 4]

for index_param in index_params:

    # we chose what to observe
    # only this parameters is can now be changed, all the other are fixed
    optim_params.set_active_parameters(index_param)
    param_label = np.array(optim_params.labels)[optim_params.active][0]
    target_geom = optim_params.get_active_values()[0]

    print("The studied parameters is '{}'".format(param_label))
    print("Its measured geometric value is {:.2f}mm".format(target_geom*1000))

    # with respect to the chosen parameter, we chose the note observed and the
    # values of this parameter.
    n_values = 100
    if param_label == 'bore0_pos_plus':  # Main Bore length
        notes = [total_notes[0]]
        values = np.linspace(0.25, 0.5, n_values).tolist()
        XLABEL = 'Main bore length (mm)'
    elif param_label == 'bore_0_radius_plus':  # Main Bore radius
        notes = [total_notes[0]]
        values = np.linspace(1e-3, 6e-3, n_values).tolist()
        XLABEL = 'Main bore radius (mm)'
    elif param_label == 'hole1_position':
        notes = [total_notes[4]]
        values = np.linspace(0.01, 0.2, n_values).tolist()
        XLABEL = 'Location of Hole 1 (mm)'
    elif param_label == 'hole1_chimney':
        notes = [total_notes[4]]
        values = np.linspace(1e-5, 4e-3, n_values).tolist()
        XLABEL = 'Chimney length of Hole 1 (mm)'
    elif param_label == 'hole1_radius':
        notes = [total_notes[4]]
        values = np.linspace(1e-5, 2e-3, n_values).tolist()
        XLABEL = 'Radius of Hole 1 (mm)'

    print("The considered fingering is '{}'".format(notes[0]))

    # %% Instanciate the inverse problem

    # 1. first frequency range studied
    # a. compute the targets
    Z_target = []
    for k, note in enumerate(notes):
        filename = foldername + note + '.txt'
        f_measured, Z_measured = read_impedance(filename, df_filt=None)
        Z_target_note = np.interp(frequencies_low, f_measured, Z_measured)
        Z_target.append(Z_target_note)
    # b. instanciate the inverse problem
    inverse = InverseFrequentialResponse(instru_physics, frequencies_low,
                                         Z_target, notes=notes, **opts_freq)
    # c. Compute the cost function
    reflection_low = np.zeros(len(values), dtype=float)
    for k, length in enumerate(values):
        reflection_low[k] = inverse.get_cost_grad_hessian([length])[0]

    # 2. second frequency range studied
    # a. recompute the targets
    Z_target = []
    for k, note in enumerate(notes):
        filename = foldername + note + '.txt'
        f_measured, Z_measured = read_impedance(filename, df_filt=None)
        Z_target_note = np.interp(frequencies_full, f_measured, Z_measured)
        Z_target.append(Z_target_note)
    # b. change the frequency axis, set the new targets
    inverse.update_frequencies_and_mesh(frequencies_full)
    inverse.set_targets_list(Z_target, notes)
    # c. recompute the costs
    reflection_full = np.zeros(len(values), dtype=float)
    for k, length in enumerate(values):
        reflection_full[k] = inverse.get_cost_grad_hessian([length])[0]

    # 3. third frequency range studied
    Z_target = []
    for k, note in enumerate(notes):
        filename = foldername + note + '.txt'
        f_measured, Z_measured = read_impedance(filename, df_filt=None)
        Z_target_note = np.interp(frequencies_2Hz, f_measured, Z_measured)
        Z_target.append(Z_target_note)
    inverse.update_frequencies_and_mesh(frequencies_2Hz)
    inverse.set_targets_list(Z_target, notes)
    reflection_2Hz = np.zeros(len(values), dtype=float)
    for k, length in enumerate(values):
        reflection_2Hz[k] = inverse.get_cost_grad_hessian([length])[0]

    # %% the plot
    mm_values = np.asarray(values)*1e3
    plt.figure()
    plt.semilogy(mm_values, reflection_low, color='k',
                 label='$f \\in [0.1, 0.5]$ kHz')

    plt.semilogy(mm_values, reflection_full, color=[.7, .7, .7],
                 label='$f \\in [0.1, 4]$ kHz')

    plt.semilogy(mm_values, reflection_2Hz, color='k', linestyle=':',
                 label='$f \\in [0.1, 4]$ kHz, 2Hz step')

    plt.grid(True)
    plt.legend()
    plt.xlabel(XLABEL)
    plt.ylabel('Cost')

    figure_name = ('Comp_freq_observable_' + param_label + '.pdf')
    plt.savefig(figure_folder + figure_name)

plt.show()
