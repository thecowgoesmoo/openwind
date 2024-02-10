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
This file illustrates the evolution of the cost function build with different
observables with respect to some design parameters.

This example generates the Figure 3 of the article:
    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946

A previous version has been used for the proceeding:
    Ernoult, Chabassier, "Bore Reconstruction of Woodwind Instruments Using
    the Full Waveform Inversion", e-Forum Acusticum 2020, Lyon
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib


from openwind import InstrumentGeometry, Player, InstrumentPhysics
from openwind.impedance_tools import read_impedance
from openwind.inversion import InverseFrequentialResponse


font = {'family': 'serif', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.close('all')

rad_type = {'bell': 'unflanged_non_causal', 'holes': 'flanged_non_causal'}
opts_phy = {'temperature': 20, 'player': Player(), 'losses': True,
            'nondim': True, 'radiation_category': rad_type,
            'matching_volume': True}

l_ele = 0.05
order = 10

root_data = 'Impedances/'
session = 'Impedance_Measure1_20degC_'
foldername = root_data + session
geom_folder = 'Geometries/'
figure_folder = 'Figures/'

os.makedirs(figure_folder, exist_ok=True)

# %% Measured geometry
common_name = 'Build_tube_Geom_'

index_params = [2, 4] # chose between [0, 1, 2, 3, 4]

for index_param in index_params:
    instru_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Sensitivities.txt',
                                     geom_folder + common_name + 'Holes_Sensitivities.txt',
                                     geom_folder + 'fingering_chart_Tube_4_holes.txt')

    optim_params = instru_geom.optim_params
    notes = instru_geom.fingering_chart.all_notes()

    frequencies = np.arange(100, 4001, 2)
    # You can enlarge the frequency step to accelerate the computation
    # frequencies = np.arange(100, 4001, 100)

    # %% chose what to observe

    #  only this parameters is can now be changed, all the other are fixed
    optim_params.set_active_parameters(index_param)
    param_label = np.array(optim_params.labels)[optim_params.active][0]
    target_geom = optim_params.get_active_values()[0]
    print("The studied parameters is '{}'".format(param_label))
    print("Its measured geometric value is {:.2f}mm".format(target_geom*1000))

    n_values = 100

    init = np.array(instru_geom.optim_params.get_geometric_values())[optim_params.active]
    if param_label == 'bore_0_pos_plus':  # Main Bore length
        notes = [notes[0]]
        values = np.linspace(0.07, 0.5, n_values).tolist()
        XLABEL = 'Main bore length (mm)'
        target = 287.5e-3

    elif param_label == 'bore_0_radius_plus':  # Main Bore radius
        notes = [notes[0]]
        values = np.linspace(1e-3, 6e-3, n_values).tolist()
        XLABEL = 'Main bore radius (mm)'
        target = 2e-3
    elif param_label == 'hole1_position':
        notes = [notes[4]]
        values = np.linspace(0.01, 0.2, n_values).tolist()
        XLABEL = 'Location of Hole 1 (mm)'
        target = 0.1
    elif param_label == 'hole1_chimney':
        notes = [notes[4]]
        values = np.linspace(1e-5, 4e-3, n_values).tolist()
        XLABEL = 'Chimney length of Hole 1 (mm)'
        target = 1.7e-3
    elif param_label == 'hole1_radius':
        notes = [notes[4]]
        values = np.linspace(1e-5, 2e-3, n_values).tolist()
        XLABEL = 'Radius of Hole 1 (mm)'
        target = 1.5e-3

    print("The considered fingering is '{}'".format(notes[0]))

    figure_name = (param_label
                   + '_{:}_{:.0f}_{:}'.format(min(frequencies),
                                              np.mean(np.diff(frequencies)),
                                              max(frequencies))
                   + '.pdf')
    # %% Instanciate the inverse problem
    Z_target = []
    for k, note in enumerate(notes):
        filename = foldername + note + '.txt'
        f_measured, Z_measured = read_impedance(filename, df_filt=None)
        Z_target_note = np.interp(frequencies, f_measured, Z_measured)
        Z_target.append(Z_target_note)

    instru_physics = InstrumentPhysics(instru_geom, **opts_phy)
    inverse = InverseFrequentialResponse(instru_physics, frequencies,
                                         Z_target, notes=notes,
                                         observable='impedance', l_ele=l_ele,
                                         order=order)

    # %% The main loop which computes the different costs at each values
    impedance = np.zeros(len(values), dtype=float)
    impedance_modulus = np.zeros(len(values), dtype=float)
    impedance_phase = np.zeros(len(values), dtype=float)
    reflection = np.zeros(len(values), dtype=float)
    reflection_modulus = np.zeros(len(values), dtype=float)
    reflection_phase = np.zeros(len(values), dtype=float)
    reflection_phase_unwraped = np.zeros(len(values), dtype=float)

    grad_impedance = np.zeros(len(values), dtype=float)
    grad_impedance_modulus = np.zeros(len(values), dtype=float)
    grad_impedance_phase = np.zeros(len(values), dtype=float)
    grad_reflection = np.zeros(len(values), dtype=float)
    grad_reflection_modulus = np.zeros(len(values), dtype=float)
    grad_reflection_phase = np.zeros(len(values), dtype=float)
    grad_reflection_phase_unwraped = np.zeros(len(values), dtype=float)

    for k, length in enumerate(values):
        print('Value {}/{}'.format(k+1, len(values)))
        inverse.set_observation('impedance')
        inverse.set_targets_list(Z_target, notes)
        impedance[k], grad_impedance[k] = inverse.get_cost_grad_hessian([length], grad_type='adjoint')[0:2]

        inverse.set_observation('impedance_modulus')
        inverse.set_targets_list(Z_target, notes)
        impedance_modulus[k], grad_impedance_modulus[k] = inverse.get_cost_grad_hessian(grad_type='adjoint')[0:2]

        inverse.set_observation('impedance_phase')
        inverse.set_targets_list(Z_target, notes)
        impedance_phase[k], grad_impedance_phase[k] = inverse.get_cost_grad_hessian(grad_type='adjoint')[0:2]

        inverse.set_observation('reflection')
        inverse.set_targets_list(Z_target, notes)
        reflection[k], grad_reflection[k] = inverse.get_cost_grad_hessian(grad_type='adjoint')[0:2]

        inverse.set_observation('reflection_modulus')
        inverse.set_targets_list(Z_target, notes)
        reflection_modulus[k], grad_reflection_modulus[k] = inverse.get_cost_grad_hessian(grad_type='adjoint')[0:2]

        inverse.set_observation('reflection_phase')
        inverse.set_targets_list(Z_target, notes)
        reflection_phase[k], grad_reflection_phase[k] = inverse.get_cost_grad_hessian(grad_type='adjoint')[0:2]

        inverse.set_observation('reflection_phase_unwraped')
        inverse.set_targets_list(Z_target, notes)
        reflection_phase_unwraped[k], grad_reflection_phase_unwraped[k] = inverse.get_cost_grad_hessian(grad_type='adjoint')[0:2]


    # %% the plot

    plt.figure()
    plt.semilogy(np.asarray(values)*1e3, impedance, label='Impedance', color=[.5, .5, .5])
    plt.semilogy(np.asarray(values)*1e3, impedance_modulus, label='Impedance Modulus' , color=[.5, .5, .5], linestyle=':')
    plt.semilogy(np.asarray(values)*1e3, impedance_phase, label='Impedance Phase', color=[.5, .5, .5], linestyle='--')
    plt.semilogy(np.asarray(values)*1e3, reflection, label='Reflection', color='k')
    plt.semilogy(np.asarray(values)*1e3, reflection_modulus, label='Reflection Modulus', color='k', linestyle=':' )
    plt.semilogy(np.asarray(values)*1e3, reflection_phase_unwraped, label='Reflection Unwrapped phase', color='k', linestyle='--')
    plt.grid(True)
    plt.xlabel(XLABEL)
    plt.ylabel('Cost')

    plt.savefig(figure_folder + 'Cost_' + figure_name)

plt.show()
