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
This file recosntruct the geometry of conical pipe with 4 side holes from
measured impedance for 3 sets of data. The obtained geometries are saved in
the folder "Results".

These results are used to generate the figures 8 and 9 of the article:
    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946

A previous version has been used for the proceeding:
    Ernoult, Chabassier, "Bore Reconstruction of Woodwind Instruments Using
    the Full Waveform Inversion", e-Forum Acusticum 2020, Lyon
"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from openwind import InstrumentGeometry, Player, InstrumentPhysics
from openwind.impedance_tools import read_impedance
from openwind.inversion import InverseFrequentialResponse
from openwind.inversion.display_inversion import (plot_evolution_geometry,
                                                  plot_evolution_impedance,
                                                  plot_evolution_observable)


# some option for nice figures
matplotlib.rc('font', family='serif', size=14)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.close('all')

# some directory considerations
root_data = 'Impedances/'
geom_folder = 'Geometries/'
figure_folder = 'Figures/'
save_folder = 'Results/'
os.makedirs(figure_folder, exist_ok=True)
os.makedirs(save_folder, exist_ok=True)

# set to True for the figure with all fingerings
all_fing = False


# %% Options

# Options for simulation
rad_type = {'bell': 'unflanged_non_causal', 'holes': 'flanged_non_causal'}

opts_phy = {'temperature': 20, 'player': Player(), 'losses': True,
            'nondim': True, 'radiation_category': rad_type,
            'matching_volume': True}

opts_freq = {'observable': 'reflection', 'l_ele': .05, 'order': 10}
frequencies = np.arange(100, 501, 100)
frequencies_wide = np.arange(100, 4001, 100)

# some options for this script
refine = False # if True refine the reconstruction with a lot of frequencies
# [50:3000] with a 1Hz step. It can convince the user that it is useless...
plot_evolution = False # Plot the evolution of the geometry for each steps
detail = False # display details at each steps of the inversion.


def macro_plot_evolution(name, title, color):
    plt.close('all')
    plot_evolution_geometry(inverse, result.x_evol, target_geom=target_geom,
                             double_plot=False, print_fig=True,
                             save_name=figure_folder + name,
                             title=title,
                             linewidth=2, color=color)

# %% The reconstruction

for k in range(3): # main loop on the set of experimental data

    # the measured data
    if k == 0:
        session = 'Impedance_Measure1_20degC_'
        save_geom = 'Reconstruct_1'
    elif k == 1:
        session = 'Impedance_Measure2_20degC_'
        save_geom = 'Reconstruct_2'
    else:
        session = 'Impedance_Measure3_20degC_'
        save_geom = 'Reconstruct_3'

    save_geom = 'Mixed_noncaus_4k_100_' + save_geom
    foldername = root_data + session



    # %% Measured geometry
    common_name = 'Build_tube_Geom_'
    target_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Fixed.txt',
                                     geom_folder + common_name + 'Holes_Fixed.txt',
                                     geom_folder + 'fingering_chart_Tube_4_holes_all.txt')

    target_positions = np.array([.2875, .10, .13, .18, .24])
    target_chimneys = np.array([17e-4, 13e-4, 15e-4, 14e-4])
    target_radius = np.array([2e-3, 1.5e-3, 1.75e-3, 1.75e-3, 1.25e-3])

    # %% Impedance Measurements
    notes = target_geom.fingering_chart.all_notes()
    notes = [notes[k] for k in [0, 1, 2, 4, 8]]

    Z_measured = []
    f_measured = []

    for k, note in enumerate(notes):
        filename = foldername + note + '.txt'
        f_meas, Z_meas = read_impedance(filename, df_filt=None)
        Z_measured.append(Z_meas)
        f_measured.append(f_meas)

    # %%  Construction of the inverse problem

    # choice of the observable used in the cost function
    observable = 'reflection'

    # choice of the starting frequency range

    # frequencies = np.linspace(100, 500, 21)

    # Construction of the target from measured impedance
    Z_target = []
    for k in range(len(notes)):
        Z_target_note = np.interp(frequencies, f_measured[k], Z_measured[k])
        Z_target.append(Z_target_note)

    # Initial state: all the design parameters must be variable
    instru_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Length_Rad_Var.txt',
                                     geom_folder + common_name + 'Holes_Pos_Chimney_Radius_Var.txt',
                                     geom_folder + 'fingering_chart_Tube_4_holes_all.txt')

    # associate physical equations to the different elements
    instru_physics = InstrumentPhysics(instru_geom, **opts_phy)

    # Construction of the inverse problem
    optim_params = instru_geom.optim_params
    inverse = InverseFrequentialResponse(instru_physics, frequencies, Z_target,
                                         notes=notes, **opts_freq)

    # Index of the different design variables in the global vector
    # print(optim_params)
    pos_index = [0, 2, 5, 8, 11]
    chim_index = [3, 6, 9, 12]
    rad_index = [1, 4, 7, 10, 13]

    # %% Rough estimation: main bore and holes locations

    t0 = time.time()
    print('\n'+ '_'*70 + '\nThe main bore geometry (length and radius)' )
    # Only the main bore length and radius are set to active
    optim_params.set_active_parameters([pos_index[0]] + [rad_index[0]])
    # Only the all closed fingering is taken into account (select the right target)
    inverse.set_targets_list(Z_target[0], notes[0])
    # The optimization

    result = inverse.optimize_freq_model(iter_detailed=detail)
    if plot_evolution:
        macro_plot_evolution('0_Main_', 'Main Bore: ', [0.236, 0.667, 0.236])

    print('\n' + '_'*70 + '\nLocation of Hole 4')
    # Only the location of the Hole 4 is set to active
    optim_params.set_active_parameters([pos_index[4]])
    # Include only the fingering 'Hole 4 open' and the corresponding impedance
    inverse.set_targets_list(Z_target[1], notes[1])
    # The optimization
    result = inverse.optimize_freq_model(iter_detailed=detail)
    if plot_evolution:
        macro_plot_evolution('1_Hole4_', 'Hole 4: ', [0.745, .236, .236])

    print('\n' + '_'*70 + '\nLocation of Hole 3')
    optim_params.set_active_parameters([pos_index[3]])
    inverse.set_targets_list(Z_target[2], notes[2])
    result = inverse.optimize_freq_model(iter_detailed=detail)
    if plot_evolution:
        macro_plot_evolution('2_Hole3_', 'Hole 3: ', [0.157, 0.42, 0.667])

    print('\n' + '_'*70 + '\nLocation of Hole 2')
    optim_params.set_active_parameters([pos_index[2]])
    inverse.set_targets_list(Z_target[3], notes[3])
    result = inverse.optimize_freq_model(iter_detailed=detail)
    if plot_evolution:
        macro_plot_evolution('3_Hole2_', 'Hole 2: ', [0.745, .236, .236])

    print('\n' + '_'*70 + '\nLocation of Hole 1')
    optim_params.set_active_parameters([pos_index[1]])
    inverse.set_targets_list(Z_target[4], notes[4])
    result = inverse.optimize_freq_model(iter_detailed=detail)
    if plot_evolution:
        macro_plot_evolution('4_Hole1_', 'Hole 1: ', [0.157, 0.42, 0.667])

    t1 = time.time()
    total = t1-t0

    print('\n' + '='*70 + '\n' + '='*70)
    print('Rough estimation results:')
    print('- Computation Time: {:.2f} sec.'.format(total))
    pos_dev = np.abs(np.asarray(optim_params.get_geometric_values())[pos_index]
                     - target_positions)
    print('- The absolute error on the position are (in mm)\n {}'.format(pos_dev*1e3))
    print('='*70 + '\n' + '='*70 + '\n')

    # %% Refining

    # change the frequency range
    inverse.update_frequencies_and_mesh(frequencies_wide)

    # redefine the targets from measurements with this new frequency range
    Z_target = []
    for k in range(len(notes)):
        Z_target_note = np.interp(frequencies_wide, f_measured[k],
                                  Z_measured[k])
        Z_target.append(Z_target_note)

    t0 = time.time()

    print('\n' + '_'*70 + '\nSecond step: all parameters few frequencies')
    # all the design variables are set active
    optim_params.set_active_parameters('all')
    # all the note and the impedance are included
    inverse.set_targets_list(Z_target, notes)
    result = inverse.optimize_freq_model(iter_detailed=detail)
    t1 = time.time()
    total = t1-t0

    if plot_evolution:
        macro_plot_evolution('5_total_', 'Total: ', [0.236, 0.667, 0.236])

    final_pos = np.asarray(optim_params.get_geometric_values())[pos_index]
    final_chim = np.asarray(optim_params.get_geometric_values())[chim_index]
    final_rad = np.asarray(optim_params.get_geometric_values())[rad_index]

    pos_dev = np.abs(final_pos - target_positions)
    chem_dev = np.abs(final_chim - target_chimneys)
    rad_dev = np.abs(final_rad - target_radius)

    print('\n' + '='*70 + '\n' + '='*70)
    print('Refining results:')
    print('- Computation Time: {:.2f} sec.'.format(total))
    print('- The geometrie \n\t+Positions: {} \n\t+Chimneys: {} '
          '\n\t+Diameters: {}'.format(final_pos*1e3, final_chim*1e3,
                                      final_rad*2e3))
    print('- The absolute error on the locations (in mm):\n{}'.format(pos_dev*1e3))
    print('- The absolute error on the chimneys (in mm):\n{}'.format(chem_dev*1e3))
    print('- The absolute error on the radii (in mm):\n{}'.format(rad_dev*1e3))
    print('='*70 + '\n' + '='*70 + '\n')

    instru_geom.write_files(save_folder + save_geom)

    # %% Final refining
    if refine:
        save_geom_refine = 'Refine_' + save_geom
        # more frequencies included
        frequencies = np.arange(50, 3001, 1)
        inverse.update_frequencies_and_mesh(frequencies)

        # update targets
        Z_target = []
        for k in range(len(notes)):
            Z_target_note = np.interp(frequencies, f_measured[k], Z_measured[k])
            Z_target.append(Z_target_note)

        print('\n' + '_'*70 + '\nComplete')
        optim_params.set_active_parameters('all')
        inverse.set_targets_list(Z_target, notes)

        t0 = time.time()
        result = inverse.optimize_freq_model(iter_detailed=True)
        t1 = time.time()
        total = t1-t0

        final_pos = np.asarray(optim_params.get_geometric_values())[pos_index]
        final_chim = np.asarray(optim_params.get_geometric_values())[chim_index]
        final_rad = np.asarray(optim_params.get_geometric_values())[rad_index]

        pos_dev = np.abs(final_pos - target_positions)
        chem_dev = np.abs(final_chim - target_chimneys)
        rad_dev = np.abs(final_rad - target_radius)

        print('\n' + '='*70 + '\n' + '='*70)
        print('Final results with a lot of frequencies:')
        print('- Computation Time: {:.2f} sec.'.format(total))
        print('- The geometrie \n\t+Positions: {} \n\t+Chimneys: {} '
              '\n\t+Diameters: {}'.format(final_pos*1e3, final_chim*1e3,
                                          final_rad*2e3))
        print('- The absolute error on the locations (in mm):\n{}'.format(pos_dev*1e3))
        print('- The absolute error on the chimneys (in mm):\n{}'.format(chem_dev*1e3))
        print('- The absolute error on the radii (in mm):\n{}'.format(rad_dev*1e3))
        print('='*70 + '\n' + '='*70 + '\n')

        instru_geom.print_files(save_folder + save_geom_refine)

plt.show()
