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
This file compute the sensitivity of the observable wr to the location of the
hole of a cylinder with 4 side holes.

It is linked to the article and plots the figure 4:
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
from openwind import InstrumentGeometry, Player, InstrumentPhysics
from openwind.inversion import InverseFrequentialResponse


import matplotlib
matplotlib.rc('font', family='serif', size=16)
matplotlib.rcParams['mathtext.fontset'] = 'cm'

plt.close('all')

frequencies = np.arange(100, 4000, 2)
# To accelerate the computation the frequency step can be enlarged:
# frequencies = np.arange(100, 4000, 100)

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
# %% The geometry
common_name = 'Build_tube_Geom_'

# the sensitivity can be computed only on "variable" parameters.
# we load geometry files in which all the parameters are variable
instru_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Sensitivities.txt',
                                 geom_folder + common_name + 'Holes_Sensitivities.txt',
                                 geom_folder + 'fingering_chart_Tube_4_holes_all.txt')

notes = instru_geom.fingering_chart.all_notes()

fig_bore = plt.figure()
instru_geom.plot_InstrumentGeometry(figure=fig_bore)

instru_physics = InstrumentPhysics(instru_geom, **opts_phy)

# %% Construction of the 'Inversion' Problem

# we chose the observable for which we would like to compute the sensitivities
observable = 'reflection'

# the sensitivity is independent on the targets: we defined them arbitrary
Z_target = [np.ones_like(frequencies) for k in notes]
optim_params = instru_geom.optim_params

inverse = InverseFrequentialResponse(instru_physics, frequencies,  Z_target,
                                     notes=notes, observable=observable,
                                     l_ele=l_ele, order=order)

# %% The sensitivity computation

pos_index = [1, 2, 5, 8, 11]
chim_index = [3, 6, 9, 12]
rad_index = [0, 4, 7, 10, 13]

# We would like to compute the sensitivity with respect to locations only

# 1-we activate only these parameters in the OptimizationParameters
optim_params.set_active_parameters(pos_index)

# 2-we compute the sensitivity
sensitivities_positions, _ = inverse.compute_sensitivity_observable()

# 3- we plot the result we some options
fig_sens, ax_sens = inverse.plot_sensitivities(logscale=True, relative=False,
                                               vmin=0, vmax=2.5,
                                               param_order=[1, 2, 3, 4, 0],
                                               text_on_map=False)
fig_sens.set_figwidth(2.5*fig_sens.get_figwidth())
ax_sens.set_yticklabels(['Hole 1 location', 'Hole 2 location',
                         'Hole 3 location', 'Hole 4 location',
                         'Main bore length'])
ax_sens.set_xlabel('Fingering')

# 4- we plot over the fingering chart
instru_geom.fingering_chart.plot_chart(figure=fig_sens, markersize=22,
                                       fillstyle='none', color='w',
                                       open_only=False, markeredgewidth=3)

ax_sens.plot(np.arange(0, len(notes)), 4*np.ones(len(notes)), markersize=22,
             fillstyle='none', color='w', linestyle='',
             markeredgewidth=3, marker='o')


fig_sens.savefig(figure_folder + 'Sensitivity_reflection.pdf')

plt.show()
