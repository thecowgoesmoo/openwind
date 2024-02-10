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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from openwind import InstrumentGeometry, Player, InstrumentPhysics
from openwind.impedance_tools import read_impedance
from openwind.inversion import InverseFrequentialResponse

"""
The file illustrates the importance of the inclusion of several fingerings to
adjust the values of the holes dimensions (radius and chimney height).

It generates Figure 7 of the article:
    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946

"""

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

# set to True for the figure with all fingerings
all_fing = False


# %% Construction of the problem

# Options for simulation
rad_type = {'bell': 'unflanged_non_causal', 'holes': 'flanged_non_causal'}

opts_phy = {'temperature': 20, 'player': Player(), 'losses': True,
            'nondim': True, 'radiation_category': rad_type,
            'matching_volume': True}

opts_freq = {'observable': 'reflection', 'l_ele': .05, 'order': 10}
frequencies = np.arange(100, 4001, 100)

# geometry
common_name = 'Build_tube_Geom_'
instru_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Fixed.txt',
                                 geom_folder + common_name + 'Holes_Sensitivities.txt',
                                 geom_folder + 'fingering_chart_Tube_4_holes_all.txt')
# physics
intru_phy = InstrumentPhysics(instru_geom, **opts_phy)

# targets
Z_target = []
total_notes = instru_geom.fingering_chart.all_notes()
if all_fing:
    notes = [total_notes[k] for k in [0, 1, 2, 4, 8]]
else:
    notes = [total_notes[8]]
for k, note in enumerate(notes):
    filename = foldername + note + '.txt'
    f_measured, Z_measured = read_impedance(filename, df_filt=None)
    Z_target_note = np.interp(frequencies, f_measured, Z_measured)
    Z_target.append(Z_target_note)

# inverse problem
inverse = InverseFrequentialResponse(intru_phy, frequencies, Z_target,
                                     notes=notes, **opts_freq)
# active only the two considered parameters
inverse.optim_params.set_active_parameters([1, 2])

# %% The main loop to compute the cost for each value

target_radius = 1.5e-3
target_chimney = 1.7e-3
radii = (np.linspace(0.8, 1.2, 30)*target_radius).tolist()
chimneys = (np.linspace(.5, 1.5, 42)*target_chimney).tolist()

reflection = np.zeros((len(radii), len(chimneys)))
for kr, rad in enumerate(radii):
    print("{}/{}".format(kr, len(radii)-1))
    for kc, chim in enumerate(chimneys):
        reflection[kr, kc] = inverse.get_cost_grad_hessian([chim, rad])[0]

# %% Plot
X, Y = np.meshgrid(np.array(chimneys)*1e3, np.array(radii)*1e3)

fig, ax = plt.subplots()
if all_fing:
    my_ax = ax.contour(X, Y, np.log10(reflection),
                       np.append(-1.89, np.arange(-1.85, -.9, .05)),
                       linewidths=3)
    fig_name = 'radius_chimney_all_fingerings'
else:
    my_ax = ax.contour(X, Y, np.log10(reflection),
                       np.arange(-3, -1, .05), linewidths=3)
    fig_name = 'radius_chimney_one_fingering'
ax.set_xlabel('Chimney (mm)')
ax.set_ylabel('Radius (mm)')
cbar = fig.colorbar(my_ax)
cbar.ax.set_ylabel('Cost (log scale)', rotation=-90, va="bottom")

fig.savefig(figure_folder + 'Cost_' + fig_name + '.pdf')

plt.show()
