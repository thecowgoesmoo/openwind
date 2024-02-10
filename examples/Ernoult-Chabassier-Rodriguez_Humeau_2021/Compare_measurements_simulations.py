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
This file compares the measured impedances with the impedance computed from
the direct geometric measurement of a cylinder with 4 side holes.

It is related to the article:
    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from openwind import (InstrumentGeometry, Player, InstrumentPhysics,
                      FrequentialSolver)
from openwind.impedance_tools import read_impedance, plot_impedance

# some option for nice figures
matplotlib.rc('font', family='serif', size=14)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.close('all')

rad_type = {'bell': 'unflanged_non_causal', 'holes': 'flanged_non_causal'}
opts_phy = {'temperature': 20, 'player': Player(), 'losses': True,
            'nondim': True, 'radiation_category': rad_type,
            'matching_volume': True}
opts_freq = {'l_ele': 0.05, 'order': 10}
frequencies = np.arange(100, 4002, 2)  # from 100 to 4000 with a 2Hz step

root_data = 'Impedances/'
session1 = 'Impedance_Measure1_20degC_'
session2 = 'Impedance_Measure2_20degC_'
session3 = 'Impedance_Measure3_20degC_'
geom_folder = 'Geometries/'


# %% Targets
common_name = 'Build_tube_Geom_'
instru_geom = InstrumentGeometry(geom_folder + common_name + 'Bore_Sensitivities.txt',
                                 geom_folder + common_name + 'Holes_Sensitivities.txt',
                                 geom_folder + 'fingering_chart_Tube_4_holes_all.txt')
instru_phy = InstrumentPhysics(instru_geom, **opts_phy)
optim_params = instru_geom.optim_params


notes = instru_geom.fingering_chart.all_notes()
notes = [notes[k] for k in [0, 1, 2, 4, 8]]

instru_freq = FrequentialSolver(instru_phy, frequencies, **opts_freq)

Z_target1 = list()
Z_target2 = list()
Z_target3 = list()
impedances = list()

for k, note in enumerate(notes):
    print(instru_geom.fingering_chart.fingering_of(note))
    # load the measured impedances: set 1
    filename1 = root_data + session1 + note + '.txt'
    f_measured1, Z_measured1 = read_impedance(filename1, df_filt=None)
    Z_target_note1 = np.interp(frequencies, f_measured1, Z_measured1)
    Z_target1.append(Z_target_note1)
    # set 2
    filename2 = root_data + session2 + note + '.txt'
    f_measured2, Z_measured2 = read_impedance(filename2, df_filt=None)
    Z_target_note2 = np.interp(frequencies, f_measured2, Z_measured2)
    Z_target2.append(Z_target_note2)
    # set 3
    filename3 = root_data + session3 + note + '.txt'
    f_measured3, Z_measured3 = read_impedance(filename3, df_filt=None)
    Z_target_note3 = np.interp(frequencies, f_measured3, Z_measured3)
    Z_target3.append(Z_target_note3)

    # compute the simulated impedance
    instru_freq.set_note(note)
    instru_freq.solve()
    impedances.append(instru_freq.imped/instru_freq.get_ZC_adim())

    # plot the impedances
    fig_imp = plt.figure()
    plot_impedance(f_measured1, Z_measured1, figure=fig_imp,
                   label=(note + ': Measure 1'))
    plot_impedance(f_measured2, Z_measured2, figure=fig_imp,
                   label=(note + ': Measure 2'))
    plot_impedance(f_measured3, Z_measured3, figure=fig_imp,
                   label=(note + ': Measure 3'))
    instru_freq.plot_impedance(figure=fig_imp, linestyle='--',
                               label=(note+': Simulation'), color=[0, 0, 0])
    ax = fig_imp.get_axes()
    ax[0].set_xlim([np.min(frequencies), np.max(frequencies)])
    ax[0].legend(loc='upper right')

for k, impedance in enumerate(impedances):
    err1 = np.linalg.norm(impedance - Z_target1[k])/np.linalg.norm(impedance)
    err2 = np.linalg.norm(impedance - Z_target2[k])/np.linalg.norm(impedance)
    err3 = np.linalg.norm(impedance - Z_target3[k])/np.linalg.norm(impedance)
    print('Relative error of {}: {:=5.2f}% {:=5.2f}% '
          '{:=5.2f}%'.format(notes[k], err1*100, err2*100, err3*100))
