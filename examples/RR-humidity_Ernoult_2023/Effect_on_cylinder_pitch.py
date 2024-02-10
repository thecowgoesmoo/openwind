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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

from openwind import InstrumentGeometry, InstrumentPhysics, FrequentialSolver, Player

"""
This file compute and plot the evolution of the modal parameters of a cylinder,
with respect to:
    - the temperature
    - the humidity rate
    - the carbon dioxyde
by using different expressions for the physical quantities.

This file is related to the research report: A.Ernoult, 2023 "Effect of air
humidity and carbon dioxide in the sound propagation for the modeling of wind
musical instruments" RR-9500, Inria. 2023, pp.28. https://hal.inria.fr/hal-04008847
"""


# Plot options
colors = list(matplotlib.colors.TABLEAU_COLORS)
plt.close('all')
font = {'family': 'serif', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# The instrument geometry
main_bore = [[0,500,4,4,'linear']]
my_geom = InstrumentGeometry(main_bore, unit='mm')

# modeling options
freq = np.linspace(100,2e3,1000)
rad = 'unflanged'
losses = 'diffrepr'
method = 'modal'
shared_options = dict(player=Player(), losses=losses, radiation_category=rad)

# Loop to study of the influence of temperature humidity and carbon rate
parameters = ['temperature', 'humidity', 'CO2']

temp_ref = 40
CO2_ref = 4.2e-4
H20_ref = 0

CO2_loop = np.linspace(0, .1,11)
HR_loop = np.linspace(0, 1,11)
t_loop = np.linspace(0,40,9)

for loop in parameters:

    # some plotting and loop options
    if loop=='temperature':
        x_loop = t_loop
        xlabel = 'Temperature [°C]'
        x_plot = t_loop

    elif loop=='humidity':
        x_loop = HR_loop
        xlabel = 'Humidity rate [%]'
        x_plot = 100*HR_loop

    elif loop=='CO2':
        x_loop = CO2_loop
        xlabel = 'CO2 molar frac. [%]'
        x_plot = 100*CO2_loop

    # init the list of modal parameters
    f_CK_v, Q_CK_v, a_CK_v = (list(), list(), list())
    f_dry , Q_dry, a_dry = (list(), list(), list())
    f_Zuck , Q_Zuck, a_Zuck = (list(), list(), list())
    f_Tsi, Q_Tsi, a_Tsi = (list(), list(), list())

    with warnings.catch_warnings():
        warnings.simplefilter("once")

        for x in x_loop:
            if loop=='temperature':
                print(f'Temperature: {x:.0f}°C')
                options = dict(temperature=x, carbon=CO2_ref, humidity=H20_ref)

            elif loop=='humidity':
                print(f'Humidity: {100*x:.0f}%')
                options = dict(temperature=temp_ref, carbon=CO2_ref, humidity=x)
            elif loop=='CO2':
                print(f'CO2: {100*x:.0f}%')
                options = dict(temperature=temp_ref, carbon=x, humidity=H20_ref)

            # instanciate Physics with different set of parameters
            phy_CK_v = InstrumentPhysics(my_geom, ref_phy_coef='Chaigne_Kergomard',
                                         **shared_options, **options)

            phy_dry = InstrumentPhysics(my_geom, ref_phy_coef='RR',
                                        **shared_options, **options)

            phy_Zuck = InstrumentPhysics(my_geom, ref_phy_coef='RR_Zuckerwar',
                                         **shared_options, **options)

            phy_Tsi = InstrumentPhysics(my_geom, ref_phy_coef='RR_Tsilingiris',
                                        **shared_options, **options)

            # freuential computation and modal parameters
            freq_CK_v = FrequentialSolver(phy_CK_v, freq, compute_method=method)
            freq_CK_v.solve()
            f_CK_v_n, Q_CK_v_n, a_CK_v_n = freq_CK_v.resonance_peaks()
            f_CK_v.append(f_CK_v_n)
            Q_CK_v.append(Q_CK_v_n)
            a_CK_v.append(a_CK_v_n)

            freq_dry = FrequentialSolver(phy_dry, freq, compute_method=method)
            freq_dry.solve()
            f_dry_n, Q_dry_n, a_dry_n = freq_dry.resonance_peaks()
            f_dry.append(f_dry_n)
            Q_dry.append(Q_dry_n)
            a_dry.append(a_dry_n)

            freq_Zuck = FrequentialSolver(phy_Zuck, freq, compute_method=method)
            freq_Zuck.solve()
            f_Zuck_n, Q_Zuck_n, a_Zuck_n = freq_Zuck.resonance_peaks()
            f_Zuck.append(f_Zuck_n)
            Q_Zuck.append(Q_Zuck_n)
            a_Zuck.append(a_Zuck_n)

            freq_Tsi = FrequentialSolver(phy_Tsi, freq, compute_method=method)
            freq_Tsi.solve()
            f_Tsi_n, Q_Tsi_n, a_Tsi_n = freq_Tsi.resonance_peaks()
            f_Tsi.append(f_Tsi_n)
            Q_Tsi.append(Q_Tsi_n)
            a_Tsi.append(a_Tsi_n)

    # keep only the first peak data
    f0_CK_v = np.array(f_CK_v)[:,0]
    f0_dry = np.array(f_dry)[:,0]
    f0_Zuck = np.array(f_Zuck)[:,0]
    f0_Tsi = np.array(f_Tsi)[:,0]

    a0_CK_v = np.abs(a_CK_v)[:,0]
    a0_dry = np.abs(a_dry)[:,0]
    a0_Zuck = np.abs(a_Zuck)[:,0]
    a0_Tsi = np.abs(a_Tsi)[:,0]

    Q0_CK_v = np.array(Q_CK_v)[:,0]
    Q0_dry = np.array(Q_dry)[:,0]
    Q0_Zuck = np.array(Q_Zuck)[:,0]
    Q0_Tsi = np.array(Q_Tsi)[:,0]

    # the figures...
    plt.figure()
    plt.plot(x_plot, 1200*np.log2(f0_CK_v/f0_CK_v[0]), 'k', label='C&K')
    plt.plot(x_plot, 1200*np.log2(f0_dry/f0_dry[0]), label='RR')
    plt.plot(x_plot, 1200*np.log2(f0_Zuck/f0_Zuck[0]), '--', label='Thermo-viscous Zuck.')
    plt.plot(x_plot, 1200*np.log2(f0_Tsi/f0_Tsi[0]), '--', label='Thermo-viscous Tsi.')
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('$f_0$ deviation [cent]')
    plt.tight_layout()
    plt.savefig('Pitch_f_' + loop + '.pdf')

    plt.figure()
    plt.plot(x_plot, 20*np.log10(a0_CK_v/a0_CK_v[0]), 'k', label='C&K')
    plt.plot(x_plot, 20*np.log10(a0_dry/a0_dry[0]), label='RR')
    plt.plot(x_plot, 20*np.log10(a0_Zuck/a0_Zuck[0]), '--', label='Thermo-viscous Zuck.')
    plt.plot(x_plot, 20*np.log10(a0_Tsi/a0_Tsi[0]), '--', label='Thermo-viscous Tsi.')
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('$a_0$ deviation [dB]')
    plt.tight_layout()
    plt.savefig('Pitch_a_' + loop + '.pdf')

    plt.figure()
    plt.plot(x_plot, 100*(Q0_CK_v/Q0_CK_v[0]-1), 'k',label='C&K')
    plt.plot(x_plot, 100*(Q0_dry/Q0_dry[0]-1), label='RR')
    plt.plot(x_plot, 100*(Q0_Zuck/Q0_Zuck[0]-1), '--', label='Thermo-viscous Zuck.')
    plt.plot(x_plot, 100*(Q0_Tsi/Q0_Tsi[0]-1), '--', label='Thermo-viscous Tsi.')
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Q-factor deviation [%]')
    plt.tight_layout()
    plt.savefig('Pitch_Q_' + loop + '.pdf')

plt.show
