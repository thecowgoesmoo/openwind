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
This file compare measured and simulated impedance for of a 436mm long cylinder
with 1.95mm inner radius for two radiation condition (unflanged and infinite
flanged). It allows the estimation of the impedance measurement uncertainties.

It is related to the article:
    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from openwind.impedance_tools import (read_impedance, plot_impedance,
                                      plot_reflection)
from openwind import ImpedanceComputation


font = {'family': 'serif', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.close('all')

# %% Simulations
frequencies = np.arange(100, 4001, 2)
geom_cylinder = [[0, .436, 1.95e-3, 1.95e-3, 'linear']]
options = {'temperature': 20, 'losses': True, 'nondim': True,
           'matching_volume': True, 'l_ele': 0.01, 'order': 10}

simu_unflanged = ImpedanceComputation(frequencies, geom_cylinder, **options,
                                      radiation_category='unflanged', )

simu_flanged = ImpedanceComputation(frequencies, geom_cylinder, **options,
                                    radiation_category='infinite_flanged')

Z_unflanged = simu_unflanged.impedance/simu_unflanged.Zc
Z_flanged = simu_flanged.impedance/simu_flanged.Zc

# %% Measurements
filename = 'Impedances/Impedance_20degC_Measure_Cyl_436mm.txt'

f_measured, Z_measured = read_impedance(filename, df_filt=None)
Ref_measured = (Z_measured - 1) / (Z_measured + 1)
Z_target = np.interp(frequencies, f_measured, Z_measured)

# %% deviations
err = np.linalg.norm(Z_unflanged - Z_target) / np.linalg.norm(Z_unflanged)
err_flanged = np.linalg.norm(Z_flanged - Z_target) / np.linalg.norm(Z_flanged)
print('The relative errors are {:.2f}% (unflanged) and {:.2f}% '
      '(flanged)'.format(err*100, err_flanged*100))

# %% Plots
fig_imp = plt.figure()
simu_unflanged.plot_impedance(figure=fig_imp, label=('Simulation'))
simu_flanged.plot_impedance(figure=fig_imp, linestyle='--',
                            label=('Simulation flanged'))
plot_impedance(f_measured, Z_measured, figure=fig_imp, label=('Measure'))
ax = fig_imp.get_axes()
ax[0].set_xlim([np.min(f_measured), 3000])
ax[0].legend(loc='upper right')
ax[0].grid(True)
ax[1].grid(True)

fig_ref = plt.figure()
plot_reflection(frequencies, Z_unflanged, 1, figure=fig_ref,
                label=('Simulation'), complex_plane=False)
plot_reflection(frequencies, simu_flanged.impedance/simu_unflanged.Zc, 1,
                figure=fig_ref, label=('Simulation flanged'),
                complex_plane=False)
plot_reflection(f_measured, Z_measured, 1, figure=fig_ref, label=('Measure'),
                complex_plane=False)

# %%
temp = 25
options = {'temperature': temp, 'losses': True, 'nondim': True,
           'matching_volume': True, 'l_ele': 0.01, 'order': 10}

simu_25 = ImpedanceComputation(frequencies, geom_cylinder, **options,
                                      radiation_category='unflanged' )

Z_25 = simu_25.impedance/simu_25.Zc
f_25 = frequencies * np.sqrt((20 + 273.15)/ (temp + 273.15))

Z_25_interp = np.interp(frequencies, f_25, Z_25)

err_25 = np.linalg.norm(Z_unflanged - Z_25_interp) / np.linalg.norm(Z_unflanged)

print('The relative errorsbetween 20°C and 25°C is {:.2f}%'.format(err_25*100))
