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
How to activate/desactivate design variables and how to change the targets.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind.inversion import InverseFrequentialResponse

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics)

# In this example in which two holes are optimized it is presented how to activate
# desactivate design variables and how to change the targets


frequencies = np.linspace(50, 500, 100)
temperature = 20
losses = True

# %% Targets definitions
# For this example we use simulated data
# The geometry is 0.5m conical part with 2 side holes.
geom = [[0, 0.5, 2e-3, 10e-3, 'linear']]
target_hole = [['label', 'position', 'radius', 'chimney'],
               ['hole1', .25, 3e-3, 5e-3],
               ['hole2', .35, 4e-3, 7e-3]]
fingerings = [['label', 'A', 'B', 'C', 'D'],
              ['hole1', 'x', 'x', 'o', 'o'],
              ['hole2', 'x', 'o', 'x', 'o']]
noise_ratio = 0.01


target_computation = ImpedanceComputation(frequencies, geom, target_hole,
                                          fingerings,
                                          temperature=temperature,
                                          losses=losses)
notes = target_computation.get_all_notes()

Ztargets = list()
for note in notes:
    target_computation.set_note(note)
    # normalize and noised impedance
    Ztargets.append(target_computation.impedance/target_computation.Zc
                    * (1 + noise_ratio*np.random.randn(len(frequencies))))

# %% Construcion of the inverse problem

# Here we want to adjust:
# - the main bore length and conicity
# - the holes location and radius

inverse_geom = [[0, '0.05<~0.3', 2e-3, '0<~2e-3', 'linear']]
inverse_hole = [['label', 'position', 'radius', 'chimney'],
                ['hole1', '~0.1%', '~1.75e-3%', 5e-3],
                ['hole2', '~0.2%', '~1.75e-3%', 7e-3]]


instru_geom = InstrumentGeometry(inverse_geom, inverse_hole, fingerings)
print(instru_geom.optim_params)

# We can compare the two bore at the initial state
fig_geom = plt.figure()
target_computation.plot_instrument_geometry(figure=fig_geom, label='Target',
                                   color='black')
instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Initial Geometry')
fig_geom.get_axes()[0].legend()

instru_phy = InstrumentPhysics(instru_geom, temperature, Player(), losses)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztargets,
                                     notes=notes)

# %% fix the target

# by default:
# - all the notes and target are taken into account
# - all the design variables are optimized together

# it can be smart to adjust first the main bore geometry by taking into
# account only the 'A' for which all the holes are closed
# active only the main bore design variables
print("\n*Main Bore*")
instru_geom.optim_params.set_active_parameters([0, 1])
print(instru_geom.optim_params)

# Include only the 'A' and the corresponding target
inverse.set_targets_list(Ztargets[0], notes[0])

# we perform the optimization
inverse.optimize_freq_model(iter_detailed=True)

# and now, the hole 2 location on 'B' for which it is the only one open hole
print("\n*Hole 2*")
instru_geom.optim_params.set_active_parameters(4)
inverse.set_targets_list(Ztargets[1], notes[1])
inverse.optimize_freq_model(iter_detailed=True)

# then, the hole 1 location on 'C' for which it is the only one open hole
print("\n*Hole 1*")
instru_geom.optim_params.set_active_parameters(2)
inverse.set_targets_list(Ztargets[2], notes[2])
inverse.optimize_freq_model(iter_detailed=True)

# %% Include everything

# We finally re-active all the design variables
print("\n*All*")
instru_geom.optim_params.set_active_parameters('all')
print(instru_geom.optim_params)

# and all the notes and the target impedances
inverse.set_targets_list(Ztargets, notes)

inverse.optimize_freq_model(iter_detailed=True)


instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Final Geometry',
                                    linestyle=':')
fig_geom.get_axes()[0].legend()

plt.show()
