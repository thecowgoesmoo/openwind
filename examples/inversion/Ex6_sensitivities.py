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
This example present how to compute sensitivities.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind.inversion import InverseFrequentialResponse

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics)


# It is possible to observe the sensitivity of the observable with respect
# to any design variables for each fingering.

# We use again the problem of ex.4
frequencies = np.arange(100, 1000, 1)
temperature = 20
losses = True

# Targets definitions
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

#  Construcion of the inverse problem
inverse_geom = [[0, '~0.5', 2e-3, '~10e-3', 'linear']]
inverse_hole = [['label', 'position', 'radius', 'chimney'],
                ['hole1', '~.25', '~3e-3', 5e-3],
                ['hole2', '~.35', '~4e-3', 7e-3]]
# The sensitivities being computed w.r. to the optimized parameters (and not
# necessary the geometric ones) it is better to chose them equal here

instru_geom = InstrumentGeometry(inverse_geom, inverse_hole, fingerings)
instru_phy = InstrumentPhysics(instru_geom, temperature, Player(), losses)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztargets,
                                     notes=notes)

# %% Sensitivity computation
sensitivities, _ = inverse.compute_sensitivity_observable()
inverse.plot_sensitivities()

# it can be preferable to observe sensitivities for each variable type separatly
print(instru_geom.optim_params)
loc_indices = [0, 2, 4]
rad_indices = [1, 3, 5]

instru_geom.optim_params.set_active_parameters(loc_indices)
sens_loc, _ = inverse.compute_sensitivity_observable()
inverse.plot_sensitivities(logscale=True, param_order=[1, 2, 0],
                           text_on_map=False)
# the 'param_order' option reorganize the parameters on the plot
plt.suptitle('Sensitivities w.r. to locations')

instru_geom.optim_params.set_active_parameters(rad_indices)
sens_rad, _ = inverse.compute_sensitivity_observable()
inverse.plot_sensitivities(logscale=True, param_order=[1, 2, 0],
                           text_on_map=False)
plt.suptitle('Sensitivities w.r. to radii')

# until now all the frequency range was taking into account. It is possible to
# window it for each fingering
f_notes = 440*2**(np.array([0, 2, 3, 5])/12)
windows = [(f0, 10) for f0 in list(f_notes)]
# the first value of the tuple indicate the center of the window and the second
# the width in cents
sens_rad_wind, _ = inverse.compute_sensitivity_observable(windows=windows)
inverse.plot_sensitivities(logscale=True, param_order=[1, 2, 0],
                           text_on_map=False)
plt.suptitle('Windowed Sensitivities w.r. to radii')

plt.show()
