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
This example illustrates how to guarantee the side hole to be smaller that the main pipe.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind.inversion import InverseFrequentialResponse

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics)


plt.close('all')


# %% Global options

frequencies = np.linspace(50, 500, 100)
temperature = 20
losses = True

# %% Targets definitions
# For this example we use simulated data

# The geometry is 0.5m conical part with 1 side hole.
geom = [[0, 0.5, 2e-3, 10e-3, 'linear']]
target_hole = [['label', 'position', 'radius', 'chimney'],
               ['hole1', .25, 3e-3, 5e-3]]
fingerings = [['label', 'open', 'closed'],
              ['hole1', 'o', 'x']]
notes = ['open', 'closed']


noise_ratio = 0.01


target_computation = ImpedanceComputation(frequencies, geom, target_hole,
                                          fingerings, note=notes[0],
                                          temperature=temperature,
                                          losses=losses)
# The impedance used in target must be normalized
Zopen = target_computation.impedance/target_computation.Zc
# noise is added to simulate measurement
Zopen *= 1 + noise_ratio*np.random.randn(len(frequencies))


target_computation.set_note(notes[1])
Zclosed = target_computation.impedance/target_computation.Zc
Zclosed *= 1 + noise_ratio*np.random.randn(len(frequencies))

Ztarget = [Zopen, Zclosed]
# %% Definition of the optimized geometry

# Here we want to adjust only the hole location and radius

# During the optimization process we have to guarantee that:
# - the hole stays on the main bore (here its location is in [0, 0.5])
# - its radius stays smaller than the one of the main pipe at its location
# this can not be guarantee with boundaries!
inverse_hole = [['label', 'position', 'radius', 'chimney'],
                ['hole1', '0<~.1<.5', '~2e-3%', 5e-3]]
# By using '~2e-3%' the hole radius is defined as a ratio of the main bore
# radius at its location. This ratio is in [0,1].
# Similar notation can be used to define the location as a ratio of the length
# of the main bore pipe for the cases where both the hole location and the pipe
# length are optimized.

instru_geom = InstrumentGeometry(geom, inverse_hole, fingerings)
print(instru_geom.optim_params)
# We can see that the value adjusted by the algorithm for the radius does not
# correspond to the geometric value. It is the ratio.

# We can compare the two bore at the initial state
fig_geom = plt.figure()
target_computation.plot_instrument_geometry(figure=fig_geom, label='Target')
instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Initial Geometry')


# %% The optimization

instru_phy = InstrumentPhysics(instru_geom, temperature, Player(), losses)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget,
                                     notes=notes)

# Optimization process
result = inverse.optimize_freq_model(iter_detailed=True)


# and the geometry
instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Final Geometry')

print('='*30 + '\nCompare holes geometry')
print('Target Geometry')
print(target_computation.get_instrument_geometry().print_holes())
print('Optimization result:')
print(instru_geom.print_holes())

plt.show()
