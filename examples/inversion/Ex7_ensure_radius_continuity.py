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
This example present how to ensure to keep the radius continuity of the main bore
during an inversion, or inversely to give the possibility to have a radius jump.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind.inversion import InverseFrequentialResponse

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics)


plt.close('all')


# %% Global options

frequencies = np.linspace(100, 500, 10)
temperature = 20
losses = True
# %% Targets definitions
# For this example we use simulated data instead of measurement

# The geometry is composed of 2 cones, with a discontinuity of radius at the junction

target_geom = [[0, 0.25, 2e-3, 3e-3, 'linear'],
               [0.25, .5, 3.2e-3, 7e-3, 'linear']]
target_computation = ImpedanceComputation(frequencies, target_geom,
                                          temperature=temperature,
                                          losses=losses)

# The impedance used in target must be normalized
Ztarget = target_computation.impedance/target_computation.Zc

# noise is added to simulate measurement
noise_ratio = 0.01
Ztarget = Ztarget*(1 + noise_ratio*np.random.randn(len(Ztarget)))

# %% Ensure radius continuity

# We would like to find the geometry without discontinuity of section which fit
# the target impedance
geom_continuous = [[0, 0.25, 2e-3, '~4e-3', 'linear'],
               [0.25, .5, '~4e-3', 7e-3, 'linear']]
instru_geom_con = InstrumentGeometry(geom_continuous)
print(instru_geom_con.optim_params)

# Here the exact same initial value is indicated for the right end radius of the first
# cone and the left end radius of the second pipe.
# You can notice that they are treated as a unique design parameter

# this Ensure that during the optimization these two raddi will be always equal

# Inverse problem
con_phy = InstrumentPhysics(instru_geom_con, temperature, Player(), losses)
inverse_con = InverseFrequentialResponse(con_phy, frequencies, Ztarget)
result_con = inverse_con.optimize_freq_model(iter_detailed=True)

print(instru_geom_con.optim_params)
# The optimization process stops at a value in between the two 3mm and 3.2mm

# %% Give the possibility to have a discontinuity

# If we want to authorize the discontinuity, the two initial value must different,
# of at least 0.001% (1e-5) or 1e-5mm
geom_disccontinuous = [[0, 0.25, 2e-3, '~4e-3', 'linear'],
                       [0.25, .5, '~4.0001e-3', 7e-3, 'linear']]
instru_geom_disc = InstrumentGeometry(geom_disccontinuous)
print(instru_geom_disc.optim_params)
# This time two different design variables are isntanciated

# Inverse problem
disc_phy = InstrumentPhysics(instru_geom_disc, temperature, Player(), losses)
inverse_disc = InverseFrequentialResponse(disc_phy, frequencies, Ztarget)
result_disc = inverse_disc.optimize_freq_model(iter_detailed=True)

print(instru_geom_disc.optim_params)

# Now the optimization process converge to 3mm and 3.2mm
