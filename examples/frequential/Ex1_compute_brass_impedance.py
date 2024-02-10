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
How to easily compute impedance of instrument without tone holes.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation

# %% Basic computation

# Frequencies of interest: 20Hz to 2kHz by steps of 1Hz
fs = np.arange(20, 2000, 1)
geom_filename = 'Geom_trumpet.txt'

# Find file 'trumpet' describing the bore, and compute its impedance
result = ImpedanceComputation(fs, geom_filename)

# Plot the instrument geometry
result.plot_instrument_geometry()

# you can get the characteristic impedance at the entrance of the instrument
# which can be useful to normalize the impedance
Zc = result.Zc

# You can plot the impedance which is automatically normalized
fig = plt.figure()
result.plot_impedance(figure=fig, label='my label')
# here the option 'figure' specify on which window plot the impedance

# %% other useful features

# you can modify the frequency axis without redoing everything
freq_bis = np.arange(20, 2000, 100)
result.recompute_impedance_at(freq_bis)
result.plot_impedance(figure=fig, label='few frequencies', marker='o', linestyle='')
# you can use any matplotlib keyword!

# you can print the computed impedance in a file.
# It is automatically normalized by Zc
result.write_impedance('computed_impedance.txt')

plt.show()
