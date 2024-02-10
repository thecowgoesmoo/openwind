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
How to compute impedances of instrument with side holes and so several
fingerings.
"""

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation

t0 = perf_counter()

# %% Basic computation

# Frequencies of interest: 20Hz to 2kHz by steps of 1Hz
fs = np.arange(20, 2000, 1)
temperature = 25

# The three files describing the geometry and the
geom = 'Geom_trumpet.txt'
holes = 'Geom_holes.txt'
fing_chart = 'Fingering_chart.txt'
# Find file 'trumpet' describing the bore, and compute its impedance
result = ImpedanceComputation(fs, geom, holes, fing_chart, temperature=temperature)
result.technical_infos()

# Plot the instrument geometry
result.plot_instrument_geometry()

# Plot the impedance
result.plot_impedance(label='Default Fingering: all open')
plt.suptitle('Default Fingering: all open')
# without indication the impedance computed correspond to the one with all holes open

# %% Chose the fingering

# it is possible to fix the fingering when the object `ImpedanceComputation`
# is created with the option `note`
result_note = ImpedanceComputation(fs, geom, holes, fing_chart,
                                   temperature=temperature, note='A')

result_note.plot_impedance(label='A')


# or to modify it after the instanciation
fig = plt.figure()
notes = result_note.get_all_notes()
for note in notes:
    result_note.set_note(note)
    result_note.plot_impedance(figure=fig, label=note)

plt.show()


#%%

t1 = perf_counter()

print(t1-t0)

