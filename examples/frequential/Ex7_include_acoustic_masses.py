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
Present the option relative to inclusion or not of acoustic masses
(discontinuity and matching volume).
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation, InstrumentGeometry

fs = np.arange(20, 2000, 1)
temp = 25
# %% Mass due to cross section discontinuity

# We chose an instrument with a cross section discontinuity
geom = 'Oboe_instrument.txt'
holes = 'Oboe_holes.txt'
fing_chart = 'Oboe_fingering_chart.txt'

instru_geom = InstrumentGeometry(geom, holes)
instru_geom.plot_InstrumentGeometry()
# there is a discontinuity at 0.45m before the "bell"

# It is possible to chose to include or not the supplementary acoustic mass
# due to this discontinuity (by default it is included)
result_with_masses = ImpedanceComputation(fs, geom, holes, fing_chart,
                                          note='C', temperature=temp,
                                          discontinuity_mass=True)

fig = plt.figure()
result_with_masses.plot_impedance(figure=fig, label='with discontinuity mass')

# or to exclude it
result_wo_masses = ImpedanceComputation(fs, geom, holes, fing_chart,
                                        note='C',temperature=temp,
                                        discontinuity_mass=False)
result_wo_masses.plot_impedance(figure=fig, label='without discontinuity mass')


# %% Matching Volume

# it is possible to include the masses due to the matching volume between the
# circular pipe of  the main bore and the circular pipe of the side hole

# by default these masses are excluded, it can be including throug the keyword
# 'matching_volume'
result_with_matching_volume = ImpedanceComputation(fs, geom, holes, fing_chart,
                                                 note='C', temperature=temp,
                                                 matching_volume=True)
result_with_matching_volume.plot_impedance(figure=fig, label='with matching volume')

plt.show()
