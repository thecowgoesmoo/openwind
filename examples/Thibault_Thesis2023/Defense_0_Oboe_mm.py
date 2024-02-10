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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from openwind import ImpedanceComputation, InstrumentGeometry

fs = np.arange(20, 2000, 1)
temp = 25
# %% Mass due to cross section discontinuity

# We chose an instrument with a cross section discontinuity
geom = '../frequential/Oboe_instrument.txt'
holes = '../frequential/Oboe_holes.txt'
fing_chart = '../frequential/Oboe_fingering_chart.txt'

instru_geom = InstrumentGeometry(geom, holes)

fig = plt.figure(figsize=(4*5,2.6))
instru_geom.plot_InstrumentGeometry(figure=fig, double_plot=False, color="k")
plt.grid(True, 'major')
plt.minorticks_on()
plt.grid(True, 'minor', linewidth=0.3)
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_major_formatter('')
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_major_formatter('')
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.savefig("Oboe_mm.png", dpi=300)
# plt.axis('auto')
# there is a discontinuity at 0.45m before the "bell"
