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
Plot the instrument
"""

import matplotlib.pyplot as plt

from openwind import InstrumentGeometry

mm = InstrumentGeometry('simplified-trumpet')
fig = plt.figure(figsize=(4,2))
mm.plot_InstrumentGeometry(figure=fig, color='k')
ax, = fig.get_axes()
ax.axis('auto')
ax.set_ylim(0, 7e-2)
ax.vlines(0.0, 0.0, 6e-3, linestyle='--', color='gray')
ax.vlines(0.716, 0.0, 6e-3, linestyle='--', color='gray');
ax.vlines(1.335, 0.0, 6e-2, linestyle='--', color='gray');
ax.grid('both')

fig.tight_layout()
fig.savefig('Figure3.pdf')
