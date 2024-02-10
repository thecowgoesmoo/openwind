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
Simulate impulse response of a cylinder and export it to wav.
"""

from openwind import Player, simulate
from openwind.temporal.utils import export_mono

# 20cm cylinder of radius 5mm
instrument = [[0.0, 5e-3],
              [0.2, 5e-3]]

# The input signal is a flow impulse at the entrance of the tube.
# The impulse lasts 400µs.
player = Player('IMPULSE_400us')

duration = 0.2  # simulation time in seconds

""" Run the simulation
the user must provide the duration, the instrument (and optionnaly its holes
and fingering chart).
the user can specify some other paramteres as :
the player (default is an impulse flow)
the type of pipe visco-thermal losses (default is none)
the temperature inside the instrument (default is 25°C)
the radiation condition at the bell and open holes (default is 'unflanged')
the spatial discretization
and many more, see openwind.temporal_simulation.py
"""
rec = simulate(duration,
               instrument,
               # use the pre-instanciated player
               # (if not given, the source is an impulse flow)
               player=player,
               # Use diffusive representation of boundary layer losses
               # (if not given, default will be False)
               losses='diffrepr',
               # Assume a temperature of 20°C
               temperature=20,
               # Finite elements discretization parameters
               # (if not given, a default discretization will be made)
               l_ele=0.01, order=4
               )

# Export the signal that is radiated at the exit
signal = rec.values['bell_radiation_pressure']
export_mono('Ex1_cylinder_impulse_response.wav', signal, rec.ts)
