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
Simulate impulse response of two cylinders without and with wall roughness.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import simulate
from openwind.technical.player import Player

instrument = [[0.0, 5e-3],
              [0.2, 5e-3]]

# The input signal is a flow impulse at the entrance of the tube.
# The impulse lasts 400µs.
player = Player('IMPULSE_400us')

duration = 0.2  # simulation time in seconds

# Simulation 1 :
rec = simulate(duration,
               instrument,
               player=player,
               losses='diffrepr',
               temperature=20,
               l_ele=0.01, order=4
               )
#%%
rec2 = simulate(duration,
               instrument,
               player=player,
               # Custom diffusive representation model of losses : assume
               # a slight wall roughness
               losses='diffrepr tournesol',
               temperature=20,
               l_ele=0.01, order=4
               )

#%%

rec3 = simulate(duration,
               instrument,
               player=player,
               # Stronger wall roughness
               losses='diffrepr etoile',
               temperature=20,
               l_ele=0.01, order=4
               )


#%%

# Export the signal that is radiated at the exit
signal = rec.values['bell_radiation_pressure']
signal2 = rec2.values['bell_radiation_pressure']
signal3 = rec3.values['bell_radiation_pressure']
ts = rec.ts

plt.plot(ts, signal, label="diffrepr bessel")
plt.plot(ts, signal2, label="diffrepr tournesol")
plt.plot(ts, signal3, label="diffrepr etoile")
plt.xlabel("Time (s)")
plt.ylabel("Bell pressure")
plt.legend()
print("Max rel difference", np.max(abs(signal-signal2))/np.max(abs(signal)))

# export_mono('Ex1_cylinder_impulse_response.wav', signal, rec.ts)
