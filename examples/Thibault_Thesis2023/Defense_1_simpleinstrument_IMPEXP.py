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
This script is part of the numerical examples accompanying Alexis THIBAULT's
Ph.D. thesis.

Create an animation based on the results of a simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import Player, simulate, InstrumentGeometry
from openwind.technical.temporal_curves import gate
from openwind.temporal.utils import export_mono
from openwind.simu_anim import SimuAnimation

# %% Define the geometry of the simplified instrument

instrument = [[0.0, 300e-3, 5e-3, 5e-3, 'linear'],
              [300e-3, 500e-3, 5e-3, 50e-3, 'bessel', 0.7]]
holes = [['x', 'l', 'r', 'label'],
         [450e-3, 15e-3, 5e-3, 'hole1']]

ig = InstrumentGeometry(instrument, holes)

# %% Run the simulation for a "converged" result (err < 4e-6 on the first 0.02 s)

# Select reed parameters
player = Player('CLARINET')
# Parameters of the reed can be changed manually
# Available parameters are:
# "opening", "mass","section","pulsation","dissip","width",
# "mouth_pressure","model","contact_pulsation","contact_exponent"
player.update_curve("width", 2e-2)
t1 = 0.0 # Time when to start blowing
t2 = 0.9 # Time when to stop blowing
t_ramp = 2e-2 # Ramp time
pmax = 2000 # Maximal blowing pressure
player.update_curve("mouth_pressure", gate(t1, t1+t_ramp, t2-t_ramp, t2, a=pmax))

duration = 1.0   # simulation time in seconds
rec = simulate(duration,
               instrument,
               holes,
               player=player,
               losses=False,  # no viscothermal losses
               temperature=20,
               l_ele=0.01,
               order=4,  # Discretization parameters
               record_energy=False,
               verbosity=2,  # show the discretization infos
               interp_grid="original", # enable interpolation
               assembled_toneholes=True, # USE THE IMPEXP SCHEME
               )


#%%

anim = SimuAnimation(ig, rec, name="simu_IMPEXP")

anim.save_sound()
anim.save_geom()
anim.save_signal_plot()
anim.save_frames()
