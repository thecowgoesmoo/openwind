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
This file illustrates how to update the controle parameters for a time
domain simulation, and how to use the low-level of the time domain
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import InstrumentGeometry, InstrumentPhysics, TemporalSolver, Player
from openwind.temporal import RecordingDevice
from openwind.technical.temporal_curves import ADSR


# %% "Low level" implementation of scaled reed model

# This time we will use the scaled models and so the dimensionless parameters.
# We build a dictionnary with the intersting fields then instanciate the `Player`.
# This is also possible to use the default dict given in :py:mod:`default_excitator_parameters<openwind.technical.default_excitator_parameters>`.

gamma_amp = 0.45 # the amplitude of gamma, the dimensionless supply pressure
transition_time = 2e-2 # the characteristic time of the time eveolution of gamma
gamma_time = ADSR(0, 0.4, gamma_amp, transition_time, transition_time, 1, transition_time) # the time evolution of gamma
zeta = 0.35 # the value of zeta, the "reed" opening dimensionless paramters
dimless_reed = {"excitator_type" : "Reed1dof_scaled",
                "gamma" : gamma_time,
                "zeta": zeta,
                "kappa": 0.35,
                "pulsation" : 2*np.pi*2700, #in rad/s
                "qfactor": 6,
                "model" : "inwards",
                "contact_stifness": 1e4,
                "contact_exponent": 4,
                "opening" : 5e-4, #in m
                "closing_pressure": 5e3 #in Pa
                }
reed_player = Player(dimless_reed)

# We instanciate the other objects necessary to compute the sound
instrument = [[0.0, 5e-3],
              [0.5, 5e-3]]
my_geom = InstrumentGeometry(instrument) # the geometry of the instrument
temperature = 25
my_phy = InstrumentPhysics(my_geom, temperature, reed_player, 'diffrepr')
my_temp_solver = TemporalSolver(my_phy)
rec = RecordingDevice()

# we can now compute the sound for the indicated control parameters
my_temp_solver.run_simulation(0.5, callback=rec.callback)

# we extract and plot the reed displacement
y_reed = rec.values['source_y']
time = rec.ts

plt.figure()
plt.plot(time, y_reed, label=f'zeta={zeta}')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Reed displacement [m]')

# %% Change the value of constant control parameters

# We can recompute the sound for different control parameters without redoing everything
# We first modify the value of the control parameters in the Player bject then only restart the time simulation.
zeta_list = [0.3, 0.4, 0.5]
for zeta in zeta_list:
    reed_player.update_curve('zeta', zeta)
    rec = RecordingDevice()
    my_temp_solver.reset()
    my_temp_solver.run_simulation(0.5, callback=rec.callback)
    y_reed = rec.values['source_y']
    time = rec.ts
    plt.plot(time, y_reed, label=f'zeta={zeta}')

plt.legend()

# %% Change the value of time varying control parameters

# **Gamma**, the dimensionless parameters linked to the supply pressure is the only one parameters which can vary with thime.
# It is necessary to reinstanciate a new time varying function
gamma_amp_list =  [0.3, 0.4, 0.5]

plt.figure()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Reed displacement [m]')

for gamma_amp in gamma_amp_list:
    gamma_time = ADSR(0, 0.4, gamma_amp, transition_time, transition_time, 1, transition_time)
    reed_player.update_curve('gamma', gamma_time)
    rec = RecordingDevice()
    my_temp_solver.reset()
    my_temp_solver.run_simulation(0.5, callback=rec.callback)
    y_reed = rec.values['source_y']
    time = rec.ts
    plt.plot(time, y_reed, label=f'gamma={gamma_amp}')

plt.legend()
plt.show()
