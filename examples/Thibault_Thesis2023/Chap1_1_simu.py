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

The numerical schemes implemented in openwind are used to simulate a simple
instrument with one side hole.

This script generates Figures 1.1, 1.2, 1.3a and 1.3b of the thesis:
    Figure 1 : Geometry of the instrument
    Figure 2 : Time evolution of the pressure signal at the bell
    Figure 3a: Energy exchanges during the simulation
    Figure 3b: Error on the energy balance
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import Player, simulate, InstrumentGeometry
from openwind.temporal.utils import export_mono

# %% Define the geometry of the simplified instrument

instrument = [[0.0, 300e-3, 5e-3, 5e-3, 'linear'],
              [300e-3, 500e-3, 5e-3, 50e-3, 'bessel', 0.7]]
holes = [['x', 'l', 'r', 'label'],
         [450e-3, 15e-3, 5e-3, 'hole1']]

# %% Figure 1: Geometry of the instrument

# Plot the geometry of the instrument
fig = plt.figure(figsize=(5, 2))
ig = InstrumentGeometry(instrument, holes)
ig.plot_InstrumentGeometry(figure=fig, double_plot=False)
plt.tight_layout()
plt.minorticks_on()
plt.grid(True, 'minor', alpha=0.3)
plt.grid(True, 'major')

# Save the figure to a .png image
plt.savefig("simple_instrument_Fig1.png", dpi=300)

# %% Run the simulation for a "converged" result (err < 4e-6 on the first 0.02 s)

# Select reed parameters
player = Player('CLARINET')
# Parameters of the reed can be changed manually
# Available parameters are:
# "opening", "mass","section","pulsation","dissip","width",
# "mouth_pressure","model","contact_pulsation","contact_exponent"
player.update_curve("width", 2e-2)

duration = 0.2   # simulation time in seconds
rec = simulate(duration,
               instrument,
               holes,
               player=player,
               losses=False,  # no viscothermal losses
               temperature=20,
               l_ele=0.01, order=4,  # Discretization parameters
               record_energy=True,  # enable calculation of energy for Fig. 1.3
               verbosity=2,  # show the discretization infos
               )

# Extract the signal recorded at the bell
signal = rec.values['bell_radiation_pressure']
# Write it to an audio file
export_mono('simple_instrument_converged.wav', signal, rec.ts)

# %% Figure 2: time evolution of the pressure signal at the bell

plt.figure(figsize=(4, 2.6))
plt.plot(rec.ts, signal, linewidth=0.7)
plt.xlabel("Time $t$ (s)")
plt.ylabel("Pressure (Pa)")
plt.minorticks_on()
plt.grid(True, 'minor', alpha=0.3)
plt.grid(True, 'major')
plt.tight_layout()

# Save the figure to a .png image
plt.savefig("simple_instrument_Fig2.png", dpi=300)

# %% Figure 3: energy exchanges during the simulation
# (Fig. 1.3a in the thesis)

nrj = 0
source = 0 + rec.values["source_Q"]
dissip = 0

# Fetch all the recorded information concerning energy exchanges
for key in rec.values.keys():
    if key.endswith("_E"):
        print(key)
        nrj += rec.values[key]
    if key.endswith("_Q") and key != "source_Q":
        print(key)
        dissip += rec.values[key]

# Convert to milliJoules for ease of reading
nrj *= 1e3
source *= 1e3
dissip *= 1e3

plt.figure("Energy exchanges", figsize=(4, 2.6))
plt.plot(rec.ts, -np.cumsum(source), linewidth=0.8, label=r'$\int S(t)$')
plt.plot(rec.ts, nrj, ":", label=r'$\mathcal{E}(t)$')
plt.plot(rec.ts, np.cumsum(dissip), "--", label='$\int Q(t)$')
plt.legend()
plt.xlabel("Time $t$ (s)")
plt.ylabel("Energy (mJ)")
plt.minorticks_on()
plt.grid(True, 'minor', alpha=0.3)
plt.grid(True, 'major')
plt.tight_layout()

# Save the figure to a .png image
plt.savefig("simple_instrument_Fig3.png", dpi=300)

# %% Figure 4: error on the energy balance
# (Fig. 1.3b in the thesis)

# Compute the relative error on the energy balance at each time step
# nrj_ref = 2**int(np.log2(max(nrj)))
nrj_ref = max(nrj)
nrj_error_1 = ((nrj[1:] - nrj[:-1]) + (dissip[1:] + source[1:])) / nrj_ref
nrj_error = (nrj[1:] + (-nrj[:-1] + (dissip[1:] + source[1:]))) / nrj_ref

# Plot it
plt.figure(figsize=(4, 2.6))
plt.plot(rec.ts[:-1], nrj_error, '.', markersize=1)
plt.xlabel("Time $t$ (s)")
plt.ylabel("Relative error\non the energy balance")
plt.minorticks_on()
plt.grid(True, 'minor', alpha=0.3)
plt.grid(True, 'major')
plt.tight_layout()

# Save the figure to a .png image
plt.savefig("simple_instrument_Fig4.png", dpi=300)
