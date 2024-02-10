#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2021, INRIA
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
How to simulate a simplified flute (recorder) in time domain.
This example demonstrates the use of a flute-based Player. It also shows how
to play several notes while adpating the jet velocity, and compute the change of
regimes relative to the increase of the jet velocity

See also
--------
openwind.continuous.excitator
openwind.technical.player
openwind.temporal.simulate
"""
import matplotlib.pyplot as plt
import numpy as np

from openwind import Player, simulate, ImpedanceComputation
from openwind.temporal.utils import export_mono
from openwind.technical.temporal_curves  import ramp, gate, add


plt.close('all')

# %% Instrument definition

# The instrument simulated is a cylinder, 30cm long, with 6.6mm radius (from
# soprano recorder geometry measured by Blanc-AAA-2010). One small hole
# positioned at 20cm, 7mm long, 3mm of radius, is added. And two fingerings
# (hole open and closed) are defined. The player used in the default "Soprano recorder".
# Its geometry has been taken from the study of Blanc.

temperature=25
instrument = [[0.0, 6.6e-3],
              [0.30, 6.6e-3]]
holes = [['x', 'l', 'r', 'label'],
          [0.2, 7e-3, 3e-3, 'hole1']]

fingerings = [['label', 'A', 'B'],
              ['hole1', 'x','o']]

player = Player('SOPRANO_RECORDER')

# %% Simple simulation

# These information are sufficient to perform a first sound simulation. By default
# without fingering indication, the hole is open.
# Here the simulation duration is 200ms. The jet velocity is kept as indicated
# in the default player: it stays constant at 25m/s.
#
# We can observe that the jet at the edge tip location oscillates. The oscillation
# grows exponentially then saturates.

duration = .2 # the simulation duration in second
rec_simple = simulate(duration,
                      instrument, holes, fingerings,
                      player = player,
                      losses='diffrepr',
                      temperature=temperature,
                      ) # the sound simulation


time = rec_simple.ts # extraction of simulation time
eta = rec_simple.values['source_y'] # extraction of the oscillation

plt.figure() # we plot the jet position w.r. to time
plt.plot(time*1e3, eta*1e3)
plt.grid()
plt.xlabel('Time [ms]')
plt.ylabel('Jet position at the edge tip [mm]')
plt.show()

# %% Play 2 notes at constant theta

# In flute-like instrument, the dimensionless velocity theta=Uj/(W f) is often
# used. In this quantity, f is the soundind frequency, which can be approximated
# by the frequency of resonance of the instrument. When the fingering is changed
# this frequency varies, and the jet velocity must be varied together if we want
# to keep theta constant.
#
# To do so we need first to estimate the frequency of the two notes, here 'A'
# and 'B'. We estimate them using frequency domain computation, then we determine
# the jet velocity for each fingering for a given theta


theta = 10 # the imposed value of theta
W = player.control_parameters['width']

fmax = 1.5e3
fig_imp = plt.figure()
imp_res = ImpedanceComputation(np.linspace(100, fmax, 1000),instrument, holes, fingerings,
                               player=player, temperature=temperature, note='A')
f_resA = imp_res.antiresonance_frequencies(1, display_warning=False)[0]
imp_res.plot_admittance(figure=fig_imp, label='A')

imp_res.set_note('B')
f_resB = imp_res.antiresonance_frequencies(1, display_warning=False)[0]
imp_res.plot_admittance(figure=fig_imp, label='B')
plt.show()

UjA = theta*W*f_resA
UjB = theta*W*f_resB

print(f"\n The frequencies of resonances are {f_resA:.2f}Hz and {f_resA:.2f}Hz, "
      f"giving the jet velocities: {UjA:.2f}m/s and {UjB:.2f}m/s. \n")

# We now define a score, with a transition between notes at the half of the
# simulation duration, and a jet velocity wich evolve respectively with this
# transition. This is done using the "gate" functions from the `temporal_curves`
# module. The model being ill-defined for low jet velocity (especially Uj=0)
# It is important to keep a relatively high velocity during all the simulation.

duration = 1  # simulation time in seconds
switch = duration/2

score = [('A',0), ('B',switch)] # the score
Uj = add(gate(-1e-2,1e-2, switch-1e-2,switch+1e-2, shape='cos', a=UjA),
         gate(switch-1e-2,switch+1e-2, duration-1e-2,duration+1e-2, shape='cos', a=UjB)
         )

# We can now update the player and perform the simulation. Then plot the results
# and export the sound.

player.update_curve("jet_velocity", Uj) # update the jet velocity
player.update_score(score) # update the score

rec = simulate(duration,
               instrument,
               holes, fingerings,
               player = player,
               losses='diffrepr',
               temperature=temperature,
               l_ele=0.01, order=4, # Discretization parameters
               nondim=True
               )

time_2notes = rec.ts # extract time
eta_2notes = rec.values['source_y'] # extract jet position
signal_2notes = rec.values['source_flow'] # extract ac. flow in the window

fig, ax = plt.subplots(2,1) # plot the figures
ax[0].plot(time_2notes, Uj(time_2notes))
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Jet velocity [m/s]')

ax[1].plot(time_2notes, 1e3*eta_2notes)
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Jet postiion [mm]')
plt.show()

sound_file_name_2notes =  f'Simu_flute_theta_{theta:.0f}_2notes.wav'
export_mono(sound_file_name_2notes, signal_2notes, time_2notes) # export the signal in a wav file.

# %% Bifurcation of flute-like instrument

# One specificity of flute like instruments is the evolution of pitch with
# the jet velocity. Especially, at some points, the instrument jump from
# a regime to another. This can be reproduced by the model, by using a linearly
# increasing jet velocity (ramp). However this necessitate to perform a long
# simulation.

duration = 15 #s
Uj = ramp(0, 10, duration, 50)
player.update_curve("jet_velocity", Uj)
player.update_score([('A',0)])



rec_ramp = simulate(duration,
               instrument,
               holes, fingerings,
               player = player,
               losses='diffrepr',
               temperature=temperature,
               l_ele=0.01, order=4, # Discretization parameters
               nondim=True
               )

time_ramp = rec_ramp.ts
eta_ramp = rec_ramp.values['source_y']
signal_ramp = rec_ramp.values['source_flow']


plt.figure()
plt.plot(time_ramp, eta_ramp*1e3, label='eta')
plt.xlabel('Time [s]')
plt.ylabel('\eta [mm]')
plt.show()

sound_file_name_ramp = 'Simu_flute_Uj_ramp_10-50ms_15s.wav'
export_mono(sound_file_name_ramp, signal_ramp, time_ramp)
