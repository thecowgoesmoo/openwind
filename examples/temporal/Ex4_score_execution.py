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
Create a score, modify it and run temporal simulations on a cylindrical
instrument. Low level classes are used.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import (InstrumentGeometry, Player, InstrumentPhysics,
                      TemporalSolver)
from openwind.technical import Score
from openwind.temporal import ExecuteScore, RecordingDevice



# a simple instrument with one hole ...
geom = [[0, 0.5, 2e-3, 10e-3, 'linear']]
hole = [['label', 'position', 'radius', 'chimney'],
        ['hole1', .25, 3e-3, 5e-3]]

# ... and 2 fingerings
fingerings = [['label', 'note1', 'note2'],
              ['hole1', 'o', 'x']]

instrument = InstrumentGeometry(geom, hole, fingerings)
# the default player is a impulse flow
player = Player('CLARINET')
instrument_physics = InstrumentPhysics(instrument, 20, player, False)
temporalsolver = TemporalSolver(instrument_physics, l_ele=0.01,
                                   order=4)
# %% low level instanciation
# ExecuteScore makes the link between a score (list of notes) and
# and instrument and its fingering
score_execution = ExecuteScore(instrument.fingering_chart,
                               temporalsolver.t_components)
no_note_events = []
# Set a score based on this empty list of notes
no_note_score = Score(no_note_events)
# set_score allows to modify the score with a series of notes
score_execution.set_score(no_note_score)
# set_fingering takes a time t (here 10) and sets the correct fingering
# according to the given notes series
score_execution.set_fingering(10)

# a list of notes and their time of beginning
note_events = [('note1', .02), ('note2', .03), ('note1', .04)]
# the second parameter is the transition duration between notes (here 1e-3)
with_note_score = Score(note_events, 1e-3)
# display the score along time
time = np.linspace(0,0.1,1000)
with_note_score.plot_score(time)
# change the score of the score_execution instance
score_execution.set_score(with_note_score)
score_execution.set_fingering(1.5)

# %% Run simulation!
# the player is updated with the empty score no_note_events
player.update_score(no_note_events)
# run a temporal simulation with a duration (here .1)
temporalsolver.run_simulation(.1)

#%% Run simulation and record output signals
# the player is updated with the new score note_events
player.update_score(note_events, 1e-3)
player.plot_controls(time)
# the output will be stored in a Recording device
rec = RecordingDevice(record_energy=False)
# run the simulation with a duration (here .1) and a callback class
temporalsolver.run_simulation(.1, callback=rec.callback)
rec.stop_recording()
# plot the output value of pressure at the bell
output_bell = rec.values['bell_radiation_pressure']
plt.figure()
plt.plot(output_bell)

plt.show()

# %% an error message when the asked notes are not in the fingering chart
strange_note = [('Do', .02), ('Re', .03), ('E', .04)]
try:
    player.update_score(strange_note, 1e-3)
    temporalsolver.run_simulation(.1)
except Exception as e:
    print(f'The printed error is:\n***\n{e}\n***')
