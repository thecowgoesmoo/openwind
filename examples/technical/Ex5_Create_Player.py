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
This example shows how to create your Player
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import Player
from openwind.technical.temporal_curves import constant_with_initial_ramp

# In order to perform a temporal computation in Openwind, you will have to
# create a player.
#
# A player consists in a couple Excitator / Score
#
# Excitators are initiated from excitator parameters dictionnaries which you can
# find in openwind/technical/default_excitator_parameters. The default one is
# UNITARY_FLOW, which looks like this:
#.. code-block:: python
#
#  UNITARY_FLOW = {
#        "excitator_type":"Flow",
#        "input_flow":1
#  }
#
# You can see that the excitator is a Flow, and it's value is constant and
# equal to one.
#
# It is important to understand that you can choose between several kinds of
# excitators (Flow and Reed1dof at the moment), and according to the excitator type
# you will have different excitator parameters. Parameters are time dependant
# functions (or curves) or constant values. For a Flow, you will only have
# "input_flow" as a parameter. For a Reed1dof, you will have "opening",
# "mass", "section", "pulsation", "dissip", "width", "mouth_pressure", "model",
# "contact_pulsation" and "contact_exponent". You can have a glimpse at the code
# here: openwind/continuous/excitator.py
#
#
# A Score is defined by a list of note_events and a transition_duration.
# - note_events are tuples with the note name and the starting time of the note
# - transition_duration is a float which give the duration between two notes
#
# Now, let's see how we create a player
#
#
# Starting by creating a default player
player = Player()

# player's Excitator is a "UNITARY_FLOW", and it's Score is empty


# 1. PLAYER MODIFICATION

# You can now change the value of the input_flow of the score
player.update_curve("input_flow",2*np.pi*3700)

# Or you can create a custom_flow dictionnary and update the player with it
custom_flow = {
    "excitator_type":"Flow",
    "input_flow": constant_with_initial_ramp(2000, 2e-2)
}
player.update_curves(custom_flow)
# pay attention to the "s" at the end of update_curves, it is not the same
# method as above

# you can check the new value of your input_flow for t =[-5,5] :
time_interval = np.linspace(-5,5,1000)
player.plot_one_control("input_flow",time_interval)

# Of course, you can update your player with all excitator dictionnaries
# that are stored in default_excitator_parameters:
player.set_defaults("IMPULSE_400us")


# IMPORTANT NOTE: if your player was created with a Flow excitator, you can not
# change it to another type of excitator. This is forbidden to prevent misusage
# of the code. If you want to use a Reed1dof instead of a Flow, you must create a
# new Player

# Let's say we want to have a player that plays Oboe:
oboe_player = Player("OBOE")

# oboe_player is using this Excitator :
# OBOE = {
#     "excitator_type" : "Reed1dof",
#     "opening" : 8.9e-5,
#     "mass" : 7.1e-4,
#     "section" : 4.5e-5,
#     "pulsation" : 2*np.pi*600,
#     "dissip" : 0.4*2*np.pi*600,
#     "width" : 9e-3,
#     "mouth_pressure" : constant_with_initial_ramp(12000, 2e-2),
#     "model" : "inwards",
#     "contact_pulsation": 316,
#     "contact_exponent": 4
# }

# Let's say you want to change the mouth pressure value, once again :
oboe_player.update_curve("mouth_pressure",
                         constant_with_initial_ramp(13000, 2e-2))

# You can plot all controls for the oboe_player :
oboe_player.plot_controls(time_interval)
# At some point, if you got lost with your player, you can check which
# default dictionnaries availables

oboe_player.print_defaults()
oboe_player.set_defaults("WOODWIND_REED")

# ---------------------------------------------------------------------------- #

# 2. SCORE MODIFICATION

# If you want to modify your Score, first create a new note_events list :
note_events = [('note1', .02), ('note2', .03), ('note1', .04)]
# Then you can change the transition_duration:
transition_duration = 1e-3

oboe_player.update_score(note_events, transition_duration)

plt.show()
