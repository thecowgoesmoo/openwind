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
This example illustrates how to compute the impedance (admittance) of flute-like
instruments and their relative specificities.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation, Player
from openwind.continuous import radiation_model
from openwind.impedance_tools import plot_impedance

# %% Instrument definition

Rp = 5e-3
geom = [[0, 0.5, Rp, Rp, "cone"]]
temperature = 25
freq = np.arange(100, 1000, 2)
my_player = Player("FLUTE")

# Let's imagine a simple cylindrical flute 50cm long with a radius Rp. As
# flute-like instruments are open-open, it is necessary to indicate this specificity
# and to give some supplementary information on the radiation condition at the entrance.
# This is done by defining a `Player`, here a default "flute" player.
#
# In admittance computation of flute-like instruments, what lies on either side
# of the edge must be treated differently.
# In Openwind, everything below the edge must be included in the "main bore" geometry,
# and everything above it (usually just radiation) is treated through the "Player".
#
# Most of the control parameters specified in the `Player` are used only for
# sound simulation, but they include a "radiation_category" and the cross-section
# area of the embouchure opening (or window), which can be different from the
# one on the main pipe. These two quantities are used to compute the radiation
# impedance and can be modified a posteriori using the update_curve method.


Rw = 4e-3  # the equivalent radius of the window
my_player.update_curve("radiation_category", "infinite_flanged")
my_player.update_curve("section", np.pi * Rw**2)


# %% Impedance computation

# The impedance can be computed similarly than for other instrument, but by
# specifying the "player". As flute-like instruments are open-open, we prefer to
# plot the admittance (inverse of impedance).

result_flute = ImpedanceComputation(freq, geom, player=my_player, temperature=temperature)

Y_fig = plt.figure()
result_flute.plot_admittance(figure=Y_fig, label="Flute: flanged radiation")

# %% Comparison with detailed computation

# This admittance corresponds to two admittances in parrallel (or impedance
# in series): the input impedance of the pipe and the radiation impedance.
# This can be verified numerically:

result_pipe = ImpedanceComputation(freq, geom, temperature=temperature)
Zp = result_pipe.impedance  # response of the pipe alone

# phy. quantities for rad impedance computation
rho, celerity = result_pipe.get_entry_coefs("rho", "c")
my_rad = radiation_model("infinite_flanged")
Zw = my_rad.get_impedance(2*np.pi*freq, Rw, rho, celerity, opening_factor=1)

Ztot = Zp + Zw
Zc_w = rho * celerity / (np.pi * Rw**2)

plot_impedance(freq, Ztot, Zc0=Zc_w, figure=Y_fig, label="Zw + Zp",
               linestyle="--", admittance=True)  # option to plot admittance instead of impedance

errZ = np.linalg.norm(Ztot - result_flute.impedance) / np.linalg.norm(Ztot)
print(f"\nRelative error between direct compuation and 1/(Zp + Zw): {errZ:.2e}\n")

# %% More complex window geometry

# The embouchure opening can have a very complex geometry, especially for recorder.
# It is possible to include some aspects of this geometry in the radiation impedance.
# To do so, the radiation category "window" must be used and the edge angle
# and the wall thickness above the edge must be specified. The admittance
# is slightly modified.

my_recorder = Player("FLUTE")
new_params = {"section": np.pi * Rw**2,
              "radiation_category": "window",
              "edge_angle": 15,  # the edge angle in degree
              "wall_thickness": 5e-3,# the wall thickness in meter
              }
my_recorder.update_curves(new_params)  # set the new control parameters

result_recorder = ImpedanceComputation(freq, geom, player=my_recorder,
                                       temperature=temperature)
result_recorder.plot_admittance(figure=Y_fig, label="Recorder with window rad")

plt.show()

# %% Transverse flute

# For the transverse flute, it is possible to indicate that the acoustic source
# is located at a side opening. In this case, the embouchure hole must be
# included in the side components like the tone holes.
# Again, the entire chimney tube below the edge should be indicated as the
# chimney length of the hole.

Remb_out = 3e-3
emb_label = 'embouchure'
side_holes = [['label',     'position', 'chimney',  'radius',   'radius_out'],
              [emb_label,   0.03,      5e-3,        5e-3,       Remb_out], # the embouchure hole
               ['hole1',     0.15,       3e-3,       4e-3,       4e-3],
               ['hole2',     0.20,       3e-3,       4e-3,       4e-3],
               ['hole3',     0.30,       3e-3,       4e-3,       4e-3],
              ]

# This hole should be excluded from the fingering chart. A warning will be printed
# because the fingering chart is analyzed before knowing that it is a special hole.
# However, the "entrance" of the pipe (where the cork is placed) can now be
# included in the fingering grid (which can be useful for other transverse instruments)

fing_chart = [['label',     'A', 'B', 'C'],
              ['entrance',  'x', '0.5', 'x'],
              ['hole1',     'x', 'x', 'x'],
              ['hole2',     'x', 'x', 'o'],
              ['hole3',     'x', 'x', 'o'],
              ]

# The source location must be indicated in `ImpedanceComputation` or `InstrumentPhysics`
# with the keyword "source_location" with the label of the desired side hole.

player_trans = Player("FLUTE")
player_trans.update_curve("radiation_category", "infinite_flanged")
player_trans.update_curve("section", np.pi * Remb_out**2)

transverse_flute = ImpedanceComputation(freq, geom, side_holes, fing_chart,
                                        player=player_trans,
                                        note='A',
                                        temperature=temperature,
                                        source_location=emb_label, # this option specify that the acoustic source (the observation point) is located at the "embouchure" hole
                                        )

fig_trans = plt.figure()
transverse_flute.plot_admittance(figure=fig_trans, label='A - entrance (cork) fully closed')
transverse_flute.set_note('B')
transverse_flute.plot_admittance(figure=fig_trans, label='B - entrance semi-closed')

# it is also possible to specify that the "entrance " (cork) is closed by using
# the keyword "radiation_category"
fing_chart2 = [['label',     'A', 'B', 'C'],
              ['hole1',     'x', 'x', 'x'],
              ['hole2',     'x', 'x', 'o'],
              ['hole3',     'x', 'x', 'o'],
              ]

transverse_flute2 = ImpedanceComputation(freq, geom, side_holes, fing_chart2,
                                        player=player_trans,
                                        note='A',
                                        temperature=temperature,
                                        source_location=emb_label,
                                        radiation_category={'entrance':'closed', 'bell':'unflanged', 'holes':'unflanged'}
                                        )

transverse_flute2.plot_admittance(figure=fig_trans, linestyle='--',
                                  label='A - entrance (cork) fully closed using radiation condition')

plt.show()
