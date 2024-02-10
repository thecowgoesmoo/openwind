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
How to simulate a simplified clarinet in time domain.

The simplified clarinet is a cylinder with a hole.
This example demonstrates the use of a reed-based Player.

See also
--------
openwind.continuous.excitator
openwind.technical.player
openwind.temporal.simulate
"""

from openwind import Player, simulate
from openwind.temporal.utils import export_mono

# 50cm cylinder
instrument = [[0.0, 5e-3],
              [0.5, 5e-3]]
# One small hole positioned at 45cm
# 1cm long, 2mm of radius, open by default.
holes = [['x', 'l', 'r', 'label'],
         [0.45, 0.01, 2e-3, 'hole1']]

player = Player('CLARINET')
# Parameters of the reed can be changed manually
# Available parameters are:
# "opening", "mass","section","pulsation","dissip","width",
# "mouth_pressure","model","contact_pulsation","contact_exponent"
player.update_curve("width", 2e-2)


duration = 0.2  # simulation time in seconds
rec = simulate(duration,
               instrument,
               holes,
               player = player,
               losses='diffrepr',
               temperature=20,
               l_ele=0.01, order=4 # Discretization parameters
               )
# show the discretization infos
rec.t_solver.discretization_infos()

signal = rec.values['bell_radiation_pressure']
export_mono('Ex2_cylinder_reed.wav', signal, rec.ts)

