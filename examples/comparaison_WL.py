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


import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation

# Frequencies of interest: 20Hz to 2kHz by steps of 1Hz
fs = np.arange(20, 2000, 1)

fig = plt.figure("Impedance")

shape = [[0.0, 2e-3],
         [0.2, 8e-3]]

for loss_model in ['bessel',
                   'diffrepr2',
                   'diffrepr4',
                   'diffrepr6',
                   'diffrepr8',
                   'wl']:
    result = ImpedanceComputation(fs, shape, temperature=25, losses=loss_model)
    result.plot_impedance(figure=fig, label=loss_model)
