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
How to fix the spatial discretization options (the mesh).
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics, FrequentialSolver)
from openwind.discretization import Mesh

# Frequencies of interest: 20Hz to 2kHz by steps of 1Hz
fs = np.arange(20, 2000, 1)
geom_filename = 'Geom_trumpet.txt'

#%% chosen fine discretisation
# choose a length for the finite elements
length_FEM = 0.1
# choose an order for the finite elements
order_FEM = 10
# Find file 'trumpet' describing the bore, and compute its impedance with s
# pecified length and order for the finite elements
result = ImpedanceComputation(fs, geom_filename, l_ele = length_FEM, order = order_FEM)

# Plot the discretisation information
result.discretization_infos()

# Plot the impedance
fig = plt.figure()
result.plot_impedance(figure=fig, label=f"given fine discretization, nb dof = {result.get_nb_dof()}")


#%% chosen coars discretisation
# choose a length for the finite elements
length_FEM = 0.1
# choose an order for the finite elements
order_FEM = 2
# Find file 'trumpet' describing the bore, and compute its impedance with s
# pecified length and order for the finite elements
result = ImpedanceComputation(fs, geom_filename, l_ele = length_FEM, order = order_FEM)

# Plot the discretisation information
result.discretization_infos()

# Plot the impedance
result.plot_impedance(figure=fig, label=f"given coarse discretization, nb dof = {result.get_nb_dof()}")

#%% default options
# default is an adaptative mesh that provides a reasonable solution with a
# low computational cost
result_adapt = ImpedanceComputation(fs, geom_filename)
result_adapt.discretization_infos()
result_adapt.plot_impedance(figure=fig, label=f"adaptive discretization, nb dof = {result_adapt.get_nb_dof()}")

#%% modify the minimal order for automatic mesh

# Load and process the instrument geometrical file
instr_geom = InstrumentGeometry(geom_filename)
# Create a player using the default value : unitary flow for impedance computation
player = Player()
# Choose the physics of the instrument from its geometry. Default models are chosen when they are not specified.
# Here losses = True means that Zwikker-Koster model is solved.
instr_physics = InstrumentPhysics(instr_geom, temperature=25, player = player, losses=True)
Mesh.ORDER_MIN = 4

# Perform the discretisation of the pipes and put all parts together ready to be solved.
freq_model = FrequentialSolver(instr_physics, fs)
# Solve the linear system underlying the impedance computation.
freq_model.solve()
freq_model.discretization_infos()
freq_model.plot_impedance(figure=fig, label=f"adaptive discretization orders > 4, nb dof = {freq_model.n_tot}")


freq_model = FrequentialSolver(instr_physics, fs, order=2)
freq_model.solve()
freq_model.discretization_infos()
freq_model.plot_impedance(figure=fig, label=f"adaptive discretization orders > 4, nb dof = {freq_model.n_tot}")

plt.show()
