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
This module treats all the aspects of the computation in frequential domain.
It converts the netlist component in frequential components, build and solves
the global linear equation.
"""

# === Frequential Components ===
from .frequential_component import FrequentialComponent

# - Pipes -
from .frequential_pipe_fem import FrequentialPipeFEM, FPipeEnd
from .frequential_pipe_diffusive_representation import FrequentialPipeDiffusiveRepresentation
from .frequential_pipe_tmm import FrequentialPipeTMM

# - One-end components -
from .frequential_radiation import FrequentialRadiation
from .frequential_radiation1dof import FrequentialRadiation1DOF
from .frequential_source import FrequentialSource, FrequentialFluteSource
from .frequential_pressure_condition import FrequentialPressureCondition

# - Junctions -
from .frequential_junction_tjoint import FrequentialJunctionTjoint
from .frequential_junction_simple import FrequentialJunctionSimple
from .frequential_junction_discontinuity import FrequentialJunctionDiscontinuity
from .frequential_junction_switch import FrequentialJunctionSwitch


# === Solving and after ===
from .frequential_interpolation import FrequentialInterpolation
from .frequential_solver import FrequentialSolver
