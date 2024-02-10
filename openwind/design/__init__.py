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
Defines openwind object used to design the bore profile. The radius evolution
of each tube is described by a design shape parametrize by several design
parameters.
"""


from .design_parameter import (DesignParameter,
                               FixedParameter,
                               VariableParameter,
                               VariableHolePosition,
                               VariableHoleRadius,
                               OptimizationParameters,
                               eval_, diff_)

# === Design Shapes ===
from .design_shape import DesignShape

# - Basic shape formulas
from .bessel import Bessel
from .circle import Circle
from .cone import Cone
from .exponential import Exponential
from .spline import Spline

# - Shape defined using another shape
from .shape_slice import ShapeSlice
