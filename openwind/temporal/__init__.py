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
Numerical schemes used for time-domain simulation.
"""

# ====== Temporal Components ======
from .tcomponent import TemporalComponent, TemporalComponentExit

# - Pipes -
from .tpipe import TemporalPipe
from .tpipe_lossy import TemporalLossyPipe
from .tpipe_rough import TemporalRoughPipe

# - One-end components -
from .tflow_condition import TemporalFlowCondition
from .tpressure_condition import TemporalPressureCondition
from .tradiation import TemporalRadiation
from .treed1dof import TemporalReed1dof
from .treed1dof_scaled import TemporalReed1dofScaled
from .tflute import TemporalFlute

# - Junctions -
from .tjunction import TemporalJunction, TemporalJunctionDiscontinuity, TemporalJunctionSwitch
from .tsimplejunction import TemporalSimpleJunction

from .ttonehole import TemporalTonehole

# ====== Managing the Simulation ======
# - the fingerings
from .execute_score import ExecuteScore

# - Running the simulation -
from .temporal_solver import TemporalSolver

# - Recording data -
from .recording_device import RecordingDevice
