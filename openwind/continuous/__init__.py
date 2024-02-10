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
This module convert the instrument in a netlist (graph) of pipes, whose
ends are connected to different conditions (radiations, junctions, etc).
Each netlist element can compute the coefficients of the corresponding physical
equation (wave propagation, losses, etc).
"""

# === Things Used Everywhere ===

from .physics import Physics
from .scaling import Scaling

# - Boundary condition (for temporal) -
# TODO maybe move this ?
# from .flow_model import (dirac_flow, chirp_flow)


# === All the Models ===

from .netlist import NetlistConnector


# - Losses -
from .thermoviscous_models import (ThermoviscousModel,
                                   ThermoviscousLossless,
                                   ThermoviscousBessel,
                                   ThermoviscousDiffusiveRepresentation,
                                   SphericalHarmonics,
                                   WebsterLokshin,
                                   Keefe,
                                   MiniKeefe,
                                   ParametricRoughness,
                                   losses_model)

# - Propagation in Pipes -
from .pipe import Pipe

# - Radiation Impedances -
from .physical_radiation import (PhysicalRadiation, RadiationPade,
				RadiationPerfectlyOpen)
from .radiation_silva import RadiationNoncausal, RadiationPade2ndOrder
from .radiation_from_data import RadiationFromData, radiation_from_data
from .radiation_pulsating_sphere import RadiationPulsatingSphere
from .radiation_model import radiation_model

# - Junctions of Pipes -
from .junction import (PhysicalJunction,
                       JunctionTjoint,
                       JunctionSimple,
                       JunctionDiscontinuity,
                       JunctionSwitch)
from .tonehole import Tonehole

# - Excitators -
from .excitator import (create_excitator, Excitator, Flow, Flute, Reed1dof, Reed1dof_Scaled)

# === Graph Representation of the Instrument ===

from .netlist import Netlist, EndPos
from .instrument_physics import InstrumentPhysics
