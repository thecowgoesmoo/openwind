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
This module contains dictionnaries that are used by the
:py:class:`Player <openwind.technical.player.Player>` class

Available dictionnaries are :

- 'WOODWIND_REED': a default Reed1dof excitator with woodwind-reed convention (inward)
- 'LIPS': a default Reed1dof excitator with lips convention (outward)
- 'UNITARY_FLOW': a Flow excitator, with flow imposed to 1
- 'TUTORIAL_LIPS': lips with parameters adapted for tutorial geometry
- 'TUTORIAL_REED': woodwind-reed with parameters adapted for tutorial geometry
- 'OBOE': reed with parameters corresponding to oboe reed
- 'CLARINET': reed with parameters corresponding to clarinet reed
- 'ZERO_FLOW': a Flow excitator, with flow imposed to 0
- 'IMPULSE_100us': a Flow excitator corresponding to an impulse with \
    characteristic time set to 100 micro-sec.
- 'IMPULSE_400us': a Flow excitator corresponding to an impulse with \
    characteristic time set to 400 micro-sec.
"""

import numpy as np
from openwind.technical.temporal_curves import constant_with_initial_ramp, dirac_flow, triangle


FLUTE = {"excitator_type" : "Flute",
         "jet_velocity": 25, # jet velocity in m/s
         "width":4e-3, # channel-edge distance in m
         "channel_height":1e-3, # channel height in m
         "convection":0.5, # ratio convection velocity/jet velocity
         "section":np.pi*16e-6, # cross section area of the embouchure in m²
         'edge_offset':1e-4, # offset between the edge tip and the channel center in m
         "loss_mag": 1, # magnitude of the non linear losses
         "radiation_category": 'infinite_flanged', # type of radiation for the window
         "gain": 1, # gain of the source
         }

# From Blanc AAA 2010 & Ernoult AAA 2017
SOPRANO_RECORDER = {"excitator_type" : "Flute",
                    "jet_velocity": 25,
                    "width":4.45e-3,
                    "channel_height":0.67e-3,
                    "convection":0.5,
                    "section": 9.5*3*1e-6,
                    'edge_offset':1e-4,
                    "loss_mag": 1,
                    "radiation_category": 'window',
                    "gain": 1,
                    "edge_angle":15,
                    "wall_thickness":4e-3,
                    "noise_level":1e-4,
                    }


WOODWIND_REED_SCALED = { # +/- median values from Tab.1, Chabassier & Auvray DAFx20in22
    "excitator_type" : "Reed1dof_scaled",
    "gamma" : constant_with_initial_ramp(0.45, 2e-2),
    "zeta": 0.35,
    "kappa": 0.35,
    "pulsation" : 2*np.pi*2700,
    "qfactor": 6,
    "model" : "inwards",
    "contact_stifness": 1e4,
    "contact_exponent": 4,
    "opening" : 5e-4, #in m
    "closing_pressure": 5e3 #in Pa
}

WOODWIND_REED = {
    "excitator_type" : "Reed1dof",
    "opening" : 1e-4,
    "mass" : 3.376e-6,
    "section" : 14.6e-5,
    "pulsation" : 2*np.pi*3700,
    "dissip" : 3000,
    "width" : 3e-2,
    "mouth_pressure" : constant_with_initial_ramp(2000, 2e-2),
    "model" : "inwards",
        #  valeurs de Bilbao 2008
    "contact_pulsation": 316,
    "contact_exponent": 4
}

# from Fréour et al, JASA 2020
LIPS = {
    "excitator_type" : "Reed1dof",
    "opening" : 1e-4,
    "mass" : 8e-5,
    "section" : 4e-5,
    "pulsation" : 2*np.pi*382,
    "dissip" : 0.3*2*np.pi*382,
    "width" : 8e-3,
    "mouth_pressure" : constant_with_initial_ramp(5500, 1e-2),#triangle(3200, .5),
    "model" : "outwards",
    "contact_pulsation": 0*316,
    "contact_exponent": 4
}

# from Fréour et al, JASA 2020 assuming a input radius of 5mm and T=25°C
LIPS_SCALED = {'excitator_type': 'Reed1dof_scaled',
               'model': 'outwards',
               'gamma': constant_with_initial_ramp(4.8, 1e-2),
               'zeta': 0.16,
               'kappa': 0.04,
               'qfactor': 3.33,
               'pulsation': 2400.,
               'contact_stifness': 0.0,
               'contact_exponent': 4,
               'opening': 0.0001, # mm
               'closing_pressure': 1152 # Pa
               }

# These parameters are only used by the basic tutorial
# but they do not correspond to anything physical
TUTORIAL_LIPS = {
    "excitator_type" : "Reed1dof",
    "opening" : 9.4e-4,
    "mass" : 6.4e-5,
    "section" : 1.9e-4,
    "pulsation" : 2*np.pi*750,
    "dissip" : 0.7*2*np.pi*750,
    "width" : 11.9e-3,
    "mouth_pressure" : constant_with_initial_ramp(5000, 1e-2),
    "model" : "outwards",
    "contact_pulsation": 0,
    "contact_exponent": 4
}

TUTORIAL_REED = {
    "excitator_type" : "Reed1dof",
    "opening" : 9.4e-4,
    "mass" : 6.4e-5,
    "section" : 1.9e-4,
    "pulsation" : 2*np.pi*750,
    "dissip" : 0.7*2*np.pi*750,
    "width" : 11.9e-3,
    "mouth_pressure" : constant_with_initial_ramp(5000, 1e-2),
    "model" : "inwards",
    "contact_pulsation": 0,
    "contact_exponent": 4
}

OBOE = {
    "excitator_type" : "Reed1dof",
    "opening" : 8.9e-5,
    "mass" : 7.1e-4,
    "section" : 4.5e-5,
    "pulsation" : 2*np.pi*600,
    "dissip" : 0.4*2*np.pi*600,
    "width" : 9e-3,
    "mouth_pressure" : constant_with_initial_ramp(12000, 2e-2),
    "model" : "inwards",
    "contact_pulsation": 316,
    "contact_exponent": 4
}

# values of Bilbao 2008
CLARINET = {
    "excitator_type" : "Reed1dof",
    "opening" : 4e-4,
    "mass" : 3.376e-6,
    "section" : 14.6e-5,
    "pulsation" : 2*np.pi*3700,
    "dissip" : 3000,
    "width" : 3e-2,
    "mouth_pressure" : constant_with_initial_ramp(2000, 2e-2),
    "model" : "inwards",
    "contact_pulsation": 316,
    "contact_exponent": 4
}


UNITARY_FLOW = {
    "excitator_type":"Flow",
    "input_flow":1
}

ZERO_FLOW = {
    "excitator_type":"Flow",
    "input_flow":0
}

IMPULSE_400us = {
    "excitator_type":"Flow",
    "input_flow": dirac_flow(4e-4)
}

IMPULSE_100us = {
    "excitator_type":"Flow",
    "input_flow": dirac_flow(1e-4)
}
