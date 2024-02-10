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
Created on Mon Jun 19 15:08:46 2023

@author: alexis
"""

from openwind.continuous import NetlistConnector

class Tonehole(NetlistConnector):
    """Models a complete tonehole as the combination of a junction,
    a chimney pipe and a radiation.

    Used for temporal simulation with locally implicit schemes"""

    def __init__(self, junct, pipe, rad,
                 label, scaling, convention='PH1'):
        super().__init__(label, scaling, convention)
        self.junct = junct
        self.pipe = pipe
        self.rad = rad
