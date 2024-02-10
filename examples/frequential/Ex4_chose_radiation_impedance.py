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
How to chose the radiation impedance imposed at the open boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation, Physics
from openwind.continuous import radiation_model

fs = np.arange(50, 1000, 2)
temp = 25
geom = 'Geom_trumpet.txt'
holes = 'Geom_holes.txt'

fig = plt.figure()

# You can modify the model used to compute the radiation impedance by using
# the optional keyword 'radiation_category'

# .. warning::
# do not use this to open/close holes!

# %% Default
# by default the radiation category is "unflanged" corresponding to the
# radiation of a pipe with inifinite thin wall.

result_default = ImpedanceComputation(fs, geom, holes, temperature=temp)
result_default.plot_impedance(figure=fig, label='Default: unflanged',
                              color='k', lw=5)

# %% Available options

# - 'planar_piston': radiation of planar piston
# - 'unflanged': radiation of an unflanged pipe (default)
# - 'infinite_flanged': radiation of en infinite flanged pipe
# - 'total_transmission': reflection  = 0 (Zrad=Zc)
# - 'closed': perfectly close (do not use that to close hole! It close also the main pipe)
# - 'unflanged_2nd_order': unflanged using causal formula from Silva, JSV 2009
# - 'flanged_2nd_order': infinite flanged using causal formula from Silva, JSV 2009
# - 'unflanged_non_causal': unflanged using non-causal formula from Silva, JSV 2009
# - 'flanged_non_causal': infinite flanged using non-causal formula from Silva, JSV 2009
# - 'perfectly_open': imposed a zero pressure
# - 'pulsating_sphere': take into account the final conicity to compute the radiation of the final portion of sphere (used with spherical waves cf. Ex8)



rad_cats = ['planar_piston', 'unflanged', 'infinite_flanged',
            'total_transmission', 'closed', 'unflanged_2nd_order',
            'flanged_2nd_order', 'unflanged_non_causal', 'flanged_non_causal',
            'pulsating_sphere', 'perfectly_open']

for rad_cat in rad_cats:
    result = ImpedanceComputation(fs, geom, holes, temperature=temp,
                                  radiation_category=rad_cat)
    result.plot_impedance(figure=fig, label=rad_cat)

# %% Radiation with supplementary info

# For some model it is possible to give supplementary information. For exemple,
# it is possible to impose the angle of the pulsating sphere by doing:
rad_cat = ('pulsating_sphere', np.pi/4)
result = ImpedanceComputation(fs, geom, holes, temperature=temp,
                              radiation_category=rad_cat)
result.plot_impedance(figure=fig, label='pulsating sphere: $\\pi/4$', ls='--')

# %% Radiation from data

# It is possible to use radiation impedance from data coming from measurement
# or other simulations.
# It is necessary to give:
# - the frequency and the normalized impedance (Z/Zc)
# - the temperature corresponding to the simulation/measurement
# - the radius of the opening

# The measured data can be given:
# - a tuple of array: (frequency, Z/Zc)
# - a file with 3 columns: frequency, real(Z/Zc), imag(Z/Zc)

# For the example the data are computed from another radiation model:

radius = 6.1e-2
rho, celerity = Physics(temp).get_coefs(0, 'rho', 'c')
freq_data = np.arange(1e-5, 1000, 2)
omega = freq_data*2*np.pi
Zc = rho*celerity/(np.pi*radius**2)

rad_data = radiation_model('unflanged')
Z_data = rad_data.get_impedance(omega, radius, rho, celerity, 1.)
data = (freq_data, Z_data/Zc)

# Instead of these lines, it is possible to set: `data = 'filename.txt'`

# .. warning::
# the dimensionless wavenumber kr of the given data must cover the
# range needed to compute the radiation of each opening (no extrapolation)

# To give these information the radiation category the data info are given as
# supplementary informations to the radiation category `from_data`:

data_info = (data, temp, radius)

rad_test = radiation_model(('from_data', data_info))

result_data = ImpedanceComputation(fs, geom, holes, temperature=temp,
                                   radiation_category=('from_data', data_info))

fig_data = plt.figure()
result_default.plot_impedance(figure=fig_data, label='Default: unflanged',
                              color='k', lw=3)
result_data.plot_impedance(figure=fig_data, label='from data (unflanged)')

# %% Different condition at each opening

# It is possible to give a radiation category different for the main bore bell
# and the holes by giving a dictionnary:
rad_dict = {'bell': 'pulsating_sphere', 'holes': 'infinite_flanged'}

result_dict = ImpedanceComputation(fs, geom, holes, temperature=temp,
                                   radiation_category=rad_dict)

fig_dict = plt.figure()
result_default.plot_impedance(figure=fig_dict, label='Default: unflanged',
                              color='k', lw=3)
result_dict.plot_impedance(figure=fig_dict,
                           label='Bell: sphere; Holes: flanged')

# A radiation category different can be given to each radiating opening but
# it is necessary to now the label of each of the holes like in the fingering
# chart (cf: technical example 2). Here they are given in the file 'Geom_holes.txt'

rad_dict_complex = {'bell': 'closed', 'hole1': 'unflanged',
                    'hole2': 'infinite_flanged',
                    'hole3': ('pulsating_sphere', np.pi/4),
                    'hole4': ('from_data', data_info)}

result_dict_complex = ImpedanceComputation(fs, geom, holes, temperature=temp,
                                           radiation_category=rad_dict_complex)
result_dict_complex.plot_impedance(figure=fig_dict,
                                   label='Different for each opening')

# .. warning::
# it is a really unconventional way to set fingerings...

plt.show()
