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
How to chose the temperature and the air composition (humidity and CO2 rate).
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation
from openwind.continuous import Physics



# %% Default computation

# by default the temperature is 25°C
fs = np.arange(20, 2000, 1)
geom = 'Geom_trumpet.txt'

result = ImpedanceComputation(fs, geom)
result.plot_instrument_geometry()

fig = plt.figure()
result.plot_impedance(figure=fig, label='Default Temp.: 25°C')

# %% Uniform temperature
# if you want you can impose a uniform temperature
result_30 = ImpedanceComputation(fs, geom, temperature=30)
result_30.plot_impedance(figure=fig, label='30°C')

# %% Variable temperature

# You can also apply a variable temperature by defining a function

# In this case, the temperature variation is along the main axes.
# In the holes the temperature is uniform and equals the one in the main bore
# at their location.
total_length = result.get_instrument_geometry().get_main_bore_length()
def grad_temp(x):
    T0 = 37
    T1 = 21
    return 37 + x*(T1 - T0)/total_length
result_var = ImpedanceComputation(fs, geom, temperature=grad_temp)
result_var.plot_impedance(figure=fig, label='Variable Temp.')

# %% Get access to physical quantities

# It can be useful to get the value of the physical quantities at
# the input (entrance) of the instrument. They can be obtained with the dedicated
# method, in which the desired quantities are indicated as string. For example
# here: the air density, the speed of sound, the specific heat and the viscosity:

rho, celerity, Cp, mu = result_var.get_entry_coefs('rho', 'c', 'Cp', 'mu')
print(f'At the entrance: rho={rho:.2f} kg/m3; celerity={celerity:.2f} m/s; Cp={Cp:.2f} J/(kg.K), mu={mu:.4g} kg/(m.s).')

# It is also possible to get these values with respect to the position without
# specifying an instrument, by using the class :py:class:`Physics<openwind.continuous.physics.Physics>`
my_phy = Physics(grad_temp)
x = np.linspace(0,total_length, 100)
c_x, rho_x = my_phy.get_coefs(x, 'c', 'rho')


fig2, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(x*1000, c_x)
ax[0].set_xlabel('Position [mm]')
ax[0].set_ylabel('Celerity [m/s]')
ax[1].plot(x*1000, rho_x)
ax[1].set_xlabel('Position [mm]')
ax[1].set_ylabel('Density [kg/m^3]')


# %% Air composition

# In addition to the temperature, it is possible to adjust the air composition
# through the humidity rate and the carbon dioxide rate. These both quantities can
# vary a lot during the playing or between measurements.

# This can be done by using the keywords "humidity" and "carbon" with values between
# 0 and 1 (corresponding to 0% and 100%). By default the humidity rate is set to
# 0.5 (50%) which is a typical ambiant value.

# The default carbon rate is 4.2e-4 (420ppm) which correspond to the mean ambiant value.
# Here the computation is performed for a humidity rate of 80% and 10% of CO2 which are
# reasonable playing condition.

# These rates can also be variable along the instrument similarly than for the temperature.


result_playing = ImpedanceComputation(fs, geom, temperature=grad_temp,
                                      humidity=.8, carbon=.1)
result_playing.plot_impedance(figure=fig, label='Playing conditions')


plt.show()
