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
Introduction to the basic aspects of bore reconstruction with OpenWInD.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind.inversion import InverseFrequentialResponse

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics)



plt.close('all')


# %% Global options

frequencies = np.linspace(100, 500, 10)
temperature = 20
losses = True
# %% Targets definitions
# For this example we use simulated data instead of measurement

# The geometry is 0.5m cylinder with a radius of 2mm.
target_geom = [[0, 0.5, 2e-3, 2e-3, 'linear']]
target_computation = ImpedanceComputation(frequencies, target_geom,
                                          temperature=temperature,
                                          losses=losses)

# The impedance used in target must be normalized
Ztarget = target_computation.impedance/target_computation.Zc

# noise is added to simulate measurement
noise_ratio = 0.01
Ztarget = Ztarget*(1 + noise_ratio*np.random.randn(len(Ztarget)))

# %% Definition of the optimized geometry

# Here we want to adjust only the pipe length: only this parameter is preceded by "~"
inverse_geom = [[0, '~0.3', 2e-3, 2e-3, 'linear']]
# the initial length is set here to 0.3m

instru_geom = InstrumentGeometry(inverse_geom)

# During the process, an attribute `optim_param` has been instanciated.
# It contains all the information on the parameters included in the optimization
print(instru_geom.optim_params)

# We can compare the two bore at the initial state
fig_geom = plt.figure()
target_computation.plot_instrument_geometry(figure=fig_geom, label='Target')
instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Initial Geometry')
fig_geom.legend()


# %% Construction of the inverse problem

# Instanciate a player with defaults
player = Player()

# Instanciation of the physical equation
instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)

# Instanciation of the inverse problem
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget)

# We can now compare the impedances at the initial state
inverse.solve()
fig_imped = plt.figure()
target_computation.plot_impedance(figure=fig_imped, label='Target', marker='o',
                                  linestyle=':')
inverse.plot_impedance(figure=fig_imped, label='Initial', marker='x',
                       linestyle=':')

# %% Optimization process

# the InverseFrequentialResponse has a method which computes the cost and
# gradient for a given value of the design parameters
cost, grad = inverse.get_cost_grad_hessian([], grad_type='adjoint')[0:2]
print('With current geometry: Cost={:.2e}; Gradient={:.2e}'.format(cost,
                                                                   grad[0]))

# This method can be used with any optimization algorithm.
# This is what it is done in the dedicated method:
result = inverse.optimize_freq_model(iter_detailed=True)
# The default optimization algorithm chosen is 'lm' for "Levenberg-Marquart"
# (from scipy) which is often the most efficient for unconstrained problem.

# %% Plot the result
print('The final length is {:.2f}m'.format(result.x[0]))
print('The deviation w.r. to the target value is '
      '{:.2e}m'.format(np.abs(result.x[0] - 0.5)))

# we add the final impedance to the curve:
inverse.plot_impedance(figure=fig_imped, label='Final', marker='+',
                       linestyle=':')

# we add the final geometry
instru_geom.plot_InstrumentGeometry(figure=fig_geom, linestyle=':', color='k',
                                    label='Final Geometry')
fig_geom.legend()

# plot the evolution of the length
plt.figure()
plt.plot(np.arange(0, result.nit), np.array(result.x_evol))
plt.xlabel('Iterations')
plt.ylabel('Length (m)')

plt.show()
