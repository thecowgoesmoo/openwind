#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2022, INRIA
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
This example presents how to compute the TMM of a complex shape using FEM. And
how to use this matrix to "remove" a part from an impedance.

"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import InstrumentGeometry, InstrumentPhysics, Player, FrequentialSolver
from openwind.compute_transfer_matrix import ComputeTransferMatrix, remove_adaptor
from openwind.impedance_tools import plot_impedance

# %% Compute the transfer matrix of a complex shape

# We define a complex shape, like a adaptor for impedance measurement, for which no expression
# of the transfer matrix exists.

adaptor_geom = [[0,3e-2,2e-3,4e-3,'spline',1.3e-2, 1.7e-2, 2.5e-3, 3.5e-3],
                   [3e-2,4e-2, 4e-3,5e-3,'cone']
                   ]

adaptor = InstrumentGeometry(adaptor_geom)
adaptor.plot_InstrumentGeometry()

# it is possible to compute the transfer matrix of this shape by using FEM simulation,
# with different radiation conditions. This has been implemented in a specific function:

freq = np.arange(100, 2001, 1)
temperature = 25
mesh_options = {'order':10, 'l_ele':10e-3 }
phy_options = {'humidity':.5, 'spherical_waves': False} # you can use any options, except the radiation condition
A, B, C, D = ComputeTransferMatrix(adaptor_geom, freq, temperature, **mesh_options, **phy_options)

plt.figure()
plt.semilogy(freq, np.abs(A), label='A')
plt.semilogy(freq, np.abs(B), label='B')
plt.semilogy(freq, np.abs(C), label='C')
plt.semilogy(freq, np.abs(D), label='D')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Modulus (logscale)')

# we can check that, when the mesh is refined enough, the determinant of the obtained matrix equals 1.

determinant = (A*D - B*C)

print(f'The maximal deviation of the determinant to 1 is: {np.max(np.abs(determinant)-1):e}')


# %% Remove a part in impedance measurement

# This transfer matrix can be used to remove appart from measured impedance,
# it can be especially usefull for instruments measured with an adaptor.
# lets imagine that a trumpet body is measured with this adaptor:

trumpet_geom= [[0,1,5e-3,5e-3,'cone'],
               [1,1.5,5e-3,7e-3,'cone'],
               [1.5,2,7e-3,0.1,'bessel',.75]]
trumpet_body = InstrumentGeometry(trumpet_geom)

full_trumpet = adaptor + trumpet_body
full_phy = InstrumentPhysics(full_trumpet, temperature, Player(), True, **phy_options)
full_freq = FrequentialSolver(full_phy, freq, **mesh_options)
full_freq.solve()
full_freq.write_impedance('Ex12_full_trumpet.txt', normalize=True)

# knowing the geometry of the adaptor, it can be removed, using this transfer matrix:
freq_remove, Z_remove = remove_adaptor(adaptor_geom, 'Ex12_full_trumpet.txt', temperature, write_files=False, **phy_options, **mesh_options)

# for the example we can compare the obtained impedance to the one computed for the body only
body_phy = InstrumentPhysics(trumpet_body, temperature, Player(), True, **phy_options)
body_freq = FrequentialSolver(body_phy, freq, **mesh_options)
body_freq.solve()
Zbody = body_freq.impedance/body_freq.get_ZC_adim()

fig_imp = plt.figure()
plot_impedance(freq_remove[0], Z_remove[0], figure=fig_imp, label='Adpator removed')
body_freq.plot_impedance(figure=fig_imp, label='body only', linestyle='--')

error = np.linalg.norm(Z_remove[0] - Zbody)/np.linalg.norm(Zbody)
print(f"The relative deviation between the 2 impedances is: {error:.2e}.")


plt.show()
