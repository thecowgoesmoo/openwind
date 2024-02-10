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
This example presents how to compute with the Transfer Matrix Method (TMM).
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation


fs = np.arange(20, 500, 1)

# %% Purely TMM

# To use the TMM it is necessary to have a geometry with conical parts only
# (or cylinders)
geom_cone = 'Geom_trumpet_conical.txt'
res_tmm_low = ImpedanceComputation(fs, geom_cone, compute_method='TMM')

fig = plt.figure()
res_tmm_low.plot_impedance(figure=fig, label='TMM')

# To improve the quality of the computation it can be necessary to subdivide
# the conical part (cf. Tournemenne and Chabassier, ACTA 2019)
res_tmm_10 = ImpedanceComputation(fs, geom_cone, compute_method='TMM', nb_sub=5)
res_tmm_10.plot_impedance(figure=fig, label='TMM subdivided 5')

res_tmm = ImpedanceComputation(fs, geom_cone, compute_method='TMM', nb_sub=40)
res_tmm.plot_impedance(figure=fig, label='TMM subdivided 40')
# %% Hybrid method

# It is also possible to use hybrid method combining TMM and FEM, for which only
# the cylinders are computed with the TMM. This has the advantage to accelerate
# the computation w.r. to purely FEM if the cylinders are long without loss of
# precision

res_hydrid = ImpedanceComputation(fs, geom_cone, compute_method='hybrid')
res_hydrid.plot_impedance(figure=fig, label='Hybrid')

# the purely fem to compare
res_fem = ImpedanceComputation(fs, geom_cone)
res_fem.plot_impedance(figure=fig, label='FEM', linestyle=':')


# this time it is possible to use a complex shape containing cylinder(s)
geom_complex = 'Geom_trumpet.txt'
res_complex_hybrid = ImpedanceComputation(fs, geom_complex, compute_method='hybrid')
fig2 = plt.figure()
res_complex_hybrid.plot_impedance(figure=fig2, label='Hybrid')

res_complex_fem = ImpedanceComputation(fs, geom_complex)
res_complex_fem.plot_impedance(figure=fig2, label='FEM', linestyle=':')

plt.show()
