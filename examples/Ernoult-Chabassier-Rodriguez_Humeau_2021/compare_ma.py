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
This example illustrate the error on m_a due to the "long chimney" approximation
for the geometry of Hole 1 and 4. It is related to the article:

    Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full waveform \
    inversion for bore reconstruction of woodwind-like instruments", submitted
    to Acta Acustica. https://hal.inria.fr/hal-03231946
"""
import numpy as np

label = ['Hole 1', 'Hole 4']
chimney = [1.7e-3, 1.4e-3]
radii = [1.5e-3, 1.25e-3]

for name, t, b in zip(label, chimney, radii):
    a = 2e-3
    delta = b/a
    ta_o = -1/(1.78*np.tanh(1.84*t/b) + 0.940 + 0.540*delta + 0.285*delta**2)
    ta_c = -1/(1.78/np.tanh(1.84*t/b) + 0.940 + 0.540*delta + 0.285*delta**2)

    ta = -0.37 + 0.087*delta

    print("For {}, the relative error between 'long chimney' approx and "
          "complete formulations are:".format(name))
    print("\t open hole: {:.2e}".format(ta / ta_o - 1))
    print("\t closed hole: {:.2e}".format(ta / ta_c - 1))
