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
Simulate impulse response of a simplified trumpet and plot it.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import simulate, InstrumentGeometry, Player

instrument = 'simplified-trumpet'
mm = InstrumentGeometry(instrument)

duration = 0.03

numbers = [64, 128, 256, 512, 1024, 2048]
# numbers = [64, 128, 256]

for n_ele in numbers:
    print("\n", "*"*50)
    print("Simulation with n_ele =", n_ele)
    rec = simulate(duration,
                   instrument,
                   player=Player("IMPULSE_400us"),
                   losses='diffrepr4',
                   radiation_category='closed',
                   temperature=20, # Impulse response
                   l_ele=mm.get_main_bore_length()/n_ele, order=4, # Discretization parameters
                   spherical_waves=False,
                   cfl_alpha=1.0,
                   verbosity=1,
                   )
    # res[n_ele] = rec
    np.save(f"cv_ts_{n_ele}.npy", rec.ts)
    np.save(f"cv_pressure_{n_ele}.npy", rec.values['source_pressure'])





#%% Post-processing

ts_post = np.linspace(0, duration, 200, endpoint=False)  # Times at which to compare solutions
def interp(n_ele):
    # Interpolate solution at times ts_post
    ts = np.load(f"cv_ts_{n_ele}.npy")
    values = np.load(f"cv_pressure_{n_ele}.npy")
    return np.interp(ts_post, ts, values)

converged = interp(numbers[-1])

err = dict()

for n_ele in numbers:
    signal = interp(n_ele)
    errsq = np.max(np.abs(signal - converged)) / np.max(np.abs(converged))
    err[n_ele] = errsq

keys = np.array(list(err.keys()))
values = np.array(list(err.values()))

plt.figure(figsize=(4,2))
plt.loglog(keys, values, 'ok')
plt.xlabel('Number of elements')
# plt.ylabel(r"$\frac{\sup_{t} |p_h(x=0, t) - p(x=0, t)|}{\sup_{t} |p(x=0, t)|}$")
plt.ylabel("Maximal error on\nimpulse response")
plt.grid('both')

# order2 = [1/n_ele**2 for n_ele in err.keys()]
#order2 = keys[2:4]**-2.0
#order2 *= values[2]/order2[0]
order2x = [keys[1], keys[2], keys[1], keys[1]]
order2y = [x**2*1e-7 for x in order2x[::-1]]
#plt.loglog(keys[2:4], order2, ':')
plt.loglog(order2x, order2y, 'k-')

plt.annotate("1", xy=(np.sqrt(order2x[0]*order2x[1]), order2y[0] * 0.9),
             horizontalalignment='center', verticalalignment='top')
plt.annotate("2 ", xy=(order2x[0], np.sqrt(order2y[0]*order2y[2])),
             horizontalalignment='right', verticalalignment='center')

# plt.legend()
plt.tight_layout()
plt.savefig("Figure7.pdf")
