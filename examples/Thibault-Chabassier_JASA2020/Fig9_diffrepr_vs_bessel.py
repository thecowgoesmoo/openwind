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
Displays the entry impedances computed with the different viscothermal models
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import ImpedanceComputation, InstrumentGeometry

#%% Initialize instrument

# fs = np.arange(215, 255, 0.1)
# fs = np.arange(212, 252, 0.1)
fs = np.arange(225, 240, 0.1)
# fs = np.arange(20,2000,3)
temperature = 20
instrument = 'simplified-trumpet'
mm = InstrumentGeometry(instrument)

ordre = 10
lenEle = 0.04

#%% Compute impedance with different models

result_ZK = ImpedanceComputation(fs, instrument,
                                    temperature=temperature,
                                    l_ele=lenEle, order=ordre,
                                    losses=True,
                                    radiation_category='perfectly_open')


result_WL = ImpedanceComputation(fs, instrument,
                                    temperature=temperature,
                                    l_ele=lenEle, order=ordre,
                                    losses='wl',
                                    radiation_category='perfectly_open')

result_diffrepr = {
    N:ImpedanceComputation(fs, instrument,
                            temperature=temperature,
                            l_ele=lenEle, order=ordre,
                            losses='diffrepr'+str(N),
                            radiation_category='perfectly_open')
     for N in [2, 4, 8, 16]}



#%% Display results

fig = plt.figure(figsize=(4,3.5))

result_ZK.plot_impedance(figure=fig, label='ZK', linestyle='-', color='k', linewidth=2)

result_WL.plot_impedance(figure=fig, label='WL', linestyle=':', color='r')

for N, r in result_diffrepr.items():
    r.plot_impedance(figure=fig, label='N=%d'%N, linestyle='--', linewidth=np.log(N))



ax1, ax2 = fig.get_axes()
ax1.legend(loc='lower center', bbox_to_anchor=(0.5,1.02), ncol=3)
fig.align_labels()
fig.tight_layout()
fig.savefig("Figure9.pdf")
