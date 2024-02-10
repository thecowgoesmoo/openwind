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

from openwind import simulate, Player
from openwind.temporal import utils

#%% Fairly long simulation (~2 minutes)

instrument = 'simplified-trumpet'
duration = 0.2
results = dict()

for N, style in [(2, ':'), (4, '-.'), (8, '--'), (16, '-')]:
    print('\n'+'*'*50)
    print("{:^50}".format("DIFFUSIVE REPRESENTATION --- N = %d" % N))
    print('*'*50)
    rec = simulate(duration,
                   instrument,
                   player=Player("IMPULSE_400us"),
                   losses='diffrepr'+str(N),
                   radiation_category='perfectly_open',
                   temperature=20,
                   l_ele=0.04, order=10, # Discretization parameters
                   spherical_waves=False,
                   cfl_alpha=1.0
                   )
    results[N] = rec
    np.save(f"pressure_diffrepr{N}.npy", rec.values['source_pressure'])
    np.save(f"ts_diffrepr{N}.npy", rec.ts)

#%% Display the results

fig = plt.figure(figsize=(4,3.3))
for N, style in [(2, ':'), (4, '-.'), (8, '-'), (16, '--')]:
    style = '--'
    # rec = results[N]
    # signal = rec.values['source_pressure']
    signal = np.load(f"pressure_diffrepr{N}.npy")
    ts = np.load(f"ts_diffrepr{N}.npy")

    plt.subplot(2, 1, 1)
    plt.plot(ts, signal, style, label=f'N={N}', linewidth=np.log(N))
    plt.ticklabel_format(axis='y', scilimits=(3,3))
    plt.ylabel('$p(x=0, t)$')
    # plt.legend(loc='upper right')
    # plt.legend(loc='lower center', bbox_to_anchor=(0.55,1.02), ncol=4,
    #            frameon=True,
    #            handletextpad=0.5,
    #            columnspacing=1.0)
    plt.legend(loc='center right')
    # plt.ylabel('pressure $p(x=0, t)$ (Pa)')
    plt.subplot(2, 1, 2)
    plt.plot(ts, signal, style, linewidth=np.log(N))
    plt.ticklabel_format(axis='y', scilimits=(1,1))
    plt.xlabel('time (s)')
    plt.ylabel('$p(x=0, t)$')
    #plt.xlim(0.0075, 0.009)
    # plt.ylim(-30, 80)
    #plt.ylim(-1000, -500)
    plt.xlim(0.13,0.18)
    plt.ylim(-50,50)

fig.align_labels()
fig.tight_layout()

fig.savefig('Figure4.pdf')

#%%  Export the last one as audio
utils.export_mono('impulse.wav', signal, ts)
