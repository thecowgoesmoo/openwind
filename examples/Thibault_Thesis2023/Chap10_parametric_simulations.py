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
This script is part of the numerical examples accompanying Alexis THIBAULT's
Ph.D. thesis.

Simulate impulse response of a closed cylinder for several loss models.

This script generates figures 10.2 (a) (b) and (c) of the thesis.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import simulate, Player
from openwind.temporal import utils

HIGH_PRECISION = True

instrument = [[0, 6e-3], [1.3, 6e-3]]
duration = 0.2
results = dict()


#%% Select which loss model to use

tests = [
            (False, "lossless"),
            ('diffrepr8', "ZK"),
            # ('diffrepr8 2.0', "ZK-HR $\\alpha=2$"),
            ('diffrepr8 3.0', "ZK-HR $\\alpha=3$"),
            # ('parametric_roughness 1.0 50e-6 8', "param $\\alpha=1$"),
            # ('parametric_roughness 2.0 50e-6 8', "param $\\alpha=2$"),
            ('parametric_roughness 3.0 50e-6 8', "param $\\alpha=3$")
          ]

#%% Run the simulations

for i, (losses, label) in enumerate(tests):
    print('\n'+'*'*50)
    print("{:^50}".format(f"LOSS MODEL --- {losses}"))
    print('*'*50)
    rec = simulate(duration,
                   instrument,
                   player=Player("IMPULSE_400us"),
                   losses=losses,
                   radiation_category='perfectly_open',
                   temperature=20,
                   l_ele=0.002 if HIGH_PRECISION else 0.02,
                   order=4, # Discretization parameters
                   spherical_waves=False,
                   cfl_alpha=1.0-1e-6
                   )
    results[losses] = rec

    # Export the impulse response
    signal = rec.values['source_pressure']
    label_clean = ''.join(filter(str.isalnum, label))
    utils.export_mono(f'impulse_{label_clean}.wav', signal, rec.ts)
    np.save(f"impulse_{label_clean}.npy", signal)
    np.save(f"ts_{label_clean}.npy", rec.ts)

#%% Display the results

# fig = plt.figure(figsize=(4,2.6))
fig = plt.figure()

colors = ["gray",
          "C0", "C1",
          "C2", "C3"]
styles = [":", "-", "--", "-", "--"]
markers = [None, "s", None, "x", None]
lines = []

for i, (losses, label) in enumerate(tests):
    style = styles[i]
    color = colors[i]
    marker = markers[i]

    label_clean = ''.join(filter(str.isalnum, label))
    signal = np.load(f"impulse_{label_clean}.npy")
    ts = np.load(f"ts_{label_clean}.npy")

    # Add a vertical offset to the signals for easier visualization
    offset = 1e3 * (4-i)

    line, = plt.plot(ts*1000, signal + offset, style, label=label,
              color=color,
             )
    lines.append(line)
    plt.ticklabel_format(axis='y', scilimits=(3,3))
    plt.ylabel('$p(x=0, t)$')
    plt.xlabel('time (ms)')


plt.legend(columnspacing=0.5)
fig.tight_layout()

plt.xlim(-5, 105)
plt.grid(True, "both", alpha=0.5)

for i, (losses, label) in enumerate(tests):
    line = lines[i]
    width = line.get_linewidth()
    line.set_linewidth(3*width)
    fig.savefig(f'impulse_all_{label}.png', dpi=300)
    line.set_linewidth(width)


#%% Save additional figures (zoomed in on specific resonances)


fig = plt.figure(figsize=(4,2.6))

for i, (losses, label) in enumerate(tests):
    style = styles[i]
    color = colors[i]
    marker = markers[i]

    label_clean = ''.join(filter(str.isalnum, label))
    signal = np.load(f"impulse_{label_clean}.npy")
    ts = np.load(f"ts_{label_clean}.npy")

    offset = 0

    plt.plot(ts*1000, signal + offset, style, label=label,
               linewidth=1,
              color=color,
                marker=marker,
                markevery=0.1 * (1.3)**(i%3),
             )
    plt.ticklabel_format(axis='y', scilimits=(3,3))
    plt.ylabel('$p(x=0, t)$')
    plt.xlabel('time (ms)')

plt.legend(ncol=2,
            columnspacing=0.5,
            fontsize="small"
            )
fig.tight_layout()
plt.grid(True, "both", alpha=0.5)

plt.xlim(7.3, 10.3)
plt.ylim(-1.6e3,0.2e3)
plt.savefig('impulse_2.png', dpi=300)

#%%

plt.xlim(37.5, 40.5)
plt.ylim(-0.7e3,0.1e3)
plt.savefig('impulse_3.png', dpi=300)
