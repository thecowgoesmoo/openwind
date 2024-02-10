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
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind import Player, InstrumentGeometry, InstrumentPhysics
from openwind.discretization import Mesh

shape = [[0,0],[1,0]]
igeom = InstrumentGeometry(shape)
iphy = InstrumentPhysics(igeom, 25, Player(), False)
pipe, _ = iphy.netlist.pipes['bore0']
mesh = Mesh(pipe,l_ele=0.5, order=4)


xx = np.linspace(0,1,1000)
psi_xx = mesh.get_interp_mat_L2(xx).todense()
xL2 = mesh.get_xL2()
phi_xx = mesh.get_interp_mat_H1(xx).todense()
xH1 = mesh.get_xH1()

# fig = plt.figure(figsize=(4,2.6))
# ax2 = fig.add_axes((0,0.5,1,0.5))
# ax = fig.add_axes((0,0,1,0.5))

fig, (ax2, ax) = plt.subplots(2,1,figsize=(4,2.6))
for i in range(5):
    if i<4:
        line, = ax.plot(xx, psi_xx[:,i])
        color = line.get_color()
    else:
        # Special plot for psi_4
        k = np.searchsorted(xx, 0.5)
        line, = ax.plot(xx[:k], psi_xx[:k,i])
        color = line.get_color()
        ax.plot(xx[k-1:k+1], psi_xx[k-1:k+1,i], color=color, linestyle="--")
        ax.plot(xx[k:], psi_xx[k:,i], color=color)

    ax.scatter([xL2[i]], [1], marker='o', color=color)
    ax.text(xL2[i], 1.2, f"$\\psi_{i}$", color=color, va='center', ha='center', size='x-large')

    line2, = ax2.plot(xx, phi_xx[:,i])
    ax2.scatter([xH1[i]], [1], marker='o', color=color)
    ax2.text(xH1[i], 1.2, f"$\\phi_{i}$", color=color, va='center', ha='center', size='x-large')



ax.set_ylim(-0.3,1.3)
ax2.set_ylim(-0.3,1.3)

def cartesian_axes(ax):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

# ax.set_axis_off()
# ax2.set_axis_off()
ax.set_xlabel('x')
cartesian_axes(ax)
cartesian_axes(ax2)
plt.tight_layout()
plt.savefig("basis_functions.png", dpi=300)

#%% Display the finite element matrices

from matplotlib.image import AxesImage
from matplotlib.colors import Normalize

mesh2 = Mesh(pipe,l_ele=0.2, order=4)
wL2 = mesh2.get_weights()
wH1 = mesh2.assemble_H1_from_L2(wL2)
Bh = mesh2.get_Bh()
nL2 = len(wL2)
nH1 = len(wH1)
# fig, axs = plt.subplots(1,3,figsize=(6,2.6), sharey=True)
fig = plt.figure(figsize=(4,4/3))
ax = fig.add_axes((0,0,1,1))
norm = Normalize(vmin=-0.1, vmax=0.3)

# axs[0].imshow(np.diag(wL2), cmap="Blues", extent=(0,nL2,0,nL2))
curx = 0

img = AxesImage(ax, cmap="Blues", origin="upper", extent=(curx,curx+nH1,0,nH1), norm=norm)
img.set_data(np.diag(wH1**0.5))
ax.add_image(img)
ax.text(curx + nH1/2, -3, "$M_h^P$", ha="center", va="center")
curx += nH1 + 5

img = AxesImage(ax, cmap="Blues", origin="upper", extent=(curx,curx+nL2,0,nL2), norm=norm)
img.set_data(np.diag(wL2**0.5)) # apply sqrt for brighter visualization
ax.add_image(img)
ax.text(curx + nL2/2, -3, "$M_h^U$", ha="center", va="center")
curx += nL2 + 5

img = AxesImage(ax, cmap="RdBu", origin="upper", extent=(curx,curx+nH1,0,nL2))
img.set_data(Bh.todense())
ax.add_image(img)
ax.text(curx + nH1/2, -3, "$B_h$", ha="center", va="center")
curx += nH1 + 5


vec = np.array([[k] for k in range(nH1)])
img = AxesImage(ax, cmap="Blues", origin="upper", extent=(curx,curx+1,0,nH1), norm=norm)
img.set_data((vec == 0) * 1.0)
ax.add_image(img)
ax.text(curx + 1/2, -3, "$E_{h\\!-}$", ha="center", va="center")
curx += 1 + 7

img = AxesImage(ax, cmap="Blues", origin="upper", extent=(curx,curx+1,0,nH1), norm=norm)
img.set_data((vec == nH1-1) * 1.0)
ax.add_image(img)
ax.text(curx + 1/2, -3, "$E_{h\\!+}$", ha="center", va="center")
curx += 1

# Hack: draw an invisible diagonal line, just to set the right data limits
ax.plot([0,curx], [-5, nL2], color="#00000000")
# ax.set(xlim=(0,curx), ylim=(0,nL2))
ax.axis('equal')
ax.set_axis_off()

plt.savefig("matrix_profiles.png", dpi=300)
