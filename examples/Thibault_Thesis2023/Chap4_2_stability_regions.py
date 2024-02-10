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

Display the stability regions of various numerical schemes.
This script generates Figure 4.2 of the thesis.

Many numerical schemes for the ODE

    y' = A y

can be put in the form

    y^{n+1} = R(dt*A) y^n,

where R(z) is a quotient of polynomials.
The stability region of that scheme is the set

    S = { z complex number s.t. |R(z)| <= 1 }.

The numerical solution remains stable iff
for any eigenvalue lambda of matrix A,
dt*lambda is in S.
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Select the numerical schemes to display


# You may uncomment some lines to enable the display of the corresponding schemes.
# Do not select more than 9 schemes.
schemas = [
    # --- EULER SCHEMES ---
    ("Euler explicite", lambda z: 1+z),
    ("Euler implicite", lambda z: 1/(1-z)),

    # --- EXPLICIT RUNGE-KUTTA SCHEMES ---
    # ("ERK22", lambda z: 1 + z + z**2/2),
    ("ERK4", lambda z: 1 + z + z**2/2 + z**3/6 + z**4/24),
    # ("ERK5", lambda z: 1 + z + z**2/2 + z**3/6 + z**4/24 + z**5/120),
    # ("ERK6", lambda z: 1 + z + z**2/2 + z**3/6 + z**4/24 + z**5/120 + z**6/720)

    # --- SYMPLECTIC GAUSS METHODS ---
    ("Point milieu (Gauss2)", lambda z: (2+z)/(2-z)),
    ("Gauss4", lambda z: (12 + 6*z + z**2)/(12 - 6*z + z**2)),
    ("Gauss6", lambda z: (120 + 60*z + 12*z**2 + z**3)/(120 - 60*z + 12*z**2 - z**3)),

    # --- SUB-DIAGONAL PADE APPROXIMANTS TO THE EXPONENTIAL ---
    ("Padé[2,1]", lambda z: (1 + z/3)/(1 - 2/3*z + z**2/6)),
    ("Padé[3,2]", lambda z: (1 + 2/5*z + 1/20*z**2)/(1 - 3/5*z + 3/20*z**2 - 1/60*z**3)),
    ("Padé[2,0]", lambda z: 1 / (1 - z + 1/2*z**2)),
    # ("Pade[3,1]", lambda z: (1 + 1/4*z)/(1 - 3/4*z + 1/4*z**2 - 1/24*z**3)),
    ]

# Coordinates in the complex plane
size = 5.0  # determines xmin, xmax, ymin, ymax
xx, yy = np.meshgrid(np.linspace(-size,size,300), np.linspace(-size,size,300))
zz = xx + 1j*yy
omegas = np.linspace(-6,6,100)

#%% Plot the stability function in the complex plane

# Create the subplots
nx = 3
ny = 3
fig, axs = plt.subplots(ny, nx, figsize=(nx*3,ny*3), sharex='row', sharey='row')

for i, (name, Rz) in enumerate(schemas):
    ax = axs[i//nx][i%nx]
    # Use a logarithmic scale for the stability function
    data = np.log10(np.abs(Rz(zz)))
    # Show it using a red-blue colormap
    ax.imshow(data, cmap="RdBu_r", origin='lower', extent=(-size, size, -size, size))
    # Show the level set where data=0 i.e. |R(z)| = 1
    ax.contour(xx, yy, data, [0], cmap="gray")
    # Thin black lines for the axes
    ax.axhline(0, c='k', linewidth=0.5)
    ax.axvline(0, c='k', linewidth=0.5)
    ax.set_title(name)

# Save the figure
plt.tight_layout()
plt.savefig("Chap4_2_stability_regions.png", dpi=300)
