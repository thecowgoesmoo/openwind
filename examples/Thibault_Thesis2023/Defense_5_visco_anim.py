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
from scipy.special import jv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickFileWriter

k = 1.5
omega = 2*np.pi
xx = np.linspace(0,1, 20)
xx_v = np.linspace(0,1,10)
R = 0.2
rr = np.linspace(-R,R,100)
rr_v = np.linspace(-R,R,20)
stokes = 10
prandtl = 0.71

pp = np.exp(-1j*k*xx)
alpha_v = -1j*stokes**2
alpha_t = -1j*stokes**2*prandtl
psi_v = 1 - jv(0, np.sqrt(alpha_v)*rr_v/R) / jv(0, np.sqrt(alpha_v))
psi_t = 1 - jv(0, np.sqrt(alpha_t)*rr/R) / jv(0, np.sqrt(alpha_t))
# Velocity field
vv = -1j*np.exp(-1j*k*xx_v) * psi_v[:,np.newaxis]
# Temperature field
tt = np.exp(-1j*k*xx) * psi_t[:,np.newaxis]


fig = plt.figure(figsize=(4,2.6))
# Plot the pressure, velocity and temperature fields
temp_c = plt.contourf(xx, rr, tt.real, cmap="magma", vmin=-1, vmax=1, levels=np.linspace(-1,1, 10))
velo_q = plt.quiver(xx_v, rr_v, vv.real, 0*vv.real, scale=10, color="C0")

plt.axis('equal')
plt.gca().set_axis_off()
plt.tight_layout()

n_frames = 60
def next_frame(frame):
    global temp_c, velo_q
    t = frame/n_frames
    phase = np.exp(1j*omega*t)
    for c in temp_c.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    temp_c = plt.contourf(xx, rr, (tt*phase).real, cmap="magma", vmin=-1, vmax=1, levels=np.linspace(-1.2,1.2, 10), zorder=-1)
    # temp_c.set_array((tt * phase).real)
    velo_q.set_UVC((vv*phase).real, 0*vv.real)
    return temp_c, velo_q

anim = FuncAnimation(fig, next_frame, n_frames, interval=60)

fps = 15
f = "visco_anim.gif"
writergif = ImageMagickFileWriter(fps=fps)
print(f"Saving animation to {f}...")
anim.save(f, writer=writergif, dpi=200)
