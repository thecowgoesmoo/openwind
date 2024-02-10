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

Plot in the Laplace domain $s \in \mathbb{C}$ the operators Z_v and Y_t
and their poles and zeros.

This script generates Figures 8.1a and 8.1b of the thesis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jve, jn_zeros



xrange = (-200,200)
yrange = (-100,100)
reals = np.linspace(*xrange, 200)
imags = np.linspace(*yrange, 200)
rr, ii = np.meshgrid(reals, imags)

zz = rr + 1j*ii
qq = 1j * np.sqrt(zz)

# alpha = -iomega
# i32stokes = np.sqrt(alpha)

# Kv_cyl = 2*jve(1, i32stokes) / (i32stokes * jve(0, i32stokes))

# Zv_cyl = iomega / (1 - Kv_cyl)
# Yt_cyl = iomega * (1 + 0.4*Kv_cyl)

Ztilde = - jve(0, qq) / jve(2, qq)
Ytilde = 2*jve(1, qq) / (qq * jve(0, qq))



nz_ = 5 # Number of zeros to display
np_ = 5 # Number of poles to display
Zv_zeros = -jn_zeros(0, nz_)**2
Zv_poles = np.concatenate([[0], -jn_zeros(2, np_)**2])
Yt_zeros = -jn_zeros(1, nz_)**2
Yt_poles = -jn_zeros(0, np_)**2


#%% Plot magnitude of |Zv - 1|

# plt.figure()
# plt.contourf(rr, ii, abs(Ztilde-1), levels=np.linspace(0,1))

#%% Plot results

def plot_immitance(H, label="K_v", figname="immitances_laplace.png",
                   zeros=np.array([]), poles=np.array([]) ):
    fig, (ax1, axBar1,ax2,axBar2) = plt.subplots(1, 4, figsize=(6.2,2.5),
                                                 gridspec_kw={'width_ratios':[1,0.06,1,0.06],
                                                              'wspace':0.01},
                                                 constrained_layout=True)
    logGain = np.log10(np.abs(H))
    levelsGain = np.linspace(np.min(logGain),np.max(logGain),255)
    ticksGain = np.arange(np.ceil(np.min(logGain)),np.ceil(np.max(logGain)), dtype=int)
    con0 = ax1.contourf(rr, ii, logGain, levels=levelsGain, cmap='gray')

    # Add a colorbar to the magnitude plot
    cbar = fig.colorbar(con0, cax=axBar1, ticks=ticksGain)
    axBar1.set_yticklabels([f'{10.0**v}' for v in ticksGain])


    # ax1.contour(rr, ii, np.log(np.abs(H)), levels=np.linspace(-5,5,10), cmap='hsv')
    con1 = ax2.contourf(rr, ii, np.angle(H), levels=np.linspace(-np.pi,np.pi,256), cmap='hsv')
    con2 = ax2.contour(rr, ii, np.angle(H), levels=np.linspace(-np.pi,np.pi,5), cmap='gray')
    ax2.axvline(0, linestyle='dashed', color='k')

    ax1.plot(zeros.real, zeros.imag, "o", mfc="None", mec="r")
    ax1.plot(poles.real, poles.imag, "xr")
    ax2.plot(zeros.real, zeros.imag, "o", mfc="None", mec="k")
    ax2.plot(poles.real, poles.imag, "xk")


    ax1.set_title("Module $|"+label+"({s})|$")
    ax1.set_xlabel('$Re({s})$')
    ax2.set_title("Argument $arg("+label+"({s}))$")
    ax2.set_xlabel('$Re({s})$')
    ax1.set_ylabel('$Im({s})$')

    # Annotate a point where Re(H) changes sign
    ipt = int(len(rr) / 6)
    jpt = np.argmax(np.real(H[ipt, :]) > 0)

    if jpt > 0:
        ypt = imags[ipt]; xpt = reals[jpt]
        print("xpt, ypt = ", xpt, ypt)
        ax2.annotate(f"$Re({label}) < 0$", (xpt, ypt),
                     xytext=(-5,0), textcoords='offset points',
                     horizontalalignment='right',
                     bbox=dict(boxstyle="round", fc="1.0", lw=0),
                     fontsize='small')
        ax2.annotate(f"$Re({label}) > 0$", (xpt, ypt),
                     xytext=(5,0), textcoords='offset points',
                     horizontalalignment='left',
                     bbox=dict(boxstyle="round", fc="1.0", lw=0),
                     fontsize='small')

    # Add a colorbar to the phase plot
    cbar = fig.colorbar(con1, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi], cax=axBar2)
    fig.colorbar(con2, cax=cbar.ax)
    cbar.ax.set_yticklabels(['$-\pi$','$-\pi/2$', '0','$\pi/2$', '$\pi$'])  # vertically oriented colorbar

    ax1.set_xlim(xrange)
    ax1.set_ylim(yrange)
    ax2.set_xlim(xrange)
    ax2.set_ylim(yrange)

    # fig.tight_layout()
    fig.savefig(figname, dpi=300)


plot_immitance(Ztilde, label=r"\tilde{Z}", figname="immitances_laplace_Ztilde.png",
               zeros=Zv_zeros, poles=Zv_poles)
plot_immitance(Ytilde, label=r"\tilde{Y}", figname="immitances_laplace_Ytilde.png",
               zeros=Yt_zeros, poles=Yt_poles)
