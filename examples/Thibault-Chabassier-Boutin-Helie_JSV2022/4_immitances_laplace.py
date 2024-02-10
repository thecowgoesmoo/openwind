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
This script is part of the numerical examples accompanying the article:
Thibault, A., Chabassier, J., Boutin, H., & Hélie, T. (2023).
Transmission line coefficients for viscothermal acoustics in conical tubes.
Journal of Sound and Vibration, 543, 117355.

Plot in the Laplace domain $s \in \mathbb{C}$ the operators Z_v and Y_t
corresponding to different models of losses.

This script generates Figures 7a and 7b of the article.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jve

import mpmath as mp


xrange = (-100,100)
yrange = (-100, 100)
reals = np.linspace(*xrange, 200)
imags = np.linspace(*yrange, 200)
rr, ii = np.meshgrid(reals, imags)

iomega = rr + 1j*ii
alpha = -iomega
i32stokes = np.sqrt(alpha)

Kv_cyl = 2*jve(1, i32stokes) / (i32stokes * jve(0, i32stokes))

Zv_cyl = iomega / (1 - Kv_cyl)
Yt_cyl = iomega * (1 + 0.4*Kv_cyl)



#%% Plot results

def plot_immitance(H, label="K_v", figname="immitances_laplace.png"):
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
    axBar1.set_yticklabels([f'{10**v}' for v in ticksGain])


    # ax1.contour(rr, ii, np.log(np.abs(H)), levels=np.linspace(-5,5,10), cmap='hsv')
    con1 = ax2.contourf(rr, ii, np.angle(H), levels=np.linspace(-np.pi,np.pi,256), cmap='hsv')
    con2 = ax2.contour(rr, ii, np.angle(H), levels=np.linspace(-np.pi,np.pi,5), cmap='gray')
    ax2.axvline(0, linestyle='dashed', color='k')

    ax1.set_title("Magnitude $|"+label+"({s})|$")
    ax1.set_xlabel('$Re({s})$')
    ax2.set_title(f"Angle $arg("+label+"({s}))$")
    ax2.set_xlabel('$Re({s})$')
    ax1.set_ylabel('$Im({s})$')

    # Annotate a point where Re(H) changes sign
    ipt = int(len(rr) / 6)
    jpt = np.argmax(np.real(H[ipt, :]) > 0)
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

    # ax1.axis('equal')
    # ax2.axis('equal')
    ax1.set_xlim((-100,100))

    # fig.tight_layout()

    # fig.savefig(figname, dpi=300)


plot_immitance(Zv_cyl, label="Z_v", figname="Chap5_3_immitances_laplace_Zv_cyl.png")

#%% Same for cone

#-----------------------------------------------
# Calcul pour le cône

hyp2f1 = np.frompyfunc(
    lambda a, b, c, z: mp.hyp2f1(a, b, c, z, maxterms=10**5),
    4, 1)

def legendreP(ell, z):
    return hyp2f1(-ell, ell+1, 1, (1-z)/2)
def legendreP_prime(ell, z):
    return ell*(ell+1)/2 * hyp2f1(-ell+1, ell+2, 2, (1-z)/2)

Theta = np.pi/4
cTheta = 1/(np.sin(Theta)/(2*(1-np.cos(Theta))))
# Complex order of the spherical harmonics
eta = np.sqrt(i32stokes**2/cTheta**2 + 1/4) - 1/2
# Complex order of Legendre functions
# eta = np.sqrt(-1j*(s*hr_gr_ratio)**2/Theta**2 + 1/4) - 1/2
#eta = np.sqrt(-1j*(s)**2/Theta**2 + 1/4) - 1/2
# Loss coefficient of cone
Kv = np.sin(Theta)**2 / (1 - np.cos(Theta)) \
* legendreP_prime(eta, np.cos(Theta)) \
    / (eta * (eta+1) * legendreP(eta, np.cos(Theta)))
# Convert mpc object array to complex array
Kv_cone = np.array(Kv, dtype=complex)
#----------------------------------------------------


#%% Plot cone results

Zv_cone = iomega / (1 - Kv_cone)
Yt_cone = iomega * (1 + 0.4*Kv_cone)

plot_immitance(Zv_cone, label='Z_v', figname="Chap5_3_immitances_laplace_Zv_cone_pis4.png")
plot_immitance(Yt_cone, label='Y_t', figname="Chap5_3_immitances_laplace_Yt_cone_pis4.png")
