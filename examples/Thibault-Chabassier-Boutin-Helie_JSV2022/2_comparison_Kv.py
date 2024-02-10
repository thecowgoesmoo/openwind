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

Calculate the acoustic loss coefficients in conical pipes using different models.

This script generates Figure 4 of the article.
"""

import numpy as np


import mpmath as mp
import numpy as np
from numpy import cos, sin
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import jv, jve



hyp2f1 = np.frompyfunc(
    lambda a, b, c, z: mp.hyp2f1(a, b, c, z, maxterms=10**5),
    4, 1)

def legendreP(ell, z):
    return hyp2f1(-ell, ell+1, 1, (1-z)/2)
def legendreP_prime(ell, z):
    return ell*(ell+1)/2 * hyp2f1(-ell+1, ell+2, 2, (1-z)/2)




# for conv in ['HR']:
for conv in ['ARC', 'ALT', 'HR']:
    fig, (sub3,sub4,sub1) = plt.subplots(1,3, figsize=(9,3))
    # Stokes number
    s = np.geomspace(0.3, 100, 100)
    # Loss coefficient of a cylinder
    i32s = 1j**(3/2) * s
    Kv_cyl = 2*jve(1, i32s) / (i32s * jve(0, i32s))
    # Plot loss coef for several cone angles
    for Theta, linestyle in zip([np.pi/2, np.pi/4, np.pi/10],
                                ['-','--','-.']):
        cThetas = {
            'ARC': Theta,       # (ARC) Radius defined by arclength
            'ALT': sin(Theta),   # (ALT) Geometric radius (altitude)
            'HR': 2*(1-cos(Theta))/sin(Theta)
                                # (HR) Hydraulic radius
        }

        cTheta = cThetas[conv]

        # Complex order of the spherical harmonics
        eta = np.sqrt(-1j*s**2/cTheta**2 + 1/4) - 1/2


        # Complex order of Legendre functions
        # eta = np.sqrt(-1j*(s*hr_gr_ratio)**2/Theta**2 + 1/4) - 1/2
        #eta = np.sqrt(-1j*(s)**2/Theta**2 + 1/4) - 1/2
        # Loss coefficient of cone
        Kv = sin(Theta)**2 / (1 - cos(Theta)) \
        * legendreP_prime(eta, cos(Theta)) \
            / (eta * (eta+1) * legendreP(eta, cos(Theta)))
        # Convert mpc object array to complex array
        Kv = np.array(Kv, dtype=complex)
        ratio = Kv / Kv_cyl

        #
        line, = sub1.loglog(s, np.abs(ratio-1), linestyle, linewidth=1)
        # sub1.axhline(hr_gr_ratio, ls='--')
        # sub2.semilogx(s, np.angle(ratio), '-', color=line.get_color(),
        #               label=f'$\Theta = \pi/{np.pi/Theta:g}$')

        line, = sub3.loglog(s, abs(Kv), linestyle, linewidth=1)
        sub4.semilogx(s, np.angle(Kv), linestyle, linewidth=1, color=line.get_color(), label=f'$\Theta = \pi/{np.pi/Theta:g}$')


    # Plot the loss coefficient of a cylinder
    sub3.loglog(s, abs(Kv_cyl), ':k')
    sub4.semilogx(s, np.angle(Kv_cyl), ':k', label='cylinder')

    sub1.set_title('Relative error')
    sub1.set_xlabel('$\mathrm{St}_{%s}$' % conv.lower())
    sub1.set_ylabel('$|K_{v,cyl} / K_{v,\Theta,%s} - 1|$' % conv.lower())
    sub1.grid('both')

    sub3.set_title('Magnitude')
    sub3.grid('both')
    sub3.set_xlabel('$\mathrm{St}_{%s}$' % conv.lower())
    sub3.set_ylabel('$|K_v(\mathrm{St}_{%s})|$' % conv.lower())

    sub4.set_title('Phase')
    sub4.set_xlabel('$\mathrm{St}_{%s}$' % conv.lower())
    sub4.set_ylabel('$arg(K_v(\mathrm{St}_{%s}))$' % conv.lower())
    sub4.set_yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    sub4.set_yticklabels(['$-\pi/2$','$-\pi/4$','0','$\pi/4$', '$\pi/2$'])
    sub4.grid('both')
    sub4.legend()

    fig.tight_layout()
    plt.savefig(f"Chap5_2_comparison_Kv_{conv}.png", dpi=300)

