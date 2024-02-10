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

Show on a toy problem how the fourth-order Gauss method can be interpolated
using a degree 2 polynomial.

This script generates Fig. 4.1a and 4.1b of the thesis.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt

def source(t):
    return np.sin(t) * np.cos(1.62*t) * np.exp(-0.01*t*t)

A = -1 # cas scalaire
X0 = -1
I = 1
c_0 = 1/2 - sqrt(3)/6
c_1 = 1/2 + sqrt(3)/6

def run_scheme_gauss4_interp(A, X0, T, n_steps, n_substeps=10):
    """Gauss4 scheme.

    Oversampled to show how to interpolate using a degree 2 polynomial.
    """
    dt = T/n_steps

    C = dt*A

    Xn = X0
    XX = [X0]
    tt = [0]

    LHS = I - C/2 + C**2/12
    ILHS = LHS**(-1)

    t = 0
    t_subs = np.arange(1, n_substeps+1) / n_substeps # time of substeps, in [0,dt]
    for n in range(n_steps):
        # Calculate source term
        Ftc0 = dt*source(t + c_0*dt)
        Ftc1 = dt*source(t + c_1*dt)

        # Calculate coefficients of the polynomial interpolation
        Xt0 = Xn
        # Xt1 = ILHS * ((C-C**2/2)*Xn + c_1*(I-C/2)*Ftc0 + c_0*(I-C/2)*Ftc1)
        # Xt2 = ILHS * (C**2/2 * Xn + ((1+sqrt(3))/4*C - sqrt(3)/2*I)*Ftc0 + ((1-sqrt(3))/4*C + sqrt(3)/2*I) * Ftc1)

        Xt1 = ILHS * (- 1/2 * C**2*Xn
                        + C * (Xn
                              - (1/4 + sqrt(3)/6) * Ftc0
                              - (1/4 - sqrt(3)/6) * Ftc1)
                        + (1+sqrt(3))/2*Ftc0 + (1-sqrt(3))/2*Ftc1
                        )

        Xt2 = ILHS * (1/2 * C**2 * Xn
                      + C * ((1 + sqrt(3))/4 * Ftc0 + (1 - sqrt(3))/4 * Ftc1)
                      + sqrt(3)/2 * (Ftc1-Ftc0))


        # ------ DEBUG ------
        res0 = (Xt1 + 2*Xt2*c_0) - C*(Xt0 + Xt1*c_0 + Xt2*c_0**2) - Ftc0
        res1 = (Xt1 + 2*Xt2*c_1) - C*(Xt0 + Xt1*c_1 + Xt2*c_1**2) - Ftc1

        # print("ILHS", ILHS)
        # print("Xt0",Xt0, "Xt1",Xt1,"Xt2",Xt2)
        # print("C",C)
        # print("(Xt1 + 2*Xt2*c_0)", (Xt1 + 2*Xt2*c_0), "(Xt1 + 2*Xt2*c_1)", (Xt1 + 2*Xt2*c_1))
        # print("Ftc0",Ftc0,"Ftc1",Ftc1)
        # print("res0", res0, "res1", res1)
        assert abs(res0) < 1e-10, "Schéma faux"
        assert abs(res1) < 1e-10, "Schéma faux"
        # ------ END DEBUG ------


        for t_sub in t_subs:
            Xnpt = Xt0 + Xt1*t_sub + Xt2*t_sub**2
            XX.append(Xnpt)
            tt.append(t + t_sub*dt)

        Xnp1 = Xnpt
        t += dt
        Xn = Xnp1

    return Xn, XX, tt

#%%

duration = 10

XX_ref = None
nn_steps = [200,
            5,
            10,
            20,
            50,
            100
            ]
errs_interp = []
errs_nointerp = []

# fig, axs = plt.subplots(1, 2, figsize=(8,2.6))
# fig, axs = plt.subplots(1, 2, figsize=(8,2.6))
fig0 = plt.figure(figsize=(4,2.6))
ax0 = plt.gca()
fig1 = plt.figure(figsize=(4,2.6))
ax1 = plt.gca()
axs = [ax0, ax1]

for i, (n_steps, style) in enumerate(zip(nn_steps, ['--k','-o','-x','--+', '-.', ':'])):
    Xn, XX, tt = run_scheme_gauss4_interp(A, X0, duration, n_steps, n_substeps=10)
    XX = np.array(XX)
    axs[0].plot(tt, XX, style, markevery=10, label=str(n_steps))

    if XX_ref is None:
        XX_ref = XX
    else:
        skip = nn_steps[0] / n_steps
        assert skip == int(skip), "Must use congruent time steps"
        indices = np.arange(0, 10*nn_steps[0]+1, int(skip))
        # assert np.max(abs(tts[-1][indices] - tts[i])) < 1e-12, "Time steps do not match"
        XX_exact = XX_ref[indices]
        axs[1].semilogy(tt, abs(XX - XX_exact), style, markevery=10)
        err_nointerp = np.max(abs(XX[::10] - XX_exact[::10]))
        errs_nointerp.append(err_nointerp)
        err_interp = np.max(abs(XX - XX_exact))
        errs_interp.append(err_interp)


axs[0].legend()
axs[0].grid(True, 'both')
axs[0].set_xlabel("t")
axs[0].set_ylabel("u(t)")
axs[0].set_title("Gauss4 Interpolated solution")

axs[1].grid(True, 'both')
axs[1].set_ylim((1e-8,1))
axs[1].set_xlabel("t")
axs[1].set_ylabel("$|u(t) - u_{ref}(t)|$")
axs[1].set_title("Gauss4 error on interpolated solution")

fig0.tight_layout()
fig0.savefig("Chap4_gauss_interpolation_demo_0.png")
fig1.tight_layout()
fig1.savefig("Chap4_gauss_interpolation_demo_1.png")


#%% Convergence of interpolated solution

dts = duration / np.array(nn_steps[1:])
plt.figure(figsize=(4,2.6))
plt.loglog(dts, errs_interp, "o--", label="interpolated")
plt.loglog(dts, (dts/dts[0])**3 * errs_interp[0], ":r", label="dt^3")
plt.loglog(dts, errs_nointerp, "o--", label="on time steps")
plt.loglog(dts, (dts/dts[0])**4 * errs_nointerp[0], ":k", label="dt^4")
plt.xlabel("dt")
plt.ylabel("Consecutive $L^\\infty$ error on interpolated solution")
plt.title("Gauss4 interpolation convergence")
plt.grid(True, "both")
plt.legend()

plt.tight_layout()
plt.savefig("Chap4_gauss_interpolation_demo_convergence.png")
