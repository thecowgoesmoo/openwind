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

Run convergence tests on the coupled Gauss4 method and compare it
to the coupled Leap-Frog scheme.

This script generates Figures 4.5 and 4.6 of the thesis.
"""
import time

import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt

from openwind.temporal import TemporalSolver

# Use the numerical method defined in the previous script
from Chap4_3_coupled_gauss4 import (run_scheme_gauss4_coupled, physics, rayon,
                                    instr_physics, longueur1, longueur2,
                                    bump, bump_c, bump_w)


# %% Construction des matrices d'éléments finis de nos tuyaux

celerity = physics.c(0)
Zc = physics.rho(0) * celerity / (np.pi * rayon**2)


# %% Convergence curve --- computations for Gauss4


duration = 0.001
nn_steps = np.array([
    # 10,
    20, 50,
    100,
    200, 500,
    1000,
    2000,
    5000,
    # 10000, # CRASH
    # 20000
])
P_1s = []
P_2s = []
xP_1s = []
xP_2s = []
dts = duration / nn_steps
order = 6        # en espace
# 1 élément = distance parcourue en order pas de temps
l_eles = celerity * dts * order
maxerrs = []
process_times = []
for l_ele, n_steps in zip(l_eles, nn_steps):
    print("*"*50)
    print(f"Running coupled Gauss4, n_steps={n_steps}, l_ele={l_ele}")
    print("*"*50)
    t1 = time.process_time()  # CPU time only, does not include time elapsed during sleep
    res = run_scheme_gauss4_coupled(l_ele=l_ele, order=order, duration=duration, n_steps=n_steps,
                                    plot_final_result=False)
    t2 = time.process_time()
    print("Process time =", t2 - t1)
    process_times.append(t2 - t1)
    errs = res["errs"]
    maxerrs.append(np.max(errs))

    P_1s.append(res["getP_1"] @ res["X_1_n"])
    P_2s.append(res["getP_2"] @ res["X_2_n"])
    xP_1s.append(res["xP_1"])
    xP_2s.append(res["xP_2"])


# %% Run leap-frog with the same space discretization

maxerrs_LF = []
process_times_LF = []
dts_LF = []
l_eles_LF = [0.1, 0.05, 0.02, 0.01, 0.005,
             0.002, 0.001, 0.0005, 0.0002, 0.0001]


def init_t_solver(l_ele):
    tsolver = TemporalSolver(instr_physics,
                             cfl_alpha=1.0-1e-6,
                             # cfl_alpha=0.9,
                             l_ele=l_ele, order=order)

    n_steps = int(np.ceil(duration / tsolver.get_dt()))
    new_dt = duration / n_steps
    # Change dt so that the simulation lasts exactly `duration`.
    tsolver._set_dt(new_dt)

    tpipe1 = tsolver.t_pipes['bore0']
    tpipe2 = tsolver.t_pipes['bore1']
    xP_1 = tpipe1.mesh.get_xH1()*longueur1
    xU_1 = tpipe1.mesh.get_xL2()*longueur1
    nU_2 = tpipe2.nL2
    nP_2 = tpipe2.nH1
    mU_1, mP_1 = tpipe1.get_mass_matrices()
    mU_2, mP_2 = tpipe2.get_mass_matrices()

    P_1_init = bump((xP_1 - bump_c) / bump_w)
    U_1_init = bump((xU_1 - bump_c) / bump_w) / Zc
    P_2_init = zeros(nP_2)
    U_2_init = zeros(nU_2)

    tpipe1.set_P0_V0(P_1_init, U_1_init)
    tpipe2.set_P0_V0(P_2_init, U_2_init)

    return tsolver, n_steps


def calc_err_from_tsolver(tsolver):
    tpipe1 = tsolver.t_pipes['bore0']
    tpipe2 = tsolver.t_pipes['bore1']
    xP_1 = tpipe1.mesh.get_xH1()*longueur1
    xU_1 = tpipe1.mesh.get_xL2()*longueur1
    xP_2 = longueur1 + tpipe2.mesh.get_xH1()*longueur2
    xU_2 = longueur1 + tpipe2.mesh.get_xL2()*longueur2
    mU_1, mP_1 = tpipe1.get_mass_matrices()
    mU_2, mP_2 = tpipe2.get_mass_matrices()
    P1, V1_nph = tpipe1.PV
    V1 = (V1_nph + tpipe1._V_prev) / 2
    X_1_np1 = np.concatenate([V1 * mU_1**0.5, P1 * mP_1**0.5])
    P2, V2_nph = tpipe2.PV
    V2 = (V2_nph + tpipe2._V_prev) / 2
    X_2_np1 = np.concatenate([V2 * mU_2**0.5, P2 * mP_2**0.5])

    def exact_sol(t):
        """Return U_1, P_1, U_2, P_2 at time t (assuming no reflection)"""
        P_1_t = bump((xP_1 - celerity*t - bump_c) / bump_w)
        U_1_t = bump((xU_1 - celerity*t - bump_c) / bump_w) / Zc
        P_2_t = bump((xP_2 - celerity*t - bump_c) / bump_w)
        U_2_t = bump((xU_2 - celerity*t - bump_c) / bump_w) / Zc
        return U_1_t, P_1_t, U_2_t, P_2_t

    U_1x, P_1x, U_2x, P_2x = exact_sol(tsolver.get_current_time())

    plt.plot(xP_1, P1)
    plt.plot(xP_2, P2)
    plt.plot(xP_1, P_1x, ':k')
    plt.plot(xP_2, P_2x, ':k')

    X_1x = np.concatenate([U_1x * mU_1**0.5, P_1x * mP_1**0.5])
    X_2x = np.concatenate([U_2x * mU_2**0.5, P_2x * mP_2**0.5])
    err_1 = X_1x - X_1_np1
    err_2 = X_2x - X_2_np1
    nrj0 = X_1x @ X_1x + X_2x @ X_2x
    nrj_err = err_1 @ err_1 + err_2 @ err_2
    return np.sqrt(nrj_err / nrj0)


for l_ele in l_eles_LF:
    print("*"*50)
    print(f"Running leap-frog, l_ele={l_ele}")
    print("*"*50)
    tsolver, n_steps = init_t_solver(l_ele)
    t1 = time.process_time()  # CPU time only, does not include time elapsed during sleep
    # tsolver.run_simulation(duration)
    tsolver.run_simulation_steps(n_steps)
    t2 = time.process_time()
    print("Process time =", t2 - t1)
    process_times_LF.append(t2 - t1)
    err = calc_err_from_tsolver(tsolver)
    print("Err =", err)
    maxerrs_LF.append(err)
    dts_LF.append(tsolver.get_dt())

# %%
maxerrs_LF = np.array(maxerrs_LF)
dts_LF = np.array(dts_LF)
cv_rate = np.log(maxerrs_LF[1:]/maxerrs_LF[:-1]) / \
    np.log(dts_LF[1:] / dts_LF[:-1])
print("Convergence rates for Leap Frog", cv_rate)


# %%

def plot_triangle(x0, y0, slope, rx=2):
    # Display a triangle with a given slope
    if slope > 0:
        ry = rx**slope  # ratio of y values
        plt.plot([x0, rx*x0, rx*x0, x0], [y0, y0, ry*y0, y0], "k")
        plt.text(x0*rx**0.5, y0*ry**(-0.2), "1", ha="center", va="top")
        plt.text(x0*rx, y0*ry**0.5, f" {slope}", ha="left", va="center")
    else:
        ry = rx**(slope)  # ratio of y values
        plt.plot([x0, rx*x0, rx*x0, x0], [y0, y0, ry*y0, y0], "k")
        plt.text(x0*rx**0.5, y0*ry**(-0.1), "1", ha="center", va="bottom")
        plt.text(x0*rx, y0*ry**0.5, f" {-slope}", ha="left", va="center")

# %% Convergence curve --- show


plt.figure(5, figsize=(4, 3))
plt.loglog(dts, maxerrs, "o--", label="Coupled Gauss4")
plt.loglog(dts_LF, maxerrs_LF, "o--", label="Leap-Frog")
# plt.loglog(dts, maxerrs[0] * (dts/dts[0])**1, ":r", label="$\Delta t^1$")
# plt.loglog(dts, maxerrs[0] * (dts/dts[0])**3, ":r", label="$\Delta t^3$")
# plt.loglog(dts, maxerrs[0] * (dts/dts[0])**4, ":k", label="$\Delta t^4$")
plot_triangle(5e-6, 1e-4, 4)
plot_triangle(2e-6, 1e-3, 2)
plt.title("Space-time convergence")
# plt.xlabel(r"$\Delta t$, with $h = 6\times c \times \Delta t$")
plt.xlabel(r"$\Delta t$")
plt.ylabel("$L^\infty([0,T], L^2([0,L]))$ error\nagainst exact solution")
plt.grid(True, "both")
plt.legend()
plt.tight_layout()
plt.savefig("Chap4_4_convergence_error.png", dpi=300)

# %% Display error as a function of CPU time

plt.figure(6, figsize=(4, 3))
plt.loglog(process_times, maxerrs, "o--", label="Coupled Gauss4")
plt.loglog(process_times_LF, maxerrs_LF, "o--", label="Leap-Frog")
plot_triangle(4e-2, 1e-2, -1.5, rx=4)
plot_triangle(2e0, 2e-4, -3, rx=4)
plt.xlabel("CPU time")
plt.ylabel("Error")
plt.grid(True, 'both')
plt.legend()
plt.tight_layout()
plt.savefig("Chap4_4_convergence_CPU.png", dpi=300)

# %% Compute and print the numerical rates of convergence for each method

maxerrs = np.array(maxerrs)
process_times = np.array(process_times)
process_times_LF = np.array(process_times_LF)
rate_LF = np.log(maxerrs_LF[1:]/maxerrs_LF[0]) / \
    np.log(process_times_LF[1:] / process_times_LF[0])
rate_Gauss = np.log(maxerrs[1:]/maxerrs[:-1]) / \
    np.log(process_times[1:] / process_times[:-1])
print("Rates LF:", rate_LF)
print("Rates Gauss", rate_Gauss)
