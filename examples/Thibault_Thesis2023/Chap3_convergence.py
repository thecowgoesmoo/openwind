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

Compare the space-time convergence of the EXP and IMPEXP schemes on the
"simple instrument" example.

Make sure you run Chap1_2_convergence.py first, as this script relies
on the results of the EXP simulation.

This script generates Figure 3.3 of the thesis.
"""

import numpy as np
import matplotlib.pyplot as plt

from simple_instrument_common import (calculate_solution, l_eles,
                                      calc_error)
from Chap1_2_convergence import run_simulations_EXP, calc_errs_EXP


def run_simulations_IMPEXP():
    for l_ele in l_eles:
        try:
            np.load(f"IMPEXP_{l_ele:.3g}_ts.npy")
            print(f"--- REUSING the solution computed for l_ele={l_ele:.3g}")
        except FileNotFoundError:
            rec = calculate_solution(l_ele, assembled_toneholes=True)
            np.save(f"IMPEXP_{l_ele:.3g}_ts.npy", rec.ts)
            # np.save(f"{l_ele:.3g}_P_interp.npy", rec.values["P_interp"])
            # np.save(f"{l_ele:.3g}_V_interp.npy", rec.values["V_interp"])
            # np.save(f"{l_ele:.3g}_gradP_interp.npy", rec.values["gradP_interp"])
            print("Wrote binary files!", f"IMPEXP_{l_ele:.3g}_XXX.npy")


def calc_errs_IMPEXP():
    errs_IMPEXP = []
    dts_IMPEXP = []
    for k in range(len(l_eles) - 1):
        print("Computing error for k =", k, "; l_ele =", l_eles[k])
        # Calculate the sum of the relative errors on P, V, gradP
        err_k = 0
        for (field, dt_shift) in [("P", 0.5), ("V", 1),
                                  ("gradP", 0.5)
                                  ]:
            err_field = calc_error(
                k, field, dt_shift, assembled_toneholes=True)
            print("Error on", field, "is", err_field)
            err_k += err_field
        errs_IMPEXP.append(err_k)
        dts_IMPEXP.append(np.load(f"IMPEXP_{l_eles[k]:.3g}_ts.npy")[0]*2)
    return errs_IMPEXP, dts_IMPEXP


def plot_spacetime_convergence_both(errs, dts, errs_IMPEXP, dts_IMPEXP):
    plt.figure(figsize=(4, 2.6))
    plt.loglog(dts[:], errs[:], "s--", label="EXP")
    plt.loglog(dts_IMPEXP[:], errs_IMPEXP[:], "d--", label="IMPEXP")
    plt.legend()
    plt.grid(True, "major")
    plt.grid(True, "minor", alpha=0.5)
    plt.xlabel("Time step $\Delta t$ (s)")
    plt.ylabel("Relative $L^\\infty(H^1 \\times L^2)$ error\n on $P,V$")

    # Display a triangle with 2:1 slope
    plt.plot([1e-6, 2e-6, 2e-6, 1e-6], [1e-4, 1e-4, 4e-4, 1e-4], "k")
    plt.text(1.5e-6, 0.8e-4, "1", ha="center", va="top")
    plt.text(2.2e-6, 2e-4, "2", ha="left", va="center")
    plt.tight_layout()
    plt.savefig("simple_instrument_convergence_IMPEXP.png", dpi=300)


if __name__ == "__main__":
    run_simulations_EXP()
    run_simulations_IMPEXP()
    errs, dts = calc_errs_EXP()
    errs_IMPEXP, dts_IMPEXP = calc_errs_IMPEXP()
    plot_spacetime_convergence_both(errs, dts, errs_IMPEXP, dts_IMPEXP)
