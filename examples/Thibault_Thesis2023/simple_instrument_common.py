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
This file is a set of utility functions used to test the convergence of the
numerical methods of Alexis THIBAULT's Ph.D. thesis on one specific example.

The functions are used in:
    `simple_instrument_convergence.py`
    `simple_instrument_convergence_IMPEXP.py`
"""

from os import system
from functools import lru_cache

import numpy as np
from numpy.linalg import norm as npnorm
import h5py

from openwind import Player, simulate

# %% Geometry and reed parameters of the simplified instrument

instrument = [[0.0, 300e-3, 5e-3, 5e-3, 'linear'],
              [300e-3, 500e-3, 5e-3, 50e-3, 'bessel', 0.7]]
holes = [['x', 'l', 'r', 'label'],
         [450e-3, 15e-3, 5e-3, 'hole1']]
player = Player('CLARINET')
player.update_curve("width", 2e-2)

# %% Define how to run the simulation for a given set of parameters

duration = 0.2
 # THIS IS THE FACTOR IN FRONT OF THE CFL
 # CHANGE THIS TO 0.9 TO GENERATE THE OTHER CURVE OF FIG 1.4
cfl_alpha = 1.0-1e-6 # <<<---

@lru_cache(256)
def calculate_solution(l_ele=0.01, order=4, interp_grid=0.001,
                       assembled_toneholes=False):
    print("="*80)
    print(f"CALCULATING SOLUTION FOR l_ele={l_ele}, order={order}")
    print("="*80)
    print()

    hdf5_file = f"IMPEXP_{l_ele:.3g}.hdf5" if assembled_toneholes else f"{l_ele:.3g}.hdf5"
    rec = simulate(duration,
                   instrument,
                   holes,
                   player=player,
                   losses=False,
                   temperature=20,
                   l_ele=l_ele, order=order,  # Discretization parameters
                   record_energy=False,
                   interp_grid=interp_grid,
                   hdf5_file=hdf5_file,
                   cfl_alpha=cfl_alpha,
                   assembled_toneholes=assembled_toneholes,
                   )
    # show the discretization infos
    rec.t_solver.discretization_infos()
    return rec


def clean_folder():
    """ Clean the current folder by deleting all saved simulation results. """
    system("rm *.npy *.hdf5")


# %% Calculate the solutions for various element lengths


l_eles = [
    0.02,
    0.015,
    0.01,
    0.005, 0.003, 0.002,
    0.0015, 0.001,
    0.0005,
]

# %% Compute space-time convergence


def interp_dataset(t, ts, dset):
    """Linearly interpolate the dataset at time t"""
    assert isinstance(t, float)
    i = np.searchsorted(ts, t)
    if i == 0 or i == len(ts):
        raise ValueError("t must be between min(ts) and max(ts)")
    t0 = ts[i-1]
    t1 = ts[i]
    data0 = dset[i-1, ...]
    data1 = dset[i, ...]
    x = (t - t0) / (t1 - t0)
    return (1-x) * data0 + x * data1


def calc_error(k, field, dt_shift, ord_time=np.inf, ord_space=2,
               assembled_toneholes=False):
    """Calculate the relative error between the k-th solution and the reference solution
    on the given field (P, V, gradP)

    dt_shift: how much we should shift the time grid for the given field.
    By default rec.ts gives (n+1/2)*dt for n = 0 ... N-1
    But P_interp is recorded at times (n+1)*dt    -> dt_shift = 0.5
    and V_interp is recorder at times (n+3/2)*dt  -> dt_shift = 1

    """
    try:
        ts_ref0 = np.load(f"{l_eles[-1]:.3g}_ts.npy")
        dt_ref = 2*ts_ref0[0]

        prefix = "IMPEXP_" if assembled_toneholes else ""
        ts_k0 = np.load(prefix + f"{l_eles[k]:.3g}_ts.npy")
        dt_k = 2*ts_k0[0]

        # Linear interpolation in time
        # field_k = np.load(f"{l_eles[k]:.3g}_{field}_interp.npy")
        # field_k_interp = interp1d(ts_k + dt_shift*dt_k, field_k, axis=0)(ts_ref[20:-20] + dt_shift*dt_ref)
        # Drop the first and last 10 points to avoid interpolating before the first sample on the other solutions
        # field_ref = np.load(f"{l_eles[-1]:.3g}_{field}_interp.npy")[20:-20]
        # return np.sqrt(np.sum(abs(field_ref - field_k_interp)**2) / np.sum(abs(field_ref)**2))

        ts_k = ts_k0 + dt_shift*dt_k
        ts_ref = ts_ref0 + dt_shift*dt_ref

        # Calculate the solution at t_interp, for a few regularly spaced time steps,
        # excluding the very beginning and the very end to avoid interpolation issues
        # tt_interp = np.linspace(0.01*duration, 0.99*duration, 100)
        tt_interp = np.linspace(0.001, 0.199, 100)

        dset_k = h5py.File(prefix + f"{l_eles[k]:.3g}.hdf5")[f"{field}_interp"]
        print("Opening file", f"{l_eles[-1]:.3g}.hdf5")
        dset_ref = h5py.File(f"{l_eles[-1]:.3g}.hdf5")[f"{field}_interp"]
        norm_err_tt = []
        norm_ref_tt = []
        for t_interp in tt_interp:
            field_ref = interp_dataset(t_interp, ts_ref, dset_ref)
            field_k = interp_dataset(t_interp, ts_k, dset_k)
            norm_err_tt.append(npnorm(field_ref - field_k, ord_space))
            norm_ref_tt.append(npnorm(field_ref, ord_space))

        norm_err = npnorm(norm_err_tt, ord_time)
        norm_ref = npnorm(norm_ref_tt, ord_time)
        return norm_err / norm_ref
    except FileNotFoundError as e:
        print("ERROR: Could not open the file containing simulation results. "
              "Did you run the simulations in the right order?")
        raise e
